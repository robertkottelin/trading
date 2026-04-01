"""
BTC ML Training -- Bearish Models
================================================================================
Trains 2 primary downside prediction models using the existing training dataset.
These are SHORT-signal models — when they fire, they predict BTC will fall.

Targets:
  - target_down_12_0003: BTC falls ≥0.3% in next 60min (12 × 5-min candles)
  - target_down_24_0003: BTC falls ≥0.3% in next 120min (24 × 5-min candles)

Output:
  models/bearish/prod_bear_12_0003_lgb.pkl
  models/bearish/prod_bear_24_0003_lgb.pkl
  models/bearish/production_config_bearish.json

Usage:
  python model_training/train_bearish.py

Runtime: ~30-60 minutes
"""

import gc
import json
import logging
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# PATHS & CONSTANTS
# =============================================================================
VERSION = "bearish_v1"
PROCESSED_DIR = "processed_data"
FEATURES_FILE = os.path.join(PROCESSED_DIR, "btc_training_dataset.parquet")
LOG_DIR = os.path.join("model_training", "logs")
LOG_FILE = os.path.join(LOG_DIR, "train_bearish.log")
MODEL_DIR = os.path.join("models", "bearish")

# Warm-start from existing Optuna params
WARMSTART_FILES = [
    os.path.join("models", "v2_all", "optuna_params.json"),
    os.path.join("models", "v23", "optuna_params_v23.json"),
]

FEE_RT = 0.0006          # 0.06% round-trip taker fees
N_SPLITS = 10
CANDLES_PER_DAY = 288
OPTUNA_TRIALS = 40
N_TOP_FEATURES = 100
NAN_THRESHOLD = 0.50
CORR_THRESHOLD = 0.95
MIN_TRADES_PER_SPLIT = 8
QUALITY_MIN_POSITIVE_SPLITS = 7  # out of 10 — must show edge in 70% of time periods

# The 2 bearish targets to train
BEARISH_TARGETS = [
    ("bear_12_0003", "target_down_12_0003", 12),   # 1h -0.3%
    ("bear_24_0003", "target_down_24_0003", 24),   # 2h -0.3%
]

PROTECTED_FEATURES = {"hour_of_day", "day_of_week", "close", "volume"}

DEFAULT_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "n_estimators": 1000, "learning_rate": 0.01, "num_leaves": 24,
    "max_depth": 5, "min_child_samples": 500, "subsample": 0.6,
    "subsample_freq": 1, "colsample_bytree": 0.4, "reg_alpha": 1.0,
    "reg_lambda": 10.0, "path_smooth": 10.0,
}

log = logging.getLogger("train_bearish")


# =============================================================================
# LOGGING
# =============================================================================
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)


# =============================================================================
# UTILITY FUNCTIONS (copied/adapted from train_v2_all.py)
# =============================================================================
def auc_roc(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    y_true, y_prob = y_true[mask], y_prob[mask]
    if len(y_true) == 0:
        return 0.5
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = auc_val = prev_fpr = prev_tpr = 0.0
    for val in y_sorted:
        if val == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        tpr = tp / n_pos
        auc_val += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr
    return auc_val


def get_feature_cols(df):
    exclude = {"open_time_ms", "timestamp"}
    target_cols = {c for c in df.columns if c.startswith("target_")}
    return sorted(c for c in df.columns if c not in exclude and c not in target_cols)


def train_lgb(X_tr, y_tr, X_va, y_va, params, seed=42):
    p = dict(params)
    p["random_state"] = seed
    p["verbose"] = -1
    p["n_jobs"] = -1
    m = lgb.LGBMClassifier(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return m


def wf_splits(n_total, n_splits, test_frac=0.05):
    test_size = int(n_total * test_frac)
    splits = []
    for s in range(n_splits):
        test_end = n_total - s * test_size
        test_start = test_end - test_size
        if test_start < 10000:
            break
        splits.append((test_start, test_end))
    return splits


def split_data(df_values, y_values, train_end, test_start, test_end, purge, val_frac=0.15):
    n = len(df_values)
    val_size = int(train_end * val_frac)
    val_start = train_end - val_size - purge
    idx = np.arange(n)
    train_mask = idx < val_start
    val_mask = (idx >= val_start + purge) & (idx < train_end)
    test_mask = (idx >= test_start) & (idx < test_end)
    valid_tr = train_mask & ~np.isnan(y_values)
    valid_va = val_mask & ~np.isnan(y_values)
    valid_te = test_mask & ~np.isnan(y_values)
    X_tr, y_tr = df_values[valid_tr], y_values[valid_tr]
    X_va, y_va = df_values[valid_va], y_values[valid_va]
    X_te, y_te = df_values[valid_te], y_values[valid_te]
    if len(X_tr) < 1000 or len(X_va) < 100 or len(X_te) < 100:
        return None
    test_indices = np.where(valid_te)[0]
    return X_tr, y_tr, X_va, y_va, X_te, y_te, test_indices


def backtest_short(close, y_prob, horizon, fee_rt, prob_threshold):
    """Backtest a SHORT signal model.
    When model fires (prob >= threshold), simulate a SHORT trade.
    Net return = -(forward_return) - fee_rt
    A winning trade is one where price falls (forward_return < 0).
    """
    n = len(close)
    trades = []
    for i in range(n - horizon):
        if y_prob[i] >= prob_threshold:
            entry = close[i]
            exit_ = close[i + horizon]
            if entry <= 0:
                continue
            fwd_ret = (exit_ - entry) / entry   # positive = price went UP (loss for short)
            net_ret = -fwd_ret - fee_rt          # short profit = negative of price return
            trades.append({"idx": i, "net_ret": net_ret})

    if not trades:
        return {"n": 0, "net": 0.0, "wr": 0.0, "sharpe": 0.0, "max_dd": 0.0}

    rets = np.array([t["net_ret"] for t in trades])
    n_trades = len(trades)
    net = np.prod(1 + rets) - 1
    wr = (rets > 0).mean()

    # Daily returns
    idxs = np.array([t["idx"] for t in trades])
    n_days = max(1, (idxs.max() - idxs.min()) / CANDLES_PER_DAY)
    total_days = int(n_days) + 1
    daily = np.zeros(total_days)
    for t in trades:
        d = int((t["idx"] - idxs.min()) / CANDLES_PER_DAY)
        if d < total_days:
            daily[d] += t["net_ret"]

    sharpe = (daily.mean() / daily.std()) * np.sqrt(365.25) if daily.std() > 0 else 0.0

    # Max drawdown
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    return {"n": n_trades, "net": net, "wr": wr, "sharpe": sharpe, "max_dd": max_dd}


def compute_split_metrics_short(trades):
    if not trades:
        return {"net": 0.0, "n": 0, "sharpe": 0.0, "wr": 0.0}
    rets = np.array([t["net_ret"] for t in trades])
    net = np.prod(1 + rets) - 1
    wr = (rets > 0).mean()
    idxs = np.array([t["idx"] for t in trades])
    n_days = max(1, (idxs.max() - idxs.min()) / CANDLES_PER_DAY)
    total_days = int(n_days) + 1
    daily = np.zeros(total_days)
    for t in trades:
        d = int((t["idx"] - idxs.min()) / CANDLES_PER_DAY)
        if d < total_days:
            daily[d] += t["net_ret"]
    sharpe = (daily.mean() / daily.std()) * np.sqrt(365.25) if daily.std() > 0 else 0.0
    return {"net": net, "n": len(trades), "sharpe": sharpe, "wr": wr}


# =============================================================================
# PHASE 0: DATA LOAD + PRUNING
# =============================================================================
def phase0_load_and_prune(df, all_feat_cols):
    log.info("")
    log.info("=" * 70)
    log.info("  PHASE 0: DATA LOADING + FEATURE PRUNING")
    log.info("=" * 70)
    log.info(f"  Input: {len(df):,} rows x {len(all_feat_cols)} features")

    nan_fracs = df[all_feat_cols].isna().mean()
    high_nan = nan_fracs[nan_fracs > NAN_THRESHOLD].index.tolist()
    keep = [c for c in all_feat_cols if c not in high_nan]
    log.info(f"  Dropped {len(high_nan)} high-NaN features -> {len(keep)} remain")

    sample_idx = np.arange(0, len(df), 5)
    sample = df.iloc[sample_idx][keep].values.astype(np.float64)
    col_medians = np.nanmedian(sample, axis=0)
    for j in range(sample.shape[1]):
        mask = np.isnan(sample[:, j])
        sample[mask, j] = col_medians[j]

    log.info(f"  Computing correlation matrix ({len(sample_idx):,} x {len(keep)})...")
    means = sample.mean(axis=0)
    stds = sample.std(axis=0)
    stds[stds < 1e-12] = 1.0
    standardized = (sample - means) / stds
    corr = np.dot(standardized.T, standardized) / len(standardized)

    variances = np.var(sample, axis=0)
    feat_to_idx = {f: i for i, f in enumerate(keep)}
    protected_idx = {feat_to_idx[f] for f in PROTECTED_FEATURES if f in feat_to_idx}

    to_drop = set()
    for i in range(len(keep)):
        if i in to_drop:
            continue
        for j in range(i + 1, len(keep)):
            if j in to_drop:
                continue
            if abs(corr[i, j]) > CORR_THRESHOLD:
                if i in protected_idx:
                    to_drop.add(j)
                elif j in protected_idx:
                    to_drop.add(i)
                elif variances[i] >= variances[j]:
                    to_drop.add(j)
                else:
                    to_drop.add(i)

    pruned = [f for i, f in enumerate(keep) if i not in to_drop]
    log.info(f"  Correlation pruning: dropped {len(to_drop)} -> {len(pruned)} features")
    del sample, corr, standardized
    gc.collect()
    return pruned


# =============================================================================
# PHASE 1: FEATURE SELECTION + OPTUNA per target
# =============================================================================
def phase1_optimize(df, pruned_features, tgt_name, tgt_col, horizon, warm_params):
    log.info(f"\n  --- {tgt_name} (horizon={horizon} candles, {horizon*5}min) ---")
    purge = 2 * horizon
    n_total = len(df)
    y_all = df[tgt_col].values.astype(float)

    pos_rate = np.nanmean(y_all)
    log.info(f"  Target base rate: {pos_rate:.3%} ({int(np.nansum(y_all)):,} positive of {int(np.sum(~np.isnan(y_all))):,})")

    if pos_rate < 0.03 or pos_rate > 0.50:
        log.warning(f"  Skipping {tgt_name}: base rate {pos_rate:.3%} outside useful range")
        return None, None

    # Feature selection on 70/15/15 split
    base_params = warm_params.get(tgt_name, DEFAULT_PARAMS).copy()
    if "n_estimators" not in base_params:
        base_params["n_estimators"] = 3000
    base_params.setdefault("objective", "binary")
    base_params.setdefault("metric", "binary_logloss")
    base_params.setdefault("boosting_type", "gbdt")

    valid = ~np.isnan(y_all)
    valid_idx = np.where(valid)[0]
    n_v = len(valid_idx)
    tr_end = int(n_v * 0.70)
    va_end = int(n_v * 0.85)

    X_all = df[pruned_features].values
    try:
        m_fs = train_lgb(
            df.iloc[valid_idx[:tr_end]][pruned_features].values,
            y_all[valid_idx[:tr_end]],
            df.iloc[valid_idx[tr_end:va_end]][pruned_features].values,
            y_all[valid_idx[tr_end:va_end]],
            params=base_params)
    except Exception as e:
        log.warning(f"  Feature selection failed: {e}")
        selected_feats = pruned_features[:N_TOP_FEATURES]
        return base_params, selected_feats

    imp = pd.DataFrame({"feat": pruned_features, "imp": m_fs.feature_importances_})
    imp = imp.sort_values("imp", ascending=False)
    selected_feats = imp.head(N_TOP_FEATURES)["feat"].tolist()
    log.info(f"  Feature selection: top-{N_TOP_FEATURES} from {len(pruned_features)}")
    log.info(f"  Top 5 features: {imp.head(5)['feat'].tolist()}")

    # Optuna optimization
    X_sel = df[selected_feats].values
    opt_splits = []
    wf = wf_splits(n_total, 3)
    for test_start, test_end in wf:
        train_end = test_start - purge
        if train_end < 10000:
            continue
        data = split_data(X_sel, y_all, train_end, test_start, test_end, purge)
        if data is not None:
            opt_splits.append(data)

    if not opt_splits:
        log.warning(f"  No valid Optuna splits for {tgt_name}")
        return base_params, selected_feats

    def objective(trial):
        params = {
            "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
            "verbose": -1, "n_jobs": -1,
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=250),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 10, 60),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 300, 2000, step=100),
            "subsample": trial.suggest_float("subsample", 0.3, 0.9),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 30.0, log=True),
            "path_smooth": trial.suggest_float("path_smooth", 1.0, 30.0),
        }
        aucs = []
        for data in opt_splits:
            X_tr, y_tr, X_va, y_va, X_te, y_te, _ = data
            try:
                m = train_lgb(X_tr, y_tr, X_va, y_va, params=params)
                prob = m.predict_proba(X_te)[:, 1]
                aucs.append(auc_roc(y_te, prob))
            except Exception:
                aucs.append(0.5)
        return np.mean(aucs)

    # Enqueue warm-start params if available
    study = optuna.create_study(direction="maximize")
    if tgt_name in warm_params:
        wp = {k: v for k, v in warm_params[tgt_name].items()
              if k not in ("objective", "metric", "boosting_type", "verbose", "n_jobs")}
        try:
            study.enqueue_trial(wp)
        except Exception:
            pass

    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_params = {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        **study.best_params
    }
    log.info(f"  Optuna best AUC: {study.best_value:.4f}")
    return best_params, selected_feats


# =============================================================================
# PHASE 2: WALK-FORWARD VALIDATION
# =============================================================================
def phase2_wf_validate(df, tgt_name, tgt_col, horizon, best_params, selected_feats):
    log.info(f"  Walk-forward validation ({N_SPLITS} splits)...")
    purge = 2 * horizon
    n_total = len(df)
    y_all = df[tgt_col].values.astype(float)
    X_sel = df[selected_feats].values
    close_all = df["close"].values.astype(float)

    splits = wf_splits(n_total, N_SPLITS)
    split_results = []
    oof_probs = np.full(n_total, np.nan)
    oof_labels = np.full(n_total, np.nan)
    saved_models = []

    # Try prob thresholds
    PROB_THRESHOLDS = [0.35, 0.40, 0.45, 0.50]
    best_prob_thresh = PROB_THRESHOLDS[0]
    best_thresh_sharpe = -np.inf

    for test_start, test_end in splits:
        train_end = test_start - purge
        if train_end < 10000:
            continue
        data = split_data(X_sel, y_all, train_end, test_start, test_end, purge)
        if data is None:
            continue
        X_tr, y_tr, X_va, y_va, X_te, y_te, test_idx = data
        try:
            m = train_lgb(X_tr, y_tr, X_va, y_va, params=best_params)
            prob_te = m.predict_proba(X_te)[:, 1]
            oof_probs[test_idx] = prob_te
            oof_labels[test_idx] = y_te
            saved_models.append((test_idx, m, prob_te))
        except Exception as e:
            log.warning(f"    Split failed: {e}")
            continue

        # Evaluate at multiple prob thresholds on this split
        for pt in PROB_THRESHOLDS:
            trades = []
            for i_local, i_global in enumerate(test_idx):
                if i_global + horizon < n_total and prob_te[i_local] >= pt:
                    entry = close_all[i_global]
                    exit_ = close_all[i_global + horizon]
                    if entry > 0:
                        fwd_ret = (exit_ - entry) / entry
                        net_ret = -fwd_ret - FEE_RT
                        trades.append({"idx": i_global, "net_ret": net_ret})
            sr = compute_split_metrics_short(trades)
            split_results.append({
                "prob_thresh": pt,
                "n": sr["n"], "net": sr["net"], "sharpe": sr["sharpe"], "wr": sr["wr"]
            })

    if not split_results:
        log.warning(f"  No valid WF splits for {tgt_name}")
        return None

    # Find best prob threshold by Sharpe
    from collections import defaultdict
    by_thresh = defaultdict(list)
    for sr in split_results:
        by_thresh[sr["prob_thresh"]].append(sr)

    best_overall = None
    for pt, results in by_thresh.items():
        valid = [r for r in results if r["n"] >= MIN_TRADES_PER_SPLIT]
        if len(valid) < 5:
            continue
        sharpes = [r["sharpe"] for r in valid]
        pos_splits = sum(1 for r in valid if r["net"] > 0)
        avg_sharpe = np.mean(sharpes)
        if (pos_splits >= QUALITY_MIN_POSITIVE_SPLITS
                and avg_sharpe > (best_overall["avg_sharpe"] if best_overall else -np.inf)):
            best_overall = {
                "prob_thresh": pt,
                "pos_splits": pos_splits,
                "total_splits": len(valid),
                "avg_sharpe": avg_sharpe,
                "avg_net": np.mean([r["net"] for r in valid]),
                "avg_wr": np.mean([r["wr"] for r in valid]),
            }

    if best_overall is None:
        log.warning(f"  {tgt_name}: No configuration passed quality gate "
                    f"({QUALITY_MIN_POSITIVE_SPLITS}+/{N_SPLITS} positive splits required)")
        return None

    best_prob_thresh = best_overall["prob_thresh"]
    log.info(f"  Best prob_threshold: {best_prob_thresh:.2f}")
    log.info(f"  Quality: {best_overall['pos_splits']}/{best_overall['total_splits']} splits positive")
    log.info(f"  Avg Sharpe: {best_overall['avg_sharpe']:.2f}, Avg Net/split: {best_overall['avg_net']:+.2%}")
    log.info(f"  Avg Win Rate: {best_overall['avg_wr']:.1%}")

    # Overall AUC
    valid_mask = ~(np.isnan(oof_probs) | np.isnan(oof_labels))
    overall_auc = auc_roc(oof_labels[valid_mask], oof_probs[valid_mask]) if valid_mask.any() else 0.5
    log.info(f"  OOF AUC: {overall_auc:.4f}")

    return {
        "tgt_name": tgt_name,
        "tgt_col": tgt_col,
        "horizon": horizon,
        "features": selected_feats,
        "params": best_params,
        "prob_threshold": best_prob_thresh,
        "auc": overall_auc,
        "quality": best_overall,
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    setup_logging()
    t_start = time.time()

    log.info("=" * 70)
    log.info(f"  BTC BEARISH ML MODEL TRAINING — {VERSION}")
    log.info("=" * 70)
    log.info(f"  Dataset: {FEATURES_FILE}")
    log.info(f"  Targets: {[t[0] for t in BEARISH_TARGETS]}")
    log.info(f"  Output:  {MODEL_DIR}/")

    # Load dataset
    log.info(f"\n  Loading dataset...")
    df = pd.read_parquet(FEATURES_FILE)
    log.info(f"  Loaded: {len(df):,} rows x {len(df.columns)} columns")

    # Verify targets exist
    for tgt_name, tgt_col, horizon in BEARISH_TARGETS:
        if tgt_col not in df.columns:
            log.error(f"  Target column not found: {tgt_col}")
            sys.exit(1)

    all_feat_cols = get_feature_cols(df)
    log.info(f"  Feature columns: {len(all_feat_cols)}")

    # Load warm-start params
    warm_params = {}
    for pf in WARMSTART_FILES:
        if os.path.exists(pf):
            with open(pf) as f:
                loaded = json.load(f)
            for k, v in loaded.items():
                if k not in warm_params:
                    warm_params[k] = v
            log.info(f"  Loaded warm-start from {pf} ({len(loaded)} entries)")

    # Phase 0: Feature pruning
    pruned_features = phase0_load_and_prune(df, all_feat_cols)

    # Train each bearish target
    os.makedirs(MODEL_DIR, exist_ok=True)
    prod_config = {
        "version": VERSION,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "models": [],
    }

    for tgt_name, tgt_col, horizon in BEARISH_TARGETS:
        log.info("")
        log.info("=" * 70)
        log.info(f"  TRAINING: {tgt_name}")
        log.info("=" * 70)

        # Phase 1: Feature selection + Optuna
        best_params, selected_feats = phase1_optimize(
            df, pruned_features, tgt_name, tgt_col, horizon, warm_params)

        if best_params is None or selected_feats is None:
            log.warning(f"  Skipping {tgt_name} — optimization failed")
            continue

        # Phase 2: Walk-forward validation
        result = phase2_wf_validate(df, tgt_name, tgt_col, horizon, best_params, selected_feats)

        if result is None:
            log.warning(f"  {tgt_name} did not pass quality gate — not saving")
            continue

        # Train FINAL model on all data (except last 5% held out for out-of-sample)
        log.info(f"  Training final model on full dataset...")
        y_all = df[tgt_col].values.astype(float)
        X_sel = df[selected_feats].values
        valid = ~np.isnan(y_all)
        valid_idx = np.where(valid)[0]
        n_v = len(valid_idx)
        tr_end = int(n_v * 0.85)
        va_end = int(n_v * 0.93)

        try:
            final_model = train_lgb(
                df.iloc[valid_idx[:tr_end]][selected_feats].values,
                y_all[valid_idx[:tr_end]],
                df.iloc[valid_idx[tr_end:va_end]][selected_feats].values,
                y_all[valid_idx[tr_end:va_end]],
                params=best_params,
            )
        except Exception as e:
            log.error(f"  Final model training failed: {e}")
            continue

        # Save model
        model_filename = f"prod_{tgt_name}_lgb.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)
        with open(model_path, "wb") as f:
            pickle.dump(final_model, f)
        log.info(f"  Saved: {model_filename}")

        # Add to production config
        entry = {
            "name": tgt_name,
            "target": tgt_col,
            "target_name": tgt_name,
            "lgb_file": model_filename,
            "horizon": horizon,
            "prob_threshold": result["prob_threshold"],
            "features": selected_feats,
            "auc": round(result["auc"], 4),
            "quality_weight": 1.0,
            "use_ensemble": False,
            "lgb_weight": 1.0,
            "direction": "BEARISH",
            "quality": result["quality"],
        }
        prod_config["models"].append(entry)
        log.info(f"  {tgt_name}: prob_thresh={result['prob_threshold']:.2f}, AUC={result['auc']:.4f}")

    # Save production config
    if not prod_config["models"]:
        log.error("  No models passed quality gate. Nothing saved.")
        sys.exit(1)

    config_path = os.path.join(MODEL_DIR, "production_config_bearish.json")
    with open(config_path, "w") as f:
        json.dump(prod_config, f, indent=2, default=float)
    log.info(f"\n  Saved config: {config_path}")
    log.info(f"  Models saved: {len(prod_config['models'])}")

    elapsed = time.time() - t_start
    log.info(f"\n  Total runtime: {elapsed / 60:.1f} minutes")
    log.info("  Done.")

    print(f"\n{'='*60}")
    print(f"BEARISH MODELS TRAINED SUCCESSFULLY")
    print(f"{'='*60}")
    for entry in prod_config["models"]:
        q = entry["quality"]
        print(f"  {entry['name']}: AUC={entry['auc']:.4f}, "
              f"prob_thresh={entry['prob_threshold']:.2f}, "
              f"{q['pos_splits']}/{q['total_splits']} splits positive, "
              f"avg Sharpe={q['avg_sharpe']:.2f}")
    print(f"\nNext step: restart the bot to activate bearish ML signals.")
    print(f"  pkill -f run_pipeline.py")
    print(f"  nohup python run_pipeline.py --loop --interval 300 --no-testnet --live >> logs/pipeline_live.log 2>&1 &")


if __name__ == "__main__":
    main()
