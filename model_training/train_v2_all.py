"""
BTC ML Training -- v2_all: Iteration 2 -- DD Control & Veto Filters
================================================================================
Building on v1_all (Sharpe 8.65, 10/10, worst DD -22.05%), this version
addresses the main gap: worst DD exceeding the -20% target.

Key changes from v1_all:
  1. Actually APPLY down-model veto filters during walk-forward backtesting
  2. Grid-search veto thresholds per horizon on OOF data
  3. Stricter selection: 8/10 positive (not 7/10)
  4. Model prediction deduplication: max 3 configs per unique target
  5. Much lower max_concurrent sweep (c4..c14 focus)
  6. Warm-start from v1_all Optuna params
  7. Tighter DD configs (DD=0.15%, DD=0.1%)

Usage:
  python model_training/train_v2_all.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import os
import sys
import time
import json
import pickle
import warnings
import logging
import gc
from datetime import datetime, timezone

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from sklearn.isotonic import IsotonicRegression
    HAS_ISOTONIC = True
except ImportError:
    HAS_ISOTONIC = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# PATHS & CONSTANTS
# =============================================================================
VERSION = "v2_all"
PROCESSED_DIR = "processed_data"
FEATURES_FILE = os.path.join(PROCESSED_DIR, "btc_training_dataset.parquet")
LOG_DIR = os.path.join("model_training", "logs")
LOG_FILE = os.path.join(LOG_DIR, f"{VERSION}.log")
MODEL_DIR = os.path.join("models", VERSION)
FINDINGS_FILE = os.path.join("model_training", "findings_2.md")

# Warm-start from v1_all
V1_OPTUNA_FILE = os.path.join("models", "v1_all", "optuna_params.json")
V23_OPTUNA_FILE = os.path.join("models", "v23", "optuna_params_v23.json")

FEE_RT = 0.0006          # 0.06% round-trip
N_SPLITS = 10
CANDLES_PER_DAY = 288     # 5-min candles
QUICK_SPLITS = 3
OPTUNA_TRIALS = 40
N_TOP_FEATURES = 100
CORR_THRESHOLD = 0.95
NAN_THRESHOLD = 0.50
MIN_TRADES_PER_SPLIT = 10
MAX_CONFIGS_PER_TARGET = 3  # v2: limit model crowding

# Target sweep configuration
HORIZONS = [6, 12, 24, 36, 48]
THRESHOLDS = ["0002", "0003", "0005", "001"]
PROB_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]
TOP_PCTS = [None, 0.20, 0.10]

PROTECTED_FEATURES = {"hour_of_day", "day_of_week", "close", "volume"}

DEFAULT_PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
    "n_estimators": 1000, "learning_rate": 0.01, "num_leaves": 24,
    "max_depth": 5, "min_child_samples": 500, "subsample": 0.6,
    "subsample_freq": 1, "colsample_bytree": 0.4, "reg_alpha": 1.0,
    "reg_lambda": 10.0, "path_smooth": 10.0,
}

# v2: Focus DD sweep on lower concurrency to control drawdowns
DD_CONFIGS = [
    # (dd_limit, cooldown, max_concurrent, scale, label)
    (0.001, 10, 6,  1.0, "dd1m_c6"),
    (0.001, 10, 8,  1.0, "dd1m_c8"),
    (0.001, 10, 10, 1.0, "dd1m_c10"),
    (0.001, 10, 12, 1.0, "dd1m_c12"),
    (0.0015, 10, 6,  1.0, "dd15_c6"),
    (0.0015, 10, 8,  1.0, "dd15_c8"),
    (0.0015, 10, 10, 1.0, "dd15_c10"),
    (0.0015, 10, 12, 1.0, "dd15_c12"),
    (0.002, 10, 6,  1.0, "dd2m_c6"),
    (0.002, 10, 8,  1.0, "dd2m_c8"),
    (0.002, 10, 10, 1.0, "dd2m_c10"),
    (0.002, 10, 12, 1.0, "dd2m_c12"),
    (0.002, 10, 14, 1.0, "dd2m_c14"),
    (0.002, 10, 16, 1.0, "dd2m_c16"),
    (0.002, 10, 8,  0.8, "dd2m_c8_s80"),
    (0.002, 10, 10, 0.8, "dd2m_c10_s80"),
    (0.002, 10, 12, 0.8, "dd2m_c12_s80"),
]

# Veto threshold candidates for down-model filters
VETO_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


# =============================================================================
# LOGGING
# =============================================================================
log = logging.getLogger("train_v2_all")


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
# UTILITY FUNCTIONS
# =============================================================================
def auc_roc(y_true, y_prob):
    """Fast AUC-ROC without sklearn dependency."""
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
    """Identify feature columns (exclude targets, metadata)."""
    exclude = {"open_time_ms", "timestamp"}
    target_cols = {c for c in df.columns if c.startswith("target_")}
    return sorted(c for c in df.columns if c not in exclude and c not in target_cols)


def get_date_range(df_slice, col="open_time_ms"):
    dates = pd.to_datetime(df_slice[col], unit="ms", utc=True)
    return f"{dates.iloc[0].strftime('%Y-%m-%d')}/{dates.iloc[-1].strftime('%Y-%m-%d')}"


def train_lgb(X_tr, y_tr, X_va, y_va, params, seed=42, n_jobs=-1):
    p = dict(params)
    p["random_state"] = seed
    p["verbose"] = -1
    p["n_jobs"] = n_jobs
    m = lgb.LGBMClassifier(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return m


def train_catboost(X_tr, y_tr, X_va, y_va, seed=42):
    if not HAS_CATBOOST:
        return None
    m = CatBoostClassifier(
        iterations=3000, learning_rate=0.01, depth=5,
        l2_leaf_reg=10.0, random_strength=2.0,
        bagging_temperature=0.5, border_count=128,
        min_data_in_leaf=500, random_seed=seed,
        verbose=0, eval_metric="Logloss",
        early_stopping_rounds=100,
    )
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
    return m


def wf_splits(n_total, n_splits, test_frac=0.05):
    """Generate walk-forward split indices."""
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
    """Split data for WF with validation set and purge gap."""
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


def backtest_threshold(close, y_prob, horizon, fee_rt, prob_threshold,
                       top_pct=None, hours=None, down_probs=None, veto_thresh=None):
    """Backtest with threshold + optional down-model veto."""
    n = len(close)
    candidates = []
    i = 0
    while i + horizon < n:
        if y_prob[i] > prob_threshold:
            # v2: Actually apply down-model veto
            if down_probs is not None and veto_thresh is not None:
                if down_probs[i] > veto_thresh:
                    i += horizon
                    continue
            raw_ret = (close[i + horizon] - close[i]) / close[i]
            candidates.append({
                "idx": i, "prob": y_prob[i], "raw_ret": raw_ret,
                "net_ret": raw_ret - fee_rt,
                "hour": int(hours[i]) if hours is not None else -1,
            })
        i += horizon

    if not candidates:
        return candidates

    if top_pct is not None:
        trades = []
        past_probs = []
        min_warmup = 20
        for t in candidates:
            past_probs.append(t["prob"])
            if len(past_probs) < min_warmup:
                trades.append(t)
            else:
                expanding_threshold = np.quantile(past_probs, 1.0 - top_pct)
                if t["prob"] >= expanding_threshold:
                    trades.append(t)
        return trades
    return candidates


def portfolio_backtest(model_trades_list, max_dd_pct=None, cooldown=20,
                       max_concurrent=None, position_scale=1.0,
                       model_weights=None):
    """Run portfolio-level backtest with circuit breaker."""
    merged = []
    for name, trades, horizon in model_trades_list:
        w = model_weights.get(name, 1.0) if model_weights else 1.0
        for t in trades:
            merged.append({**t, "model": name, "horizon": horizon, "model_weight": w})
    merged.sort(key=lambda t: t["idx"])

    empty = {"n": 0, "net": 0.0, "wr": 0.0, "max_dd": 0.0, "sharpe": 0.0,
             "sortino": 0.0, "calmar": 0.0, "profit_factor": 0.0,
             "model_counts": {}, "trades": []}

    if not merged:
        return empty

    # Concurrency filter
    if max_concurrent is not None:
        filtered = []
        active = []
        for t in merged:
            active = [(e, x) for e, x in active if x > t["idx"]]
            if len(active) < max_concurrent:
                filtered.append(t)
                active.append((t["idx"], t["idx"] + t["horizon"]))
        merged = filtered

    if not merged:
        return empty

    # Circuit breaker
    if max_dd_pct is not None:
        filtered = []
        cum = 1.0
        peak = 1.0
        breaker = False
        skip_count = 0
        for t in merged:
            if breaker:
                skip_count += 1
                if skip_count >= cooldown:
                    breaker = False
                continue
            t_copy = dict(t)
            t_copy["net_ret"] = t["net_ret"] * position_scale * t["model_weight"]
            filtered.append(t_copy)
            cum *= (1 + t_copy["net_ret"])
            peak = max(peak, cum)
            dd = (cum - peak) / peak
            if dd < -max_dd_pct:
                breaker = True
                skip_count = 0
        merged = filtered
    else:
        for t in merged:
            t["net_ret"] = t["net_ret"] * position_scale * t["model_weight"]

    if not merged:
        return empty

    rets = np.array([t["net_ret"] for t in merged])
    cum = np.cumprod(1 + rets)
    total_net = cum[-1] - 1
    wr = (rets > 0).mean()
    cummax = np.maximum.accumulate(cum)
    max_dd = ((cum - cummax) / cummax).min()

    # Daily returns (calendar-time, zero-trade days included)
    idxs = np.array([t["idx"] for t in merged])
    min_idx, max_idx = idxs.min(), idxs.max()
    n_days = max(1, (max_idx - min_idx) / CANDLES_PER_DAY)
    total_days = int(n_days) + 1
    daily_rets = np.zeros(total_days)
    for t in merged:
        day = int((t["idx"] - min_idx) / CANDLES_PER_DAY)
        if day < total_days:
            daily_rets[day] += t["net_ret"]

    mu_d = daily_rets.mean()
    sigma_d = daily_rets.std()
    sharpe = (mu_d / sigma_d) * np.sqrt(365.25) if sigma_d > 0 else 0.0

    # Sortino (downside deviation)
    downside = daily_rets[daily_rets < 0]
    down_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-9
    sortino = (mu_d / down_std) * np.sqrt(365.25) if down_std > 0 else 0.0

    # Calmar
    annual_ret = (1 + total_net) ** (365.25 / max(n_days, 1)) - 1
    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0

    # Profit factor
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    profit_factor = wins / losses if losses > 1e-9 else 999.0

    model_counts = {}
    for t in merged:
        model_counts[t["model"]] = model_counts.get(t["model"], 0) + 1

    return {
        "n": len(merged), "net": total_net, "wr": wr, "max_dd": max_dd,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "profit_factor": profit_factor, "model_counts": model_counts,
        "daily_rets": daily_rets, "annual_ret": annual_ret,
        "trades": merged,
    }


def compute_split_metrics(trades, horizon):
    """Compute per-split metrics from a list of trades."""
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
# PHASE 0: DATA LOADING + CORRELATION-BASED FEATURE PRUNING
# =============================================================================
def phase0_load_and_prune(df, all_feat_cols):
    log.info("")
    log.info("=" * 80)
    log.info("  PHASE 0: DATA LOADING + CORRELATION-BASED FEATURE PRUNING")
    log.info("=" * 80)
    log.info(f"  Input: {len(df):,} rows x {len(all_feat_cols)} features")

    nan_fracs = df[all_feat_cols].isna().mean()
    high_nan = nan_fracs[nan_fracs > NAN_THRESHOLD].index.tolist()
    keep = [c for c in all_feat_cols if c not in high_nan]
    log.info(f"  Dropped {len(high_nan)} features with >{NAN_THRESHOLD*100:.0f}% NaN -> {len(keep)} remain")

    sample_idx = np.arange(0, len(df), 5)
    sample = df.iloc[sample_idx][keep].values.astype(np.float64)

    col_medians = np.nanmedian(sample, axis=0)
    for j in range(sample.shape[1]):
        mask = np.isnan(sample[:, j])
        sample[mask, j] = col_medians[j]

    log.info(f"  Computing correlation matrix on {len(sample_idx):,} x {len(keep)} sample...")
    t0 = time.time()

    means = sample.mean(axis=0)
    stds = sample.std(axis=0)
    stds[stds < 1e-12] = 1.0
    standardized = (sample - means) / stds
    corr = np.dot(standardized.T, standardized) / len(standardized)

    elapsed = time.time() - t0
    log.info(f"  Correlation matrix computed in {elapsed:.1f}s")

    variances = np.var(sample, axis=0)
    feat_to_idx = {f: i for i, f in enumerate(keep)}
    protected_idx = {feat_to_idx[f] for f in PROTECTED_FEATURES if f in feat_to_idx}

    to_drop = set()
    n_features = len(keep)
    for i in range(n_features):
        if i in to_drop:
            continue
        for j in range(i + 1, n_features):
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

    pruned_features = [f for i, f in enumerate(keep) if i not in to_drop]
    log.info(f"  Correlation pruning (|r|>{CORR_THRESHOLD}): dropped {len(to_drop)} -> {len(pruned_features)} features")
    log.info(f"  Protected features preserved: {[f for f in PROTECTED_FEATURES if f in set(pruned_features)]}")

    return pruned_features


# =============================================================================
# PHASE 1: SYSTEMATIC TARGET SWEEP -- QUICK SCREEN
# =============================================================================
def phase1_target_sweep(df, pruned_features):
    log.info("")
    log.info("=" * 80)
    log.info("  PHASE 1: SYSTEMATIC TARGET SWEEP -- QUICK SCREEN")
    log.info("=" * 80)

    up_targets = []
    for h in HORIZONS:
        for th in THRESHOLDS:
            col = f"target_up_{h}_{th}"
            if col in df.columns:
                up_targets.append((f"up_{h}_{th}", col, h))

    fav_targets = []
    for h in [12, 36]:
        for th in ["0002", "0003", "0005"]:
            col = f"target_favorable_{h}_{th}"
            if col in df.columns:
                fav_targets.append((f"fav_{h}_{th}", col, h))

    all_targets = up_targets + fav_targets
    log.info(f"  Found {len(all_targets)} target columns to evaluate")

    n_total = len(df)
    splits = wf_splits(n_total, QUICK_SPLITS)
    X_all = df[pruned_features].values
    close_all = df["close"].values
    hours_all = df["hour_of_day"].values if "hour_of_day" in df.columns else None

    candidates = []
    target_aucs = {}

    t0_phase = time.time()
    log.info(f"  Training {len(all_targets)} targets x {QUICK_SPLITS} splits with default params...")

    for tgt_name, tgt_col, horizon in all_targets:
        purge = 2 * horizon
        y_all = df[tgt_col].values

        split_predictions = {}

        for s_idx, (test_start, test_end) in enumerate(splits):
            train_end = test_start - purge
            if train_end < 10000:
                continue
            data = split_data(X_all, y_all, train_end, test_start, test_end, purge)
            if data is None:
                continue
            X_tr, y_tr, X_va, y_va, X_te, y_te, test_idx = data

            try:
                m = train_lgb(X_tr, y_tr, X_va, y_va, params=DEFAULT_PARAMS)
                p_test = m.predict_proba(X_te)[:, 1]
            except Exception:
                continue

            close_te = close_all[test_idx]
            hours_te = hours_all[test_idx] if hours_all is not None else None
            auc = auc_roc(y_te, p_test)
            split_predictions[s_idx] = (p_test, close_te, hours_te, y_te, auc)

        if not split_predictions:
            continue

        target_aucs[tgt_name] = [sp[4] for sp in split_predictions.values()]
        avg_auc = np.mean(target_aucs[tgt_name])

        for prob_thresh in PROB_THRESHOLDS:
            for top_pct in TOP_PCTS:
                split_nets = []
                split_trade_counts = []
                for s_idx, (p_test, close_te, hours_te, y_te, _auc) in split_predictions.items():
                    trades = backtest_threshold(
                        close_te, p_test, horizon, FEE_RT,
                        prob_thresh, top_pct, hours=hours_te)
                    if trades:
                        net = np.prod([1 + t["net_ret"] for t in trades]) - 1
                        split_nets.append(net)
                    else:
                        split_nets.append(0.0)
                    split_trade_counts.append(len(trades) if trades else 0)

                pos = sum(1 for n_ in split_nets if n_ > 0)
                med_trades = int(np.median(split_trade_counts))
                total_splits = len(split_nets)

                # v2: keep same Phase 1 criteria (filter harder in Phase 3)
                if (avg_auc > 0.55
                        and total_splits >= QUICK_SPLITS
                        and pos >= 2
                        and med_trades >= MIN_TRADES_PER_SPLIT):
                    top_label = f"t{int(top_pct*100)}" if top_pct else "all"
                    cand_name = f"{tgt_name}_p{int(prob_thresh*100)}{top_label}"
                    candidates.append({
                        "name": cand_name,
                        "target_name": tgt_name,
                        "target_col": tgt_col,
                        "horizon": horizon,
                        "prob_thresh": prob_thresh,
                        "top_pct": top_pct,
                        "avg_auc": avg_auc,
                        "wf_pos": pos,
                        "total_splits": total_splits,
                        "med_trades": med_trades,
                        "avg_net": np.mean(split_nets),
                    })

        log.info(f"  {tgt_name:<15s} AUC={avg_auc:.4f}  "
                 f"({sum(1 for c in candidates if c['target_name'] == tgt_name)} configs pass)")

    elapsed = time.time() - t0_phase
    log.info(f"  Phase 1 complete: {len(candidates)} candidate configs from {len(all_targets)} targets in {elapsed:.0f}s")

    candidates.sort(key=lambda c: c["avg_auc"], reverse=True)

    log.info(f"\n  Top 20 candidates:")
    log.info(f"  {'Name':<30s} {'AUC':>6s} {'Pos':>5s} {'MedN':>5s} {'AvgNet':>8s}")
    for c in candidates[:20]:
        log.info(f"  {c['name']:<30s} {c['avg_auc']:>6.4f} "
                 f"{c['wf_pos']}/{c['total_splits']}  "
                 f"{c['med_trades']:>5d} {c['avg_net']:>+7.2%}")

    return candidates, target_aucs


# =============================================================================
# PHASE 2: PER-TARGET FEATURE SELECTION + OPTUNA
# =============================================================================
def phase2_feature_selection_optuna(df, candidates, pruned_features):
    log.info("")
    log.info("=" * 80)
    log.info("  PHASE 2: PER-TARGET FEATURE SELECTION + OPTUNA")
    log.info("=" * 80)

    # Load warm-start params (prefer v1_all, fallback to v23)
    warm_params = {}
    for param_file in [V1_OPTUNA_FILE, V23_OPTUNA_FILE]:
        if os.path.exists(param_file):
            with open(param_file, "r") as f:
                loaded = json.load(f)
            for k, v in loaded.items():
                if k not in warm_params:
                    warm_params[k] = v
            log.info(f"  Loaded warm-start params from {param_file} ({len(loaded)} targets)")

    log.info(f"  Total warm-start params: {len(warm_params)} targets")

    unique_targets = {}
    for c in candidates:
        tn = c["target_name"]
        if tn not in unique_targets:
            unique_targets[tn] = {
                "target_col": c["target_col"],
                "horizon": c["horizon"],
            }
    log.info(f"  Unique targets to optimize: {len(unique_targets)}")

    n_total = len(df)
    X_all = df[pruned_features].values
    target_features = {}
    target_params = {}
    target_ensemble = {}

    for tgt_name, tgt_info in unique_targets.items():
        tgt_col = tgt_info["target_col"]
        horizon = tgt_info["horizon"]
        purge = 2 * horizon
        y_all = df[tgt_col].values

        log.info(f"\n  --- {tgt_name} (horizon={horizon}) ---")

        base_params = warm_params.get(tgt_name, DEFAULT_PARAMS).copy()
        if "n_estimators" not in base_params:
            base_params["n_estimators"] = 3000
        if "objective" not in base_params:
            base_params.update({"objective": "binary", "metric": "binary_logloss",
                                "boosting_type": "gbdt"})

        # 70/15/15 split for feature selection
        valid = ~np.isnan(y_all)
        valid_idx = np.where(valid)[0]
        n_v = len(valid_idx)
        tr_end = int(n_v * 0.70)
        va_end = int(n_v * 0.85)

        try:
            m_fs = train_lgb(
                df.iloc[valid_idx[:tr_end]][pruned_features].values,
                y_all[valid_idx[:tr_end]],
                df.iloc[valid_idx[tr_end:va_end]][pruned_features].values,
                y_all[valid_idx[tr_end:va_end]],
                params=base_params)
        except Exception as e:
            log.warning(f"  Feature selection failed for {tgt_name}: {e}")
            target_features[tgt_name] = pruned_features[:N_TOP_FEATURES]
            target_params[tgt_name] = base_params
            target_ensemble[tgt_name] = (False, 1.0)
            continue

        imp = pd.DataFrame({"feat": pruned_features, "imp": m_fs.feature_importances_})
        imp = imp.sort_values("imp", ascending=False)
        selected_feats = imp.head(N_TOP_FEATURES)["feat"].tolist()
        target_features[tgt_name] = selected_feats
        log.info(f"  Feature selection: top-{N_TOP_FEATURES} from {len(pruned_features)}")
        log.info(f"    Top 5: {imp.head(5)['feat'].tolist()}")

        # Optuna
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
            target_params[tgt_name] = base_params
            target_ensemble[tgt_name] = (False, 1.0)
            continue

        enqueue_params = None
        if tgt_name in warm_params:
            wp = warm_params[tgt_name]
            enqueue_params = {
                "learning_rate": wp.get("learning_rate", 0.01),
                "num_leaves": wp.get("num_leaves", 24),
                "max_depth": wp.get("max_depth", 5),
                "min_child_samples": wp.get("min_child_samples", 500),
                "subsample": wp.get("subsample", 0.6),
                "colsample_bytree": wp.get("colsample_bytree", 0.4),
                "colsample_bynode": wp.get("colsample_bynode", 0.5),
                "reg_alpha": wp.get("reg_alpha", 1.0),
                "reg_lambda": wp.get("reg_lambda", 10.0),
                "feature_fraction_bynode": wp.get("feature_fraction_bynode", 0.5),
                "path_smooth": wp.get("path_smooth", 10.0),
            }

        def make_objective(splits_data, target_label):
            def objective(trial):
                t_start = time.time()
                params = {
                    "objective": "binary", "metric": "binary_logloss",
                    "boosting_type": "gbdt", "n_estimators": 3000,
                    "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.05, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 12, 64),
                    "max_depth": trial.suggest_int("max_depth", 3, 8),
                    "min_child_samples": trial.suggest_int("min_child_samples", 200, 2000),
                    "subsample": trial.suggest_float("subsample", 0.3, 0.8),
                    "subsample_freq": 1,
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.6),
                    "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 0.7),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 20.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
                    "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.3, 0.7),
                    "path_smooth": trial.suggest_float("path_smooth", 1.0, 30.0),
                }
                aucs = []
                for si, (X_tr, y_tr, X_va, y_va, X_te, y_te, _) in enumerate(splits_data):
                    try:
                        model = train_lgb(X_tr, y_tr, X_va, y_va, params=params)
                        p = model.predict_proba(X_te)[:, 1]
                        aucs.append(auc_roc(y_te, p))
                    except Exception:
                        return 0.5
                result = np.mean(aucs)
                elapsed_t = time.time() - t_start
                if trial.number % 5 == 0 or trial.number < 3:
                    log.info(f"    [{target_label}] trial {trial.number}: AUC={result:.4f} ({elapsed_t:.0f}s)")
                    for h in log.handlers:
                        h.flush()
                return result
            return objective

        log.info(f"  Optuna: {OPTUNA_TRIALS} trials, {len(opt_splits)}-split CV...")
        for h in log.handlers:
            h.flush()
        t0 = time.time()
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        if enqueue_params:
            study.enqueue_trial(enqueue_params)
        # timeout=3600s (1hr) safety net per target
        study.optimize(make_objective(opt_splits, tgt_name), n_trials=OPTUNA_TRIALS,
                       show_progress_bar=False, timeout=3600)
        elapsed = time.time() - t0

        best = study.best_params
        opt_p = {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt", "n_estimators": 3000, "subsample_freq": 1,
        }
        opt_p.update(best)
        target_params[tgt_name] = opt_p

        log.info(f"    Best AUC: {study.best_value:.4f} in {elapsed:.0f}s")
        log.info(f"    lr={best['learning_rate']:.4f} leaves={best['num_leaves']} "
                 f"depth={best['max_depth']} min_child={best['min_child_samples']}")

        # CatBoost ensemble decision
        log.info(f"    Testing CatBoost ensemble...")
        for h in log.handlers:
            h.flush()
        if HAS_CATBOOST:
            lgb_aucs = []
            ens_aucs = []
            for si, (X_tr, y_tr, X_va, y_va, X_te, y_te, _) in enumerate(opt_splits):
                try:
                    m_lgb = train_lgb(X_tr, y_tr, X_va, y_va, params=opt_p)
                    p_lgb = m_lgb.predict_proba(X_te)[:, 1]
                    lgb_aucs.append(auc_roc(y_te, p_lgb))

                    m_cb = train_catboost(X_tr, y_tr, X_va, y_va)
                    if m_cb is not None:
                        p_cb = m_cb.predict_proba(X_te)[:, 1]
                        p_ens = 0.7 * p_lgb + 0.3 * p_cb
                        ens_aucs.append(auc_roc(y_te, p_ens))
                    log.info(f"      CB split {si}: done")
                    for h in log.handlers:
                        h.flush()
                except Exception:
                    pass

            if lgb_aucs and ens_aucs and np.mean(ens_aucs) - np.mean(lgb_aucs) >= 0.002:
                target_ensemble[tgt_name] = (True, 0.7)
                log.info(f"    CatBoost ensemble: YES (AUC +{np.mean(ens_aucs)-np.mean(lgb_aucs):.4f})")
            else:
                target_ensemble[tgt_name] = (False, 1.0)
                log.info(f"    CatBoost ensemble: NO")
        else:
            target_ensemble[tgt_name] = (False, 1.0)

        log.info(f"  --- {tgt_name} DONE ---")
        for h in log.handlers:
            h.flush()
        gc.collect()

    return target_features, target_params, target_ensemble


# =============================================================================
# PHASE 3: FULL 10-SPLIT WALK-FORWARD + OOF + CALIBRATION
# =============================================================================
def phase3_walk_forward(df, candidates, target_features, target_params, target_ensemble):
    log.info("")
    log.info("=" * 80)
    log.info("  PHASE 3: FULL 10-SPLIT WALK-FORWARD + OOF COLLECTION")
    log.info("=" * 80)

    n_total = len(df)
    splits = wf_splits(n_total, N_SPLITS)
    close_all = df["close"].values
    hours_all = df["hour_of_day"].values if "hour_of_day" in df.columns else None

    all_results = {}
    oof_predictions = {}
    feature_stability = {}
    saved_models = {}

    target_to_candidates = {}
    for c in candidates:
        tn = c["target_name"]
        if tn not in target_to_candidates:
            target_to_candidates[tn] = []
        target_to_candidates[tn].append(c)

    log.info(f"  {len(candidates)} candidates from {len(target_to_candidates)} unique targets, {N_SPLITS} splits")

    for tgt_name, tgt_candidates in target_to_candidates.items():
        if tgt_name not in target_features or tgt_name not in target_params:
            continue

        feat = target_features[tgt_name]
        params = target_params[tgt_name]
        use_ens, lgb_w = target_ensemble.get(tgt_name, (False, 1.0))
        tgt_col = tgt_candidates[0]["target_col"]
        horizon = tgt_candidates[0]["horizon"]
        purge = 2 * horizon

        X_feat = df[feat].values
        y_all = df[tgt_col].values

        if tgt_name not in feature_stability:
            feature_stability[tgt_name] = {}

        t0 = time.time()

        for s_idx, (test_start, test_end) in enumerate(splits):
            train_end = test_start - purge
            if train_end < 10000:
                continue

            data = split_data(X_feat, y_all, train_end, test_start, test_end, purge)
            if data is None:
                continue
            X_tr, y_tr, X_va, y_va, X_te, y_te, test_indices = data

            try:
                m_lgb = train_lgb(X_tr, y_tr, X_va, y_va, params=params)
                p_test = m_lgb.predict_proba(X_te)[:, 1]
            except Exception as e:
                log.warning(f"  {tgt_name} split {s_idx} LGB failed: {e}")
                continue

            m_cb = None
            if use_ens and HAS_CATBOOST:
                try:
                    m_cb = train_catboost(X_tr, y_tr, X_va, y_va)
                    if m_cb is not None:
                        p_cb = m_cb.predict_proba(X_te)[:, 1]
                        p_test = lgb_w * p_test + (1 - lgb_w) * p_cb
                except Exception:
                    pass

            auc = auc_roc(y_te, p_test)
            close_te = close_all[test_indices]
            hours_te = hours_all[test_indices] if hours_all is not None else None

            top30 = np.argsort(m_lgb.feature_importances_)[-30:]
            for fi in top30:
                fn = feat[fi]
                feature_stability[tgt_name][fn] = feature_stability[tgt_name].get(fn, 0) + 1

            for c in tgt_candidates:
                name = c["name"]
                if name not in all_results:
                    all_results[name] = {}
                if name not in oof_predictions:
                    oof_predictions[name] = {"indices": [], "preds": [], "labels": []}

                trades = backtest_threshold(
                    close_te, p_test, horizon, FEE_RT,
                    c["prob_thresh"], c["top_pct"], hours=hours_te)

                all_results[name][s_idx] = {
                    "trades": trades, "auc": auc, "horizon": horizon,
                }

                oof_predictions[name]["indices"].extend(test_indices.tolist())
                oof_predictions[name]["preds"].extend(p_test.tolist())
                oof_predictions[name]["labels"].extend(y_te.tolist())

                if s_idx == 0:
                    saved_models[name] = {
                        "model_lgb": m_lgb, "model_cb": m_cb,
                        "features": feat, "target": tgt_col,
                        "horizon": horizon, "prob_threshold": c["prob_thresh"],
                        "top_pct": c["top_pct"], "auc": auc,
                        "lgb_params": params,
                        "use_ensemble": use_ens, "lgb_weight": lgb_w,
                        "target_name": tgt_name,
                    }

        elapsed = time.time() - t0
        log.info(f"  {tgt_name:<15s} {N_SPLITS} splits in {elapsed:.0f}s  "
                 f"({len(tgt_candidates)} configs)")

    # v2: STRICTER selection -- 8/10 positive (up from 7/10)
    selected = []
    log.info(f"\n  === SELECTION (>=8/10 pos, AUC>0.55, avg_net>0) ===")
    log.info(f"  {'Name':<30s} {'AUC':>6s} {'Pos':>5s} {'AvgNet':>8s} {'Sharpe':>7s} {'MedN':>5s} {'Status':>8s}")

    # Track per-target selections for deduplication
    target_selected = {}  # target_name -> [(name, avg_net, avg_sharpe)]

    for name in sorted(all_results.keys()):
        results = all_results[name]
        aucs = [r["auc"] for r in results.values()]
        avg_auc = np.mean(aucs) if aucs else 0

        trade_nets = []
        split_sharpes = []
        trade_counts = []
        for r in results.values():
            metrics = compute_split_metrics(r["trades"], r["horizon"])
            trade_nets.append(metrics["net"])
            trade_counts.append(metrics["n"])
            if metrics["sharpe"] != 0:
                split_sharpes.append(metrics["sharpe"])

        pos = sum(1 for n_ in trade_nets if n_ > 0)
        total = len(trade_nets)
        avg_net = np.mean(trade_nets) if trade_nets else 0
        avg_sharpe = np.mean(split_sharpes) if split_sharpes else 0
        med_trades = int(np.median(trade_counts)) if trade_counts else 0

        # v2: stricter -- 8/10 positive
        passed = (total >= 8 and pos >= 8 and avg_auc > 0.55 and avg_net > 0)
        status = "PASS" if passed else "drop"

        if passed:
            # Find target name
            tgt_name = None
            for c in candidates:
                if c["name"] == name:
                    tgt_name = c["target_name"]
                    break
            if tgt_name:
                if tgt_name not in target_selected:
                    target_selected[tgt_name] = []
                target_selected[tgt_name].append((name, avg_net, avg_sharpe))

        log.info(f"  {name:<30s} {avg_auc:>6.4f} {pos:>2d}/{total:<2d} "
                 f"{avg_net:>+7.2%} {avg_sharpe:>+6.1f} {med_trades:>5d} [{status}]")

    # v2: Deduplicate -- max N configs per target, keep best by avg_sharpe
    log.info(f"\n  === MODEL DEDUPLICATION (max {MAX_CONFIGS_PER_TARGET} per target) ===")
    for tgt_name, configs in target_selected.items():
        configs.sort(key=lambda x: -x[2])  # sort by avg_sharpe descending
        kept = configs[:MAX_CONFIGS_PER_TARGET]
        dropped = configs[MAX_CONFIGS_PER_TARGET:]
        for name, _, _ in kept:
            selected.append(name)
        if dropped:
            log.info(f"  {tgt_name}: kept {len(kept)}, dropped {len(dropped)} "
                     f"(e.g., {dropped[0][0]})")
        else:
            log.info(f"  {tgt_name}: kept {len(kept)}")

    log.info(f"\n  Selected: {len(selected)} models from {len(all_results)} candidates "
             f"(after 8/10 filter + {MAX_CONFIGS_PER_TARGET}-per-target dedup)")

    # Probability calibration
    calibrators = {}
    if HAS_ISOTONIC:
        log.info(f"\n  === PROBABILITY CALIBRATION (Isotonic) ===")
        for name in selected:
            oof = oof_predictions.get(name)
            if oof and len(oof["preds"]) > 100:
                try:
                    iso = IsotonicRegression(out_of_bounds="clip")
                    iso.fit(np.array(oof["preds"]), np.array(oof["labels"]))
                    calibrators[name] = iso
                except Exception:
                    pass
        log.info(f"  Calibrated {len(calibrators)} models")

    # Feature stability report
    log.info(f"\n  === FEATURE STABILITY (>=8/10 splits) ===")
    for tgt_name, feat_counts in feature_stability.items():
        stable = [(f, c) for f, c in feat_counts.items() if c >= 8]
        stable.sort(key=lambda x: -x[1])
        if stable:
            log.info(f"  {tgt_name}: {len(stable)} stable features -- "
                     f"{[f[0] for f in stable[:5]]}")

    return selected, all_results, oof_predictions, calibrators, saved_models, feature_stability


# =============================================================================
# PHASE 4: DOWN-MODEL FILTERS WITH ACTIVE VETO TUNING
# =============================================================================
def phase4_down_filters(df, selected, all_results, target_features, target_params, oof_predictions):
    log.info("")
    log.info("=" * 80)
    log.info("  PHASE 4: DOWN-MODEL FILTERS WITH ACTIVE VETO TUNING")
    log.info("=" * 80)

    selected_horizons = set()
    horizon_thresholds = {}
    for name in selected:
        for h in HORIZONS:
            for th in THRESHOLDS:
                if f"up_{h}_{th}" in name or f"fav_{h}_{th}" in name:
                    selected_horizons.add(h)
                    if h not in horizon_thresholds:
                        horizon_thresholds[h] = set()
                    horizon_thresholds[h].add(th)

    log.info(f"  Horizons with selected models: {sorted(selected_horizons)}")

    n_total = len(df)
    # v2: Use full N_SPLITS for down-model training (not just QUICK_SPLITS)
    splits = wf_splits(n_total, N_SPLITS)
    close_all = df["close"].values
    hours_all = df["hour_of_day"].values if "hour_of_day" in df.columns else None

    down_filters = {}
    down_oof_data = {}  # horizon -> {indices, preds, labels}

    for horizon in sorted(selected_horizons):
        best_down = None
        best_auc = 0

        for th in sorted(horizon_thresholds.get(horizon, THRESHOLDS)):
            down_col = f"target_down_{horizon}_{th}"
            if down_col not in df.columns:
                continue

            up_name = f"up_{horizon}_{th}"
            feats = target_features.get(up_name, list(target_features.values())[0] if target_features else [])
            params = target_params.get(up_name, DEFAULT_PARAMS)

            X_feat = df[feats].values
            y_all = df[down_col].values

            split_aucs = []
            oof_idx = []
            oof_preds = []
            oof_labels = []

            for s_idx, (test_start, test_end) in enumerate(splits):
                train_end = test_start - 2 * horizon
                if train_end < 10000:
                    continue
                data = split_data(X_feat, y_all, train_end, test_start, test_end, 2 * horizon)
                if data is None:
                    continue
                X_tr, y_tr, X_va, y_va, X_te, y_te, test_indices = data
                try:
                    m = train_lgb(X_tr, y_tr, X_va, y_va, params=params)
                    p = m.predict_proba(X_te)[:, 1]
                    split_aucs.append(auc_roc(y_te, p))
                    oof_idx.extend(test_indices.tolist())
                    oof_preds.extend(p.tolist())
                    oof_labels.extend(y_te.tolist())
                except Exception:
                    continue

            if split_aucs and np.mean(split_aucs) > 0.55:
                avg_auc = np.mean(split_aucs)
                if avg_auc > best_auc:
                    best_auc = avg_auc
                    best_down = {
                        "target_col": down_col, "features": feats,
                        "params": params, "avg_auc": avg_auc,
                        "threshold": th,
                        "oof_idx": np.array(oof_idx),
                        "oof_preds": np.array(oof_preds),
                        "oof_labels": np.array(oof_labels),
                    }

        if best_down is None:
            log.info(f"  Horizon {horizon}: No viable down-model (AUC < 0.55)")
            continue

        log.info(f"  Horizon {horizon}: down_{horizon}_{best_down['threshold']} "
                 f"AUC={best_down['avg_auc']:.4f}")

        # v2: Actually tune veto threshold by measuring improvement on OOF data
        # For each veto threshold, compute how many trades would be vetoed
        # and what the net return impact is
        oof_idx_arr = best_down["oof_idx"]
        oof_preds_arr = best_down["oof_preds"]

        # Build index-to-down-prob lookup
        down_prob_lookup = {}
        for i, idx in enumerate(oof_idx_arr):
            down_prob_lookup[idx] = oof_preds_arr[i]

        # Gather all trades from selected up-models for this horizon
        horizon_models = [n for n in selected
                         if any(f"_{horizon}_" in n for h_check in [horizon])]

        best_veto = 0.45  # default fallback
        best_veto_improvement = -999
        baseline_net = 0.0

        # Compute baseline (no veto)
        baseline_trades = []
        for name in horizon_models:
            for s_idx, r in all_results.get(name, {}).items():
                baseline_trades.extend(r["trades"])

        if baseline_trades:
            baseline_rets = np.array([t["net_ret"] for t in baseline_trades])
            baseline_net = baseline_rets.sum()
            baseline_wr = (baseline_rets > 0).mean()

            log.info(f"    Baseline (no veto): {len(baseline_trades)} trades, "
                     f"sum_ret={baseline_net:.4f}, wr={baseline_wr:.1%}")

            for veto_t in VETO_THRESHOLDS:
                # Filter: remove trades where down-model prob > veto_t
                filtered_trades = []
                vetoed = 0
                for t in baseline_trades:
                    idx = t["idx"]
                    if idx in down_prob_lookup and down_prob_lookup[idx] > veto_t:
                        vetoed += 1
                    else:
                        filtered_trades.append(t)

                if filtered_trades:
                    filt_rets = np.array([t["net_ret"] for t in filtered_trades])
                    filt_net = filt_rets.sum()
                    filt_wr = (filt_rets > 0).mean()
                    improvement = filt_net - baseline_net

                    log.info(f"    veto={veto_t:.2f}: kept={len(filtered_trades)}, "
                             f"vetoed={vetoed}, sum_ret={filt_net:.4f}, "
                             f"wr={filt_wr:.1%}, delta={improvement:+.4f}")

                    # Pick threshold that maximizes sum_ret improvement
                    # OR if no improvement, at least maximizes win rate
                    if improvement > best_veto_improvement:
                        best_veto_improvement = improvement
                        best_veto = veto_t
                else:
                    log.info(f"    veto={veto_t:.2f}: ALL trades vetoed")

        down_filters[horizon] = {
            "target_col": best_down["target_col"],
            "features": best_down["features"],
            "params": best_down["params"],
            "avg_auc": best_down["avg_auc"],
            "veto_thresh": best_veto,
            "oof_idx": oof_idx_arr,
            "oof_preds": oof_preds_arr,
        }
        down_oof_data[horizon] = down_prob_lookup

        log.info(f"    -> Best veto threshold: {best_veto:.2f} "
                 f"(improvement: {best_veto_improvement:+.4f})")

    log.info(f"  Down filters: {len(down_filters)} horizons covered")
    return down_filters, down_oof_data


# =============================================================================
# PHASE 5: PORTFOLIO BACKTESTING WITH DOWN-MODEL VETO
# =============================================================================
def phase5_portfolio_backtest(df, selected, all_results, saved_models,
                              calibrators, down_filters, down_oof_data, oof_predictions):
    log.info("")
    log.info("=" * 80)
    log.info("  PHASE 5: PORTFOLIO BACKTESTING WITH DOWN-MODEL VETO")
    log.info("=" * 80)

    # Quality weighting (recent-weighted)
    def compute_quality(name):
        nets = []
        sharpes = []
        for s_idx in range(N_SPLITS):
            if s_idx in all_results.get(name, {}):
                r = all_results[name][s_idx]
                if r["trades"]:
                    net = np.prod([1 + t["net_ret"] for t in r["trades"]]) - 1
                    weight = 2.0 if s_idx <= 2 else 1.0
                    nets.append((net, weight))
                    metrics = compute_split_metrics(r["trades"], r["horizon"])
                    if metrics["sharpe"] != 0:
                        sharpes.append((metrics["sharpe"], weight))
        if not nets:
            return 0.3
        w_pos = sum(w for n_, w in nets if n_ > 0)
        w_total = sum(w for _, w in nets)
        wf_rate = w_pos / w_total if w_total > 0 else 0
        w_sharpe = sum(s * w for s, w in sharpes) / sum(w for _, w in sharpes) if sharpes else 0
        return max(wf_rate * min(w_sharpe / 5.0, 1.5), 0.3)

    quality = {name: compute_quality(name) for name in selected}
    mean_q = np.mean(list(quality.values())) if quality else 1.0
    weights = {k: v / mean_q for k, v in quality.items()}

    log.info(f"  Quality weights (mean=1.0, floor=0.3):")
    top_weighted = sorted(selected, key=lambda n: -weights.get(n, 0))[:20]
    for name in top_weighted:
        log.info(f"    {name:<30s} q={quality[name]:.3f} w={weights[name]:.3f}")
    if len(selected) > 20:
        log.info(f"    ... ({len(selected) - 20} more)")

    # v2: Apply down-model veto to trades BEFORE portfolio backtesting
    def get_split_trades_with_veto(split_idx, model_names):
        """Get trades with down-model veto applied."""
        trades_list = []
        for name in model_names:
            if split_idx not in all_results.get(name, {}):
                continue
            r = all_results[name][split_idx]
            horizon = r["horizon"]

            # Apply down-model veto if available for this horizon
            if horizon in down_filters and horizon in down_oof_data:
                veto_thresh = down_filters[horizon]["veto_thresh"]
                down_lookup = down_oof_data[horizon]

                filtered_trades = []
                for t in r["trades"]:
                    idx = t["idx"]
                    # Only veto if we have a down-model prediction for this index
                    if idx in down_lookup and down_lookup[idx] > veto_thresh:
                        continue  # VETOED
                    filtered_trades.append(t)
                trades_list.append((name, filtered_trades, horizon))
            else:
                trades_list.append((name, r["trades"], horizon))
        return trades_list

    # Count vetoed trades
    total_vetoed = 0
    total_original = 0
    for s in range(N_SPLITS):
        for name in selected:
            if s in all_results.get(name, {}):
                r = all_results[name][s]
                original_count = len(r["trades"])
                total_original += original_count

                horizon = r["horizon"]
                if horizon in down_filters and horizon in down_oof_data:
                    veto_thresh = down_filters[horizon]["veto_thresh"]
                    down_lookup = down_oof_data[horizon]
                    vetoed = sum(1 for t in r["trades"]
                                if t["idx"] in down_lookup and down_lookup[t["idx"]] > veto_thresh)
                    total_vetoed += vetoed

    log.info(f"\n  Down-model veto: {total_vetoed}/{total_original} trades vetoed "
             f"({total_vetoed/max(total_original,1)*100:.1f}%)")

    # DD config sweep
    log.info(f"\n  === DD CONFIG SWEEP ({len(DD_CONFIGS)} configurations) ===")
    log.info(f"  {'Config':<20s} {'Pos/N':>6s} {'AvgNet':>9s} {'WrstDD':>9s} "
             f"{'Sharpe':>7s} {'Sortino':>8s} {'Calmar':>7s} {'PF':>6s} {'AvgN':>6s}")

    all_cfg_results = {}
    for dd_lim, cooldown, max_conc, scale, label in DD_CONFIGS:
        split_results = []
        for s in range(N_SPLITS):
            trades = get_split_trades_with_veto(s, selected)
            res = portfolio_backtest(
                trades, max_dd_pct=dd_lim, cooldown=cooldown,
                max_concurrent=max_conc, position_scale=scale,
                model_weights=weights)
            split_results.append(res)

        pos = sum(1 for r in split_results if r["net"] > 0)
        avg_net = np.mean([r["net"] for r in split_results])
        valid = [r for r in split_results if r["n"] > 0]
        worst_dd = min(r["max_dd"] for r in valid) if valid else 0
        avg_sharpe = np.mean([r["sharpe"] for r in valid]) if valid else 0
        avg_sortino = np.mean([r["sortino"] for r in valid]) if valid else 0
        avg_calmar = np.mean([r["calmar"] for r in valid]) if valid else 0
        avg_pf = np.mean([r["profit_factor"] for r in valid]) if valid else 0
        avg_n = np.mean([r["n"] for r in split_results])

        all_cfg_results[label] = {
            "pos": pos, "avg_net": avg_net, "worst_dd": worst_dd,
            "sharpe": avg_sharpe, "sortino": avg_sortino, "calmar": avg_calmar,
            "profit_factor": avg_pf, "avg_n": avg_n,
            "params": (dd_lim, cooldown, max_conc, scale),
            "split_results": split_results,
        }

        log.info(f"  {label:<20s} {pos:>3d}/10 {avg_net:>+8.2%} "
                 f"{worst_dd:>+8.2%} {avg_sharpe:>+6.1f} {avg_sortino:>+7.1f} "
                 f"{avg_calmar:>+6.0f} {avg_pf:>5.1f} {avg_n:>5.0f}")

    return weights, all_cfg_results


# =============================================================================
# PHASE 6: REGIME ANALYSIS
# =============================================================================
def phase6_regime_analysis(df, selected, all_results, weights, down_filters, down_oof_data):
    log.info("")
    log.info("=" * 80)
    log.info("  PHASE 6: REGIME ANALYSIS")
    log.info("=" * 80)

    regime_results = {}

    all_trades = []
    for name in selected:
        for s_idx in range(N_SPLITS):
            if s_idx in all_results.get(name, {}):
                r = all_results[name][s_idx]
                w = weights.get(name, 1.0)
                horizon = r["horizon"]

                for t in r["trades"]:
                    # Apply veto filter for consistency
                    idx = t["idx"]
                    if horizon in down_filters and horizon in down_oof_data:
                        veto_thresh = down_filters[horizon]["veto_thresh"]
                        down_lookup = down_oof_data[horizon]
                        if idx in down_lookup and down_lookup[idx] > veto_thresh:
                            continue

                    all_trades.append({
                        **t, "model": name, "horizon": horizon,
                        "model_weight": w, "split": s_idx,
                    })

    if not all_trades:
        log.info("  No trades to analyze")
        return {}

    all_trades.sort(key=lambda t: t["idx"])
    log.info(f"  Total trades across all splits (after veto): {len(all_trades):,}")

    close = df["close"].values

    # 1. Volatility regime
    log.info(f"\n  === VOLATILITY REGIME (24h realized vol terciles) ===")
    close_ret = pd.Series(close).pct_change()
    vol_24h = close_ret.rolling(288).std() * np.sqrt(288)
    vol_terciles = vol_24h.quantile([0.33, 0.67]).values

    vol_regime_trades = {"low": [], "mid": [], "high": []}
    for t in all_trades:
        idx = t["idx"]
        if idx < len(vol_24h) and not np.isnan(vol_24h.iloc[idx]):
            v = vol_24h.iloc[idx]
            if v < vol_terciles[0]:
                vol_regime_trades["low"].append(t)
            elif v < vol_terciles[1]:
                vol_regime_trades["mid"].append(t)
            else:
                vol_regime_trades["high"].append(t)

    for regime, trades in vol_regime_trades.items():
        if trades:
            rets = np.array([t["net_ret"] for t in trades])
            net = np.prod(1 + rets) - 1
            wr = (rets > 0).mean()
            log.info(f"    {regime:>5s}: n={len(trades):>5d}  net={net:>+8.2%}  wr={wr:.1%}")
    regime_results["volatility"] = {k: len(v) for k, v in vol_regime_trades.items()}

    # 2. Time-of-day
    log.info(f"\n  === TIME-OF-DAY (hour UTC) ===")
    hour_trades = {}
    for t in all_trades:
        h = t.get("hour", -1)
        if h >= 0:
            if h not in hour_trades:
                hour_trades[h] = []
            hour_trades[h].append(t["net_ret"])

    best_hours = []
    for h in sorted(hour_trades.keys()):
        rets = np.array(hour_trades[h])
        net = np.sum(rets)
        n = len(rets)
        log.info(f"    Hour {h:02d}: n={n:>4d}  sum_ret={net:>+8.4f}")
        best_hours.append((h, net, n))

    best_hours.sort(key=lambda x: -x[1])
    regime_results["time_of_day"] = {
        "best": best_hours[:3] if len(best_hours) >= 3 else best_hours,
        "worst": best_hours[-3:] if len(best_hours) >= 3 else [],
    }

    # 3. Trend state
    log.info(f"\n  === TREND STATE (close vs SMA-200) ===")
    sma200 = pd.Series(close).rolling(200).mean()

    trend_trades = {"uptrend": [], "downtrend": []}
    for t in all_trades:
        idx = t["idx"]
        if idx < len(sma200) and not np.isnan(sma200.iloc[idx]):
            if close[idx] > sma200.iloc[idx]:
                trend_trades["uptrend"].append(t)
            else:
                trend_trades["downtrend"].append(t)

    for state, trades in trend_trades.items():
        if trades:
            rets = np.array([t["net_ret"] for t in trades])
            net = np.prod(1 + rets) - 1
            wr = (rets > 0).mean()
            log.info(f"    {state:>10s}: n={len(trades):>5d}  net={net:>+8.2%}  wr={wr:.1%}")
    regime_results["trend"] = {k: len(v) for k, v in trend_trades.items()}

    # 4. Per-model contribution
    log.info(f"\n  === PER-MODEL P&L CONTRIBUTION (top 10) ===")
    model_pnl = {}
    for t in all_trades:
        m = t["model"]
        if m not in model_pnl:
            model_pnl[m] = {"total_ret": 0, "count": 0}
        model_pnl[m]["total_ret"] += t["net_ret"]
        model_pnl[m]["count"] += 1

    sorted_models = sorted(model_pnl.items(), key=lambda x: -x[1]["total_ret"])
    for name, info in sorted_models[:10]:
        log.info(f"    {name:<30s} sum_ret={info['total_ret']:>+8.4f} n={info['count']}")

    dead_weight = [(n, i) for n, i in sorted_models if i["total_ret"] < 0]
    if dead_weight:
        log.info(f"\n  Dead weight models ({len(dead_weight)}):")
        for name, info in dead_weight[:5]:
            log.info(f"    {name:<30s} sum_ret={info['total_ret']:>+8.4f}")

    regime_results["model_pnl"] = {n: i["total_ret"] for n, i in sorted_models}
    return regime_results


# =============================================================================
# PHASE 7: PRODUCTION CONFIG + MODEL SAVING
# =============================================================================
def phase7_save_production(df, selected, all_results, saved_models, weights,
                           all_cfg_results, target_params, target_features,
                           calibrators, down_filters, feature_stability):
    log.info("")
    log.info("=" * 80)
    log.info("  PHASE 7: PRODUCTION CONFIG + MODEL SAVING")
    log.info("=" * 80)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # v2: Pick best config with worst DD > -20% (the actual target)
    best_sharpe = -1
    best_cfg = None
    best_params = None

    for label, data in all_cfg_results.items():
        if data["pos"] >= 10 and data["worst_dd"] > -0.20 and data["sharpe"] > best_sharpe:
            best_sharpe = data["sharpe"]
            best_cfg = label
            best_params = data["params"]

    # Relax to -25% if nothing found
    if best_cfg is None:
        for label, data in all_cfg_results.items():
            if data["pos"] >= 10 and data["worst_dd"] > -0.25 and data["sharpe"] > best_sharpe:
                best_sharpe = data["sharpe"]
                best_cfg = label
                best_params = data["params"]

    # Further relax
    if best_cfg is None:
        for label, data in all_cfg_results.items():
            if data["pos"] >= 10 and data["sharpe"] > best_sharpe:
                best_sharpe = data["sharpe"]
                best_cfg = label
                best_params = data["params"]

    if best_cfg is None:
        for label, data in all_cfg_results.items():
            if data["sharpe"] > best_sharpe:
                best_sharpe = data["sharpe"]
                best_cfg = label
                best_params = data["params"]

    if best_cfg is None:
        log.error("  No valid configuration found!")
        return None, False, {}

    cfg = all_cfg_results[best_cfg]
    dd_lim, cooldown, max_conc, scale = best_params

    log.info(f"  Best config: {best_cfg}")
    log.info(f"    DD limit: {dd_lim}, Cooldown: {cooldown}, Max concurrent: {max_conc}, Scale: {scale}")
    log.info(f"    Sharpe: {cfg['sharpe']:.2f}, Worst DD: {cfg['worst_dd']:.2%}")

    # Identify safest and balanced configs
    safest_cfg = None
    safest_dd = -999
    balanced_cfg = None
    balanced_score = -1
    for label, data in all_cfg_results.items():
        if data["pos"] >= 10:
            if data["worst_dd"] > safest_dd:
                safest_dd = data["worst_dd"]
                safest_cfg = label
            score = data["sharpe"] * (1 - abs(data["worst_dd"]))
            if score > balanced_score:
                balanced_score = score
                balanced_cfg = label

    # Detailed per-split results
    log.info(f"\n  === PER-SPLIT RESULTS ({best_cfg}) ===")
    n_total = len(df)
    splits = wf_splits(n_total, N_SPLITS)
    total_compound = 1.0
    all_sharpes = []

    for s_idx, (test_start, test_end) in enumerate(splits):
        sr = cfg["split_results"][s_idx]
        dr = get_date_range(df.iloc[test_start:test_end])
        log.info(f"    Split {s_idx+1}: {dr}  n={sr['n']:>4d}  net={sr['net']:>+8.2%}  "
                 f"DD={sr['max_dd']:>+6.2%}  WR={sr['wr']:.1%}  Sharpe={sr['sharpe']:.1f}")
        total_compound *= (1 + sr["net"])
        if sr["n"] > 0:
            all_sharpes.append(sr["sharpe"])

    test_size = int(n_total * 0.05)
    years = N_SPLITS * (test_size / CANDLES_PER_DAY) / 365.25
    annual = (total_compound) ** (1 / years) - 1 if years > 0 else 0

    log.info(f"\n  === FINAL METRICS ===")
    log.info(f"  Positive splits: {cfg['pos']}/{N_SPLITS}")
    log.info(f"  Avg net per split: {cfg['avg_net']:>+.2%}")
    log.info(f"  Compounded return: {total_compound - 1:>+.1%}")
    log.info(f"  Annualized return: {annual:>+.0%}")
    log.info(f"  Worst max drawdown: {cfg['worst_dd']:.2%}")
    log.info(f"  Avg Sharpe: {cfg['sharpe']:.2f}")
    log.info(f"  Min Sharpe: {min(all_sharpes):.1f}" if all_sharpes else "  Min Sharpe: N/A")
    log.info(f"  Avg Sortino: {cfg['sortino']:.2f}")
    log.info(f"  Avg Calmar: {cfg['calmar']:.0f}")
    log.info(f"  Avg Profit Factor: {cfg['profit_factor']:.2f}")
    log.info(f"  Models: {len(selected)}")

    # Production readiness check
    log.info(f"\n  === PRODUCTION READINESS ===")
    checks = {
        "sharpe_4": cfg["sharpe"] >= 4.0,
        "10_10_pos": cfg["pos"] >= 10,
        "max_dd_20": cfg["worst_dd"] > -0.20,
        "annual_200": annual > 2.0,
        "pf_1.5": cfg["profit_factor"] > 1.5,
    }
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        log.info(f"    [{status}] {check}")

    is_production_ready = all(checks.values())
    if is_production_ready:
        log.info(f"  STATUS: *** ULTRA-PROFITABLE -- PRODUCTION READY ***")
    elif checks["10_10_pos"]:
        log.info(f"  STATUS: READY FOR PAPER TRADING (not yet ultra-profitable)")
    else:
        log.info(f"  STATUS: NEEDS IMPROVEMENT")

    # Save models
    log.info(f"\n  === SAVING MODELS ===")
    prod_config = {
        "version": VERSION,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "config_name": best_cfg,
        "dd_limit": dd_lim, "cooldown": cooldown,
        "max_concurrent": max_conc, "position_scale": scale,
        "fee_rt": FEE_RT,
        "quality_weighting": True,
        "model_weights": {k: float(v) for k, v in weights.items()},
        "metrics": {
            "sharpe": float(cfg["sharpe"]),
            "sortino": float(cfg["sortino"]),
            "calmar": float(cfg["calmar"]),
            "profit_factor": float(cfg["profit_factor"]),
            "pos_splits": cfg["pos"],
            "total_splits": N_SPLITS,
            "avg_net": float(cfg["avg_net"]),
            "worst_dd": float(cfg["worst_dd"]),
            "annualized_return": float(annual),
            "compounded_return": float(total_compound - 1),
        },
        "production_ready": is_production_ready,
        "models": [],
    }

    if safest_cfg:
        sd = all_cfg_results[safest_cfg]
        prod_config["safest_config"] = {
            "name": safest_cfg, "sharpe": float(sd["sharpe"]),
            "worst_dd": float(sd["worst_dd"]), "params": sd["params"],
        }
    if balanced_cfg:
        bd = all_cfg_results[balanced_cfg]
        prod_config["balanced_config"] = {
            "name": balanced_cfg, "sharpe": float(bd["sharpe"]),
            "worst_dd": float(bd["worst_dd"]), "params": bd["params"],
        }

    saved_count = 0
    for name in selected:
        if name not in saved_models:
            continue
        info = saved_models[name]

        lgb_path = os.path.join(MODEL_DIR, f"prod_{name}_lgb.pkl")
        with open(lgb_path, "wb") as mf:
            pickle.dump(info["model_lgb"], mf)

        entry = {
            "name": name, "lgb_file": f"prod_{name}_lgb.pkl",
            "features": info["features"], "target": info["target"],
            "target_name": info.get("target_name", ""),
            "horizon": info["horizon"],
            "prob_threshold": info["prob_threshold"],
            "top_pct": info["top_pct"], "auc": float(info["auc"]),
            "use_ensemble": info["use_ensemble"],
            "lgb_weight": float(info["lgb_weight"]),
            "quality_weight": float(weights.get(name, 1.0)),
        }

        if info.get("model_cb") is not None:
            cb_path = os.path.join(MODEL_DIR, f"prod_{name}_cb.pkl")
            with open(cb_path, "wb") as mf:
                pickle.dump(info["model_cb"], mf)
            entry["cb_file"] = f"prod_{name}_cb.pkl"

        if name in calibrators:
            cal_path = os.path.join(MODEL_DIR, f"prod_{name}_calibrator.pkl")
            with open(cal_path, "wb") as mf:
                pickle.dump(calibrators[name], mf)
            entry["calibrator_file"] = f"prod_{name}_calibrator.pkl"

        prod_config["models"].append(entry)
        saved_count += 1
        log.info(f"    Saved: {name} (AUC={info['auc']:.4f})")

    # Save down filters
    for horizon, df_info in down_filters.items():
        filter_entry = {
            "horizon": horizon,
            "target_col": df_info["target_col"],
            "features": df_info["features"],
            "veto_thresh": df_info["veto_thresh"],
            "avg_auc": float(df_info["avg_auc"]),
        }
        prod_config.setdefault("down_filters", []).append(filter_entry)

    # Save Optuna params
    optuna_path = os.path.join(MODEL_DIR, "optuna_params.json")
    with open(optuna_path, "w") as f:
        json.dump(target_params, f, indent=2, default=float)

    config_path = os.path.join(MODEL_DIR, "production_config.json")
    with open(config_path, "w") as f:
        json.dump(prod_config, f, indent=2, default=float)

    log.info(f"\n  Saved {saved_count} models to {MODEL_DIR}/")
    log.info(f"  Config: {config_path}")
    log.info(f"  Optuna params: {optuna_path}")

    return prod_config, is_production_ready, {
        "sharpe": cfg["sharpe"],
        "pos": cfg["pos"],
        "worst_dd": cfg["worst_dd"],
        "annual": annual,
        "profit_factor": cfg["profit_factor"],
        "n_models": len(selected),
    }


# =============================================================================
# PHASE 8: AUTO-GENERATE FINDINGS
# =============================================================================
def phase8_findings(selected, all_results, all_cfg_results, regime_results,
                    feature_stability, target_params, final_metrics,
                    pruned_feature_count, original_feature_count, down_filters,
                    is_production_ready):
    log.info("")
    log.info("=" * 80)
    log.info("  PHASE 8: AUTO-GENERATE FINDINGS")
    log.info("=" * 80)

    lines = []
    lines.append(f"# {VERSION} Findings -- Systematic ML Trading Model (Iteration 2)")
    lines.append(f"")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"")
    lines.append(f"## Changes from v1_all")
    lines.append(f"")
    lines.append(f"- Applied down-model veto filters during portfolio backtesting")
    lines.append(f"- Grid-searched veto thresholds per horizon on OOF data")
    lines.append(f"- Stricter model selection: 8/10 positive (was 7/10)")
    lines.append(f"- Model deduplication: max {MAX_CONFIGS_PER_TARGET} configs per target")
    lines.append(f"- Lower max_concurrent DD sweep (c4-c14 focus)")
    lines.append(f"- Warm-start from v1_all Optuna params")
    lines.append(f"")

    # Summary
    lines.append(f"## Summary")
    lines.append(f"")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Models selected | {len(selected)} |")
    lines.append(f"| Positive splits | {final_metrics['pos']}/{N_SPLITS} |")
    lines.append(f"| Avg Sharpe | {final_metrics['sharpe']:.2f} |")
    lines.append(f"| Worst DD | {final_metrics['worst_dd']:.2%} |")
    lines.append(f"| Annualized return | {final_metrics['annual']:.0%} |")
    lines.append(f"| Profit factor | {final_metrics['profit_factor']:.2f} |")
    lines.append(f"| Fee model | {FEE_RT*100:.2f}% RT |")
    lines.append(f"| Production ready | {'YES' if is_production_ready else 'NO'} |")
    lines.append(f"")

    # Feature pruning
    lines.append(f"## Phase 0: Feature Pruning")
    lines.append(f"")
    lines.append(f"- Original features: {original_feature_count}")
    lines.append(f"- After correlation pruning (|r|>{CORR_THRESHOLD}): {pruned_feature_count}")
    lines.append(f"- Reduction: {original_feature_count - pruned_feature_count} features removed "
                 f"({(original_feature_count - pruned_feature_count)/original_feature_count*100:.1f}%)")
    lines.append(f"")

    # Selected models
    lines.append(f"## Phase 3: Selected Models")
    lines.append(f"")
    lines.append(f"| Model | AUC | Pos/Total | Avg Net | Horizon |")
    lines.append(f"|-------|-----|-----------|---------|---------|")
    for name in sorted(selected):
        results = all_results.get(name, {})
        aucs = [r["auc"] for r in results.values()]
        avg_auc = np.mean(aucs) if aucs else 0
        nets = []
        for r in results.values():
            if r["trades"]:
                nets.append(np.prod([1 + t["net_ret"] for t in r["trades"]]) - 1)
        pos = sum(1 for n_ in nets if n_ > 0)
        avg_net = np.mean(nets) if nets else 0
        h = next((r["horizon"] for r in results.values()), "?")
        lines.append(f"| {name} | {avg_auc:.4f} | {pos}/{len(nets)} | {avg_net:+.2%} | {h} |")
    lines.append(f"")

    # Down filters
    lines.append(f"## Phase 4: Down-Model Filters")
    lines.append(f"")
    if down_filters:
        for h, info in down_filters.items():
            lines.append(f"- Horizon {h}: {info['target_col']} AUC={info['avg_auc']:.4f} "
                         f"veto_thresh={info['veto_thresh']:.2f}")
    else:
        lines.append(f"- No viable down-model filters found")
    lines.append(f"")

    # DD config comparison
    lines.append(f"## Phase 5: Portfolio Config Comparison")
    lines.append(f"")
    lines.append(f"| Config | Pos/10 | Avg Net | Worst DD | Sharpe | Sortino | PF |")
    lines.append(f"|--------|--------|---------|----------|--------|---------|----|")
    for label, data in sorted(all_cfg_results.items(), key=lambda x: -x[1]["sharpe"]):
        lines.append(f"| {label} | {data['pos']}/10 | {data['avg_net']:+.2%} | "
                     f"{data['worst_dd']:+.2%} | {data['sharpe']:.1f} | "
                     f"{data['sortino']:.1f} | {data['profit_factor']:.1f} |")
    lines.append(f"")

    # Regime analysis
    lines.append(f"## Phase 6: Regime Analysis")
    lines.append(f"")
    if regime_results:
        for regime_type, info in regime_results.items():
            lines.append(f"### {regime_type}")
            lines.append(f"```")
            lines.append(f"{json.dumps(info, indent=2, default=str)}")
            lines.append(f"```")
            lines.append(f"")

    # Improvement suggestions
    lines.append(f"## Auto-Generated Improvement Suggestions")
    lines.append(f"")
    suggestions = []

    if final_metrics["sharpe"] < 4.0:
        suggestions.append("- Sharpe below 4.0 -> Focus Optuna on best-performing horizons, try more trials")
    if final_metrics["worst_dd"] < -0.20:
        suggestions.append("- Worst DD still exceeds 20% -> Try even lower max_concurrent, scale=0.5")
    if final_metrics["worst_dd"] > -0.15:
        suggestions.append("- Worst DD is well-controlled -> Can safely increase max_concurrent")
    if final_metrics["profit_factor"] < 1.5:
        suggestions.append("- Profit factor < 1.5 -> Raise probability thresholds on weaker models")
    if is_production_ready:
        suggestions.append("- ALL criteria met -> Deploy to production / paper trading")

    for tgt_name, feat_counts in feature_stability.items():
        stable = [f for f, c in feat_counts.items() if c >= 8]
        total_tracked = len(feat_counts)
        if total_tracked > 0 and len(stable) / total_tracked < 0.3:
            suggestions.append(f"- Low feature stability for {tgt_name} -> More regularization")
            break

    if not suggestions:
        suggestions.append("- Model is performing well -- consider production deployment")

    for s in suggestions:
        lines.append(s)
    lines.append(f"")

    # Machine-readable block
    lines.append(f"## Machine-Readable Results")
    lines.append(f"")
    lines.append(f"```json")
    machine_data = {
        "version": VERSION,
        "production_ready": is_production_ready,
        "metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in final_metrics.items()},
        "n_selected": len(selected),
        "selected_models": selected,
        "suggestions": suggestions,
        "target_params_file": f"models/{VERSION}/optuna_params.json",
    }
    lines.append(json.dumps(machine_data, indent=2, default=str))
    lines.append(f"```")

    findings_text = "\n".join(lines)
    with open(FINDINGS_FILE, "w", encoding="utf-8") as f:
        f.write(findings_text)
    log.info(f"  Findings written to {FINDINGS_FILE}")

    return findings_text


# =============================================================================
# MAIN
# =============================================================================
def main():
    setup_logging()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log.info(f"BTC ML Training -- {VERSION}: DD Control & Veto Filters")
    log.info(f"{'='*80}")
    log.info(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log.info(f"CatBoost: {HAS_CATBOOST}  Isotonic: {HAS_ISOTONIC}")

    t0_total = time.time()
    log.info(f"\nLoading {FEATURES_FILE}...")
    df = pd.read_parquet(FEATURES_FILE)
    log.info(f"  Loaded: {len(df):,} rows x {len(df.columns)} cols")

    all_feat_cols = get_feature_cols(df)
    original_feature_count = len(all_feat_cols)
    target_cols = [c for c in df.columns if c.startswith("target_")]
    log.info(f"  Features: {original_feature_count}  Targets: {len(target_cols)}")

    # Phase 0
    pruned_features = phase0_load_and_prune(df, all_feat_cols)
    pruned_feature_count = len(pruned_features)

    # Phase 1
    candidates, target_aucs = phase1_target_sweep(df, pruned_features)
    if not candidates:
        log.error("No candidates passed Phase 1 screening!")
        return

    # Phase 2
    target_features, target_params, target_ensemble = phase2_feature_selection_optuna(
        df, candidates, pruned_features)

    # Phase 3
    (selected, all_results, oof_predictions, calibrators,
     saved_models, feature_stability) = phase3_walk_forward(
        df, candidates, target_features, target_params, target_ensemble)

    if not selected:
        log.error("No models passed Phase 3 selection!")
        return

    # Phase 4
    down_filters, down_oof_data = phase4_down_filters(
        df, selected, all_results, target_features, target_params, oof_predictions)

    # Phase 5
    weights, all_cfg_results = phase5_portfolio_backtest(
        df, selected, all_results, saved_models, calibrators,
        down_filters, down_oof_data, oof_predictions)

    # Phase 6
    regime_results = phase6_regime_analysis(
        df, selected, all_results, weights, down_filters, down_oof_data)

    # Phase 7
    prod_config, is_production_ready, final_metrics = phase7_save_production(
        df, selected, all_results, saved_models, weights,
        all_cfg_results, target_params, target_features,
        calibrators, down_filters, feature_stability)

    # Phase 8
    phase8_findings(
        selected, all_results, all_cfg_results, regime_results,
        feature_stability, target_params, final_metrics,
        pruned_feature_count, original_feature_count, down_filters,
        is_production_ready)

    elapsed_total = time.time() - t0_total
    log.info(f"\n{'='*80}")
    log.info(f"  {VERSION} COMPLETE -- {elapsed_total/3600:.1f} hours")
    log.info(f"  Models: {len(selected)}  Sharpe: {final_metrics['sharpe']:.2f}  "
             f"Annual: {final_metrics['annual']:.0%}")
    log.info(f"  Worst DD: {final_metrics['worst_dd']:.2%}")
    log.info(f"  Production ready: {is_production_ready}")
    log.info(f"  Log: {LOG_FILE}")
    log.info(f"  Findings: {FINDINGS_FILE}")
    log.info(f"{'='*80}")


if __name__ == "__main__":
    main()
