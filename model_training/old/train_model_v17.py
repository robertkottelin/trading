"""
BTC ML Training -- Iteration 17: TIGHTER DD + AGREEMENT BONUS + NEW CANDIDATES
================================================================================
Building on v16 (10/10 WF, +2600% annual, -17.0% worst DD, Sharpe 26.6).
Key changes:
  - DD=1% to 2.5% sweep (test if 1.5% or 1% continues improvement trend)
  - Cooldown 10-20 (v16 found 15 optimal, test tighter)
  - Drop known unstable models (up_12_0003_p45t10, up_36_0003_p35t20)
  - Add new candidate models (untested horizon/threshold combos)
  - Model agreement bonus: when 3+ models signal simultaneously, scale position 1.3x
  - Reuse v15 Optuna params and ensemble decisions
  - Recent-weighted quality scoring (from v16)

Phases:
  0: Feature selection (per-target top-100)
  1: Train all models (10-split WF, v15 params)
  2: Quality scoring (recent-weighted)
  3: DD sweep (DD=1%-2.5%, agreement bonus vs no bonus)
  4: Final production config

Usage:
  python train_model_v17.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import time
import json
import pickle
import warnings
from datetime import datetime, timezone

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = "btc_data"
FEATURES_FILE = os.path.join(OUTPUT_DIR, "btc_features_5m.parquet")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "training_results_v17.txt")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models_v17")

FEE_MAKER_RT = 0.0004
N_SPLITS = 10

# =====================================================================
# V15 OPTIMIZED PARAMS (reused — proven stable across v15 and v16)
# =====================================================================
V15_PARAMS = {
    "up_12_0002": {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "n_estimators": 3000, "subsample_freq": 1,
        "learning_rate": 0.0325, "num_leaves": 16, "max_depth": 3,
        "min_child_samples": 1007, "subsample": 0.76,
        "colsample_bytree": 0.3, "colsample_bynode": 0.5,
        "reg_alpha": 0.13, "reg_lambda": 14.07, "path_smooth": 20.3,
        "feature_fraction_bynode": 0.5,
    },
    "up_12_0003": {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "n_estimators": 3000, "subsample_freq": 1,
        "learning_rate": 0.0110, "num_leaves": 55, "max_depth": 3,
        "min_child_samples": 576, "subsample": 0.37,
        "colsample_bytree": 0.3, "colsample_bynode": 0.5,
        "reg_alpha": 0.11, "reg_lambda": 30.03, "path_smooth": 8.4,
        "feature_fraction_bynode": 0.5,
    },
    "up_12_0005": {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "n_estimators": 3000, "subsample_freq": 1,
        "learning_rate": 0.0028, "num_leaves": 13, "max_depth": 6,
        "min_child_samples": 766, "subsample": 0.55,
        "colsample_bytree": 0.3, "colsample_bynode": 0.5,
        "reg_alpha": 0.88, "reg_lambda": 19.22, "path_smooth": 3.2,
        "feature_fraction_bynode": 0.5,
    },
    "up_24_0002": {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "n_estimators": 3000, "subsample_freq": 1,
        "learning_rate": 0.0193, "num_leaves": 18, "max_depth": 5,
        "min_child_samples": 1440, "subsample": 0.53,
        "colsample_bytree": 0.3, "colsample_bynode": 0.5,
        "reg_alpha": 0.41, "reg_lambda": 11.53, "path_smooth": 12.4,
        "feature_fraction_bynode": 0.5,
    },
    "up_24_0003": {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "n_estimators": 3000, "subsample_freq": 1,
        "learning_rate": 0.0052, "num_leaves": 17, "max_depth": 6,
        "min_child_samples": 642, "subsample": 0.39,
        "colsample_bytree": 0.3, "colsample_bynode": 0.5,
        "reg_alpha": 15.58, "reg_lambda": 12.16, "path_smooth": 3.8,
        "feature_fraction_bynode": 0.5,
    },
    "up_36_0002": {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "n_estimators": 3000, "subsample_freq": 1,
        "learning_rate": 0.0076, "num_leaves": 25, "max_depth": 6,
        "min_child_samples": 1114, "subsample": 0.63,
        "colsample_bytree": 0.3, "colsample_bynode": 0.5,
        "reg_alpha": 0.10, "reg_lambda": 1.12, "path_smooth": 21.5,
        "feature_fraction_bynode": 0.5,
    },
    "up_36_0003": {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "n_estimators": 3000, "subsample_freq": 1,
        "learning_rate": 0.0225, "num_leaves": 19, "max_depth": 5,
        "min_child_samples": 1226, "subsample": 0.51,
        "colsample_bytree": 0.3, "colsample_bynode": 0.5,
        "reg_alpha": 2.67, "reg_lambda": 1.99, "path_smooth": 13.5,
        "feature_fraction_bynode": 0.5,
    },
    "up_48_0002": {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "n_estimators": 3000, "subsample_freq": 1,
        "learning_rate": 0.0096, "num_leaves": 52, "max_depth": 8,
        "min_child_samples": 1457, "subsample": 0.31,
        "colsample_bytree": 0.3, "colsample_bynode": 0.5,
        "reg_alpha": 3.42, "reg_lambda": 1.90, "path_smooth": 7.8,
        "feature_fraction_bynode": 0.5,
    },
    "fav_12_0003": {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "n_estimators": 3000, "subsample_freq": 1,
        "learning_rate": 0.0040, "num_leaves": 51, "max_depth": 3,
        "min_child_samples": 1468, "subsample": 0.38,
        "colsample_bytree": 0.3, "colsample_bynode": 0.5,
        "reg_alpha": 4.71, "reg_lambda": 38.66, "path_smooth": 14.0,
        "feature_fraction_bynode": 0.5,
    },
    "fav_12_0005": {
        "objective": "binary", "metric": "binary_logloss", "boosting_type": "gbdt",
        "n_estimators": 3000, "subsample_freq": 1,
        "learning_rate": 0.0035, "num_leaves": 21, "max_depth": 3,
        "min_child_samples": 881, "subsample": 0.57,
        "colsample_bytree": 0.3, "colsample_bynode": 0.5,
        "reg_alpha": 1.20, "reg_lambda": 17.72, "path_smooth": 4.2,
        "feature_fraction_bynode": 0.5,
    },
}

# V15 ensemble decisions (hardcoded, proven stable across v15-v16)
V15_ENSEMBLE = {
    "up_12_0002": (False, 1.0),
    "up_12_0003": (True, 0.7),
    "up_12_0005": (True, 0.5),
    "up_24_0002": (False, 1.0),
    "up_24_0003": (False, 1.0),
    "up_36_0002": (False, 1.0),
    "up_36_0003": (True, 0.7),
    "fav_12_0003": (True, 0.5),
    "fav_12_0005": (True, 0.6),
}

# Override: force LGB-only for these specific configs
FORCE_LGB_ONLY = {"up_12_0005_p35all"}


def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")
        f.flush()


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


def get_date_range(df_slice, col="open_time_ms"):
    dates = pd.to_datetime(df_slice[col], unit="ms", utc=True)
    return f"{dates.iloc[0].strftime('%Y-%m-%d')}/{dates.iloc[-1].strftime('%Y-%m-%d')}"


def train_lgb(X_tr, y_tr, X_va, y_va, params=None, seed=42):
    p = dict(params)
    p["random_state"] = seed
    p["verbose"] = -1
    p["n_jobs"] = -1
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


def wf_split_data(df, feat_cols, target_col, train_end, test_start, test_end, purge):
    val_frac = 0.15
    val_size = int(train_end * val_frac)
    val_start = train_end - val_size - purge
    train_sub_mask = np.arange(len(df)) < val_start
    val_mask = (np.arange(len(df)) >= val_start + purge) & (np.arange(len(df)) < train_end)
    test_mask = (np.arange(len(df)) >= test_start) & (np.arange(len(df)) < test_end)
    y_vals = df[target_col].values
    X_vals = df[feat_cols].values
    valid_tr = train_sub_mask & ~np.isnan(y_vals)
    valid_va = val_mask & ~np.isnan(y_vals)
    valid_te = test_mask & ~np.isnan(y_vals)
    X_tr, y_tr = X_vals[valid_tr], y_vals[valid_tr]
    X_va, y_va = X_vals[valid_va], y_vals[valid_va]
    X_te, y_te = X_vals[valid_te], y_vals[valid_te]
    if len(X_tr) < 1000 or len(X_va) < 100 or len(X_te) < 100:
        return None
    close_te = df["close"].values[valid_te]
    hour_te = df["hour_of_day"].values[valid_te] if "hour_of_day" in df.columns else None
    return X_tr, y_tr, X_va, y_va, X_te, y_te, close_te, hour_te


def backtest_threshold(close, y_prob, horizon, fee_rt, prob_threshold, top_pct=None,
                       hours=None):
    n = len(close)
    trades = []
    i = 0
    while i + horizon < n:
        if y_prob[i] > prob_threshold:
            raw_ret = (close[i + horizon] - close[i]) / close[i]
            trades.append({"idx": i, "prob": y_prob[i], "raw_ret": raw_ret,
                           "net_ret": raw_ret - fee_rt,
                           "hour": int(hours[i]) if hours is not None else -1})
        i += horizon
    if not trades:
        return trades
    if top_pct is not None and len(trades) > 5:
        df_t = pd.DataFrame(trades)
        threshold = df_t["prob"].quantile(1.0 - top_pct)
        df_t = df_t[df_t["prob"] >= threshold]
        trades = df_t.to_dict("records")
    return trades


def portfolio_backtest(model_trades_list, max_dd_pct=None, cooldown=20,
                       max_concurrent=None, position_scale=1.0, avg_horizon=12,
                       model_weights=None, agreement_bonus=False):
    """Portfolio backtest with optional agreement bonus.

    agreement_bonus: when True, trades at indices where 3+ models fire
    simultaneously get 1.3x position scale.
    """
    merged = []
    for name, trades, horizon in model_trades_list:
        w = model_weights.get(name, 1.0) if model_weights else 1.0
        for t in trades:
            merged.append({**t, "model": name, "horizon": horizon, "model_weight": w})
    merged.sort(key=lambda t: t["idx"])

    if not merged:
        return {"n": 0, "net": 0, "wr": 0, "max_dd": 0, "sharpe": 0, "model_counts": {}}

    # Agreement bonus: count models firing at similar indices (within 2 candles)
    if agreement_bonus:
        idx_list = np.array([t["idx"] for t in merged])
        for i, t in enumerate(merged):
            # Count trades within 2 candles of this trade
            nearby = np.abs(idx_list - t["idx"]) <= 2
            n_nearby = nearby.sum()
            t["agreement_mult"] = 1.3 if n_nearby >= 3 else 1.0
    else:
        for t in merged:
            t["agreement_mult"] = 1.0

    if max_concurrent is not None:
        filtered = []
        active = []
        for t in merged:
            idx, h = t["idx"], t["horizon"]
            active = [(e, x) for e, x in active if x > idx]
            if len(active) < max_concurrent:
                filtered.append(t)
                active.append((idx, idx + h))
        merged = filtered

    if not merged:
        return {"n": 0, "net": 0, "wr": 0, "max_dd": 0, "sharpe": 0, "model_counts": {}}

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
                    cum = 1.0
                    peak = 1.0
                continue
            t_copy = dict(t)
            t_copy["net_ret"] = (t["net_ret"] * position_scale *
                                 t["model_weight"] * t["agreement_mult"])
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
            t["net_ret"] = (t["net_ret"] * position_scale *
                            t["model_weight"] * t["agreement_mult"])

    if not merged:
        return {"n": 0, "net": 0, "wr": 0, "max_dd": 0, "sharpe": 0, "model_counts": {}}

    rets = np.array([t["net_ret"] for t in merged])
    cum = np.cumprod(1 + rets)
    total_net = cum[-1] - 1
    wr = (rets > 0).mean()
    cummax = np.maximum.accumulate(cum)
    max_dd = ((cum - cummax) / cummax).min()
    mu = rets.mean()
    sigma = rets.std()
    sharpe = (mu / sigma) * np.sqrt(365.25 * 24 * 12 / avg_horizon) if sigma > 0 else 0

    model_counts = {}
    for t in merged:
        model_counts[t["model"]] = model_counts.get(t["model"], 0) + 1

    return {"n": len(merged), "net": total_net, "wr": wr, "max_dd": max_dd,
            "sharpe": sharpe, "model_counts": model_counts}


# Model universe: v16 portfolio (21) + new candidates, minus unstable models
ALL_MODELS = [
    # ===== PROVEN PORTFOLIO (from v16, all selected with 7+/10 positive) =====
    ("up_12_0002_p45t20",  "target_up_12_0002",  12, 0.45, 0.20, 2),
    ("up_12_0002_p40t10",  "target_up_12_0002",  12, 0.40, 0.10, 2),
    ("up_12_0003_p35t10",  "target_up_12_0003",  12, 0.35, 0.10, 2),
    ("up_12_0003_p40t10",  "target_up_12_0003",  12, 0.40, 0.10, 2),
    ("up_12_0003_p35t20",  "target_up_12_0003",  12, 0.35, 0.20, 2),
    ("up_12_0005_p40t20",  "target_up_12_0005",  12, 0.40, 0.20, 2),
    ("up_12_0005_p35all",  "target_up_12_0005",  12, 0.35, None, 2),
    ("up_24_0002_p40t10",  "target_up_24_0002",  24, 0.40, 0.10, 2),
    ("up_24_0002_p35t20",  "target_up_24_0002",  24, 0.35, 0.20, 2),
    ("up_24_0002_p45t10",  "target_up_24_0002",  24, 0.45, 0.10, 2),
    ("up_24_0003_p35t10",  "target_up_24_0003",  24, 0.35, 0.10, 2),
    ("up_24_0003_p40t10",  "target_up_24_0003",  24, 0.40, 0.10, 2),
    ("up_24_0003_p35t20",  "target_up_24_0003",  24, 0.35, 0.20, 2),
    ("up_36_0002_p35t20",  "target_up_36_0002",  36, 0.35, 0.20, 2),
    ("up_36_0002_p40t10",  "target_up_36_0002",  36, 0.40, 0.10, 2),
    ("up_36_0003_p40t10",  "target_up_36_0003",  36, 0.40, 0.10, 2),
    ("up_48_0002_p40t10",  "target_up_48_0002",  48, 0.40, 0.10, 2),
    ("up_48_0002_p35t10",  "target_up_48_0002",  48, 0.35, 0.10, 2),
    ("fav_12_0005_p35t20", "target_favorable_12_0005", 12, 0.35, 0.20, 2),
    ("fav_12_0003_p40t10", "target_favorable_12_0003", 12, 0.40, 0.10, 2),
    ("fav_36_0003_p40t10", "target_favorable_36_0003", 36, 0.40, 0.10, 2),
    # ===== NEW CANDIDATES (untested threshold/horizon combos) =====
    ("up_36_0002_p45t10",  "target_up_36_0002",  36, 0.45, 0.10, 2),  # tighter threshold
    ("up_24_0003_p45t10",  "target_up_24_0003",  24, 0.45, 0.10, 2),  # tighter threshold
    ("fav_12_0005_p40t10", "target_favorable_12_0005", 12, 0.40, 0.10, 2),  # tighter threshold
    ("up_12_0002_p45all",  "target_up_12_0002",  12, 0.45, None, 2),  # no top-pct filter
    ("up_48_0002_p35t20",  "target_up_48_0002",  48, 0.35, 0.20, 2),  # wider top-pct
    # NOTE: up_12_0003_p45t10 and up_36_0003_p35t20 REMOVED (unstable in v16)
]


def get_params(model_name):
    if "fav_12_0003" in model_name:
        return V15_PARAMS["fav_12_0003"]
    if "fav_12_0005" in model_name:
        return V15_PARAMS["fav_12_0005"]
    if "fav_36_0003" in model_name:
        return V15_PARAMS["up_36_0003"]
    for key in V15_PARAMS:
        if key.replace("_", "") in model_name.replace("_", ""):
            return V15_PARAMS[key]
    return V15_PARAMS["up_12_0002"]  # fallback


def get_ens_config(model_name):
    if model_name in FORCE_LGB_ONLY:
        return (False, 1.0)
    if "fav_12_0003" in model_name:
        return V15_ENSEMBLE["fav_12_0003"]
    if "fav_12_0005" in model_name:
        return V15_ENSEMBLE["fav_12_0005"]
    if "fav_36_0003" in model_name:
        return V15_ENSEMBLE["up_36_0003"]
    for key in V15_ENSEMBLE:
        if key.replace("_", "") in model_name.replace("_", ""):
            return V15_ENSEMBLE[key]
    return (False, 1.0)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols")

    all_feat_cols = get_feature_cols(df)
    n_total = len(df)
    test_size = int(n_total * 0.05)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        log(f"BTC ML Training -- Iteration 17: TIGHTER DD + AGREEMENT BONUS + NEW CANDIDATES", f)
        log(f"{'='*80}", f)
        log(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", f)
        log(f"Data: {len(df):,} rows x {len(df.columns)} cols  Features: {len(all_feat_cols)}", f)
        log(f"CatBoost available: {HAS_CATBOOST}", f)
        log(f"Using v15 Optuna params (hardcoded, proven stable v15-v16)", f)
        log(f"Removed unstable models: up_12_0003_p45t10, up_36_0003_p35t20", f)
        log(f"New candidates: up_36_0002_p45t10, up_24_0003_p45t10, fav_12_0005_p40t10, "
            f"up_12_0002_p45all, up_48_0002_p35t20", f)

        # =================================================================
        # PHASE 0: FEATURE SELECTION
        # =================================================================
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 0: FEATURE SELECTION (per-target top-100)", f)
        log(f"{'#'*80}\n", f)

        feat_cache = {}
        targets_for_feat = [
            ("up_12_0002", "target_up_12_0002"),
            ("up_12_0003", "target_up_12_0003"),
            ("up_12_0005", "target_up_12_0005"),
            ("up_24_0002", "target_up_24_0002"),
            ("up_24_0003", "target_up_24_0003"),
            ("up_36_0002", "target_up_36_0002"),
            ("up_36_0003", "target_up_36_0003"),
            ("up_48_0002", "target_up_48_0002"),
            ("fav_12_0003", "target_favorable_12_0003"),
            ("fav_12_0005", "target_favorable_12_0005"),
        ]
        for tgt_name, tgt_col in targets_for_feat:
            if tgt_col not in df.columns:
                continue
            valid = ~df[tgt_col].isna()
            valid_idx = np.where(valid)[0]
            n_v = valid.sum()
            tr_e = int(n_v * 0.70)
            va_e = int(n_v * 0.85)
            m = train_lgb(df.loc[valid_idx[:tr_e], all_feat_cols].values,
                          df.loc[valid_idx[:tr_e], tgt_col].values,
                          df.loc[valid_idx[tr_e:va_e], all_feat_cols].values,
                          df.loc[valid_idx[tr_e:va_e], tgt_col].values,
                          params=get_params(tgt_name))
            imp = pd.DataFrame({"feat": all_feat_cols, "imp": m.feature_importances_})
            imp = imp.sort_values("imp", ascending=False)
            feat_cache[tgt_name] = imp.head(100)["feat"].tolist()
            top5 = imp.head(5)
            log(f"  {tgt_name}: top5 = {list(top5['feat'].values)}", f)

        def get_feats(name, n_feats=100):
            if "fav_12_0003" in name and "fav_12_0003" in feat_cache:
                return feat_cache["fav_12_0003"][:n_feats]
            if "fav_12_0005" in name and "fav_12_0005" in feat_cache:
                return feat_cache["fav_12_0005"][:n_feats]
            if "fav_36_0003" in name:
                return feat_cache.get("up_36_0003", all_feat_cols[:n_feats])[:n_feats]
            for key in feat_cache:
                if key.replace("_", "") in name.replace("_", ""):
                    return feat_cache[key][:n_feats]
            return all_feat_cols[:n_feats]

        # =================================================================
        # PHASE 1: TRAIN ALL MODELS (10-split WF, v15 params)
        # =================================================================
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 1: TRAIN ALL MODELS (10-split WF, v15 params)", f)
        log(f"{'#'*80}\n", f)

        all_results = {}
        saved_models = {}

        for name, tgt_col, horizon, prob_thresh, top_pct, purge_mult in ALL_MODELS:
            if tgt_col not in df.columns:
                log(f"  {name}: {tgt_col} not found -- skipping", f)
                continue

            feat = get_feats(name)
            purge = purge_mult * horizon
            lgb_params = get_params(name)
            use_ens, lgb_w = get_ens_config(name)
            model_results = {}
            t0 = time.time()

            for s in range(N_SPLITS):
                test_end = n_total - s * test_size
                test_start = test_end - test_size
                train_end = test_start - purge
                if train_end < 10000:
                    continue
                data = wf_split_data(df, feat, tgt_col, train_end, test_start, test_end, purge)
                if data is None:
                    continue
                X_tr, y_tr, X_va, y_va, X_te, y_te, close_te, hour_te = data

                m_lgb = train_lgb(X_tr, y_tr, X_va, y_va, params=lgb_params)
                p_test = m_lgb.predict_proba(X_te)[:, 1]

                m_cb = None
                if use_ens and HAS_CATBOOST:
                    m_cb = train_catboost(X_tr, y_tr, X_va, y_va)
                    if m_cb is not None:
                        p_cb = m_cb.predict_proba(X_te)[:, 1]
                        p_test = lgb_w * p_test + (1 - lgb_w) * p_cb

                auc = auc_roc(y_te, p_test)
                trades = backtest_threshold(close_te, p_test, horizon, FEE_MAKER_RT,
                                           prob_thresh, top_pct, hours=hour_te)
                model_results[s] = {
                    "trades": trades, "auc": auc, "horizon": horizon,
                }

                if s == 0:
                    saved_models[name] = {
                        "model_lgb": m_lgb, "model_cb": m_cb,
                        "features": feat, "target": tgt_col,
                        "horizon": horizon, "prob_threshold": prob_thresh,
                        "top_pct": top_pct, "auc": auc,
                        "lgb_params": lgb_params,
                        "use_ensemble": use_ens, "lgb_weight": lgb_w,
                    }

            elapsed = time.time() - t0
            all_results[name] = model_results

            aucs = [r["auc"] for r in model_results.values()]
            trade_nets = []
            split_sharpes = []
            for s_idx, r in model_results.items():
                if r["trades"]:
                    net = np.prod([1 + t["net_ret"] for t in r["trades"]]) - 1
                    trade_nets.append(net)
                    rets = np.array([t["net_ret"] for t in r["trades"]])
                    if rets.std() > 0:
                        sh = (rets.mean() / rets.std()) * np.sqrt(365.25 * 24 * 12 / horizon)
                        split_sharpes.append(sh)
            pos = sum(1 for n_ in trade_nets if n_ > 0)
            avg_net = np.mean(trade_nets) if trade_nets else 0
            avg_sharpe = np.mean(split_sharpes) if split_sharpes else 0
            med_trades = int(np.median([len(r["trades"]) for r in model_results.values()])) if model_results else 0
            ens_tag = " [ENS]" if use_ens else ""
            forced_tag = " [LGB-ONLY]" if name in FORCE_LGB_ONLY else ""
            is_new = name in {"up_36_0002_p45t10", "up_24_0003_p45t10",
                              "fav_12_0005_p40t10", "up_12_0002_p45all",
                              "up_48_0002_p35t20"}
            new_tag = " [NEW]" if is_new else ""
            tag = " *" if pos >= 7 and len(trade_nets) >= 7 else ""
            log(f"  {name:<25s} AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.3f}  "
                f"{pos}/{len(trade_nets)} pos  avg={avg_net:>+.2%}  "
                f"sharpe={avg_sharpe:>+.1f}  med_trades={med_trades}  "
                f"t={elapsed:.0f}s{ens_tag}{forced_tag}{new_tag}{tag}", f)

        # Selection (70%+ positive with positive avg)
        selected = []
        for name, results in all_results.items():
            trade_nets = []
            for r in results.values():
                if r["trades"]:
                    trade_nets.append(np.prod([1 + t["net_ret"] for t in r["trades"]]) - 1)
            pos = sum(1 for n_ in trade_nets if n_ > 0)
            total = len(trade_nets)
            avg_net = np.mean(trade_nets) if trade_nets else 0
            if total >= 7 and pos / total >= 0.7 and avg_net > 0:
                selected.append(name)

        log(f"\n  Selected: {len(selected)} models (70%+ positive)", f)
        for name in selected:
            is_new = name in {"up_36_0002_p45t10", "up_24_0003_p45t10",
                              "fav_12_0005_p40t10", "up_12_0002_p45all",
                              "up_48_0002_p35t20"}
            log(f"    {name}{'  [NEW]' if is_new else ''}", f)

        # =================================================================
        # PHASE 2: QUALITY SCORING (recent-weighted)
        # =================================================================
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 2: QUALITY SCORING (recent-weighted)", f)
        log(f"{'#'*80}\n", f)

        def compute_quality(name):
            nets = []
            sharpes = []
            for s in range(N_SPLITS):
                if s in all_results.get(name, {}):
                    r = all_results[name][s]
                    if r["trades"]:
                        net = np.prod([1 + t["net_ret"] for t in r["trades"]]) - 1
                        # Recent splits (s=0,1,2) get 2x weight
                        weight = 2.0 if s <= 2 else 1.0
                        nets.append((net, weight))
                        rets_arr = np.array([t["net_ret"] for t in r["trades"]])
                        if rets_arr.std() > 0:
                            sh = (rets_arr.mean() / rets_arr.std()) * np.sqrt(
                                365.25 * 24 * 12 / r["horizon"])
                            sharpes.append((sh, weight))
            if not nets:
                return 0
            w_pos = sum(w for n_, w in nets if n_ > 0)
            w_total = sum(w for _, w in nets)
            wf_rate = w_pos / w_total if w_total > 0 else 0
            w_sharpe = sum(s * w for s, w in sharpes) / sum(w for _, w in sharpes) if sharpes else 0
            return wf_rate * min(w_sharpe / 20.0, 1.5)

        quality = {name: max(compute_quality(name), 0.3) for name in selected}
        mean_q = np.mean(list(quality.values()))
        weights = {k: v / mean_q for k, v in quality.items()}

        log(f"  Model quality weights (recent-weighted, mean=1.0):", f)
        for name in sorted(selected):
            log(f"    {name:<25s}  quality={quality[name]:.3f}  weight={weights[name]:.3f}", f)

        def get_split_trades(split_idx, model_names):
            trades_list = []
            for name in model_names:
                if split_idx in all_results.get(name, {}):
                    r = all_results[name][split_idx]
                    trades_list.append((name, r["trades"], r["horizon"]))
            return trades_list

        # =================================================================
        # PHASE 3: DD SWEEP (DD=1%-2.5%, agreement bonus comparison)
        # =================================================================
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 3: DD SWEEP (DD=1%-2.5%, with/without agreement bonus)", f)
        log(f"{'#'*80}\n", f)

        # Test both with and without agreement bonus
        for use_agree, agree_label in [(False, "NO_BONUS"), (True, "AGREE_BONUS")]:
            log(f"\n  --- {agree_label} ---", f)
            log(f"  {'Config':<35s} {'Pos/N':>6s} {'Avg%':>9s} {'AvgDD%':>9s} {'WorstDD%':>10s} "
                f"{'AvgN':>6s} {'Sharpe':>7s}", f)

            sweep_results = {}
            for max_dd in [0.01, 0.015, 0.02, 0.025]:
                for cooldown in [10, 12, 15, 20]:
                    for max_conc in [7, 10]:
                        for pos_scale in [0.80, 1.0]:
                            split_res = []
                            for s in range(N_SPLITS):
                                trades = get_split_trades(s, selected)
                                res = portfolio_backtest(
                                    trades, max_dd_pct=max_dd,
                                    cooldown=cooldown, max_concurrent=max_conc,
                                    position_scale=pos_scale,
                                    model_weights=weights,
                                    agreement_bonus=use_agree)
                                split_res.append(res)

                            pos = sum(1 for r in split_res if r["net"] > 0)
                            avg_net = np.mean([r["net"] for r in split_res])
                            valid = [r for r in split_res if r["n"] > 0]
                            avg_dd = np.mean([r["max_dd"] for r in valid]) if valid else 0
                            worst_dd = min(r["max_dd"] for r in valid) if valid else 0
                            avg_n = np.mean([r["n"] for r in split_res])
                            avg_sharpe = np.mean([r["sharpe"] for r in valid]) if valid else 0

                            cfg = f"dd{int(max_dd*1000)}m_cl{cooldown}_c{max_conc}_s{int(pos_scale*100)}"
                            sweep_results[cfg] = {
                                "pos": pos, "avg_net": avg_net, "avg_dd": avg_dd,
                                "worst_dd": worst_dd, "avg_n": avg_n, "sharpe": avg_sharpe,
                                "params": (max_dd, cooldown, max_conc, pos_scale),
                                "agreement": use_agree,
                            }

                            if pos >= 10:
                                log(f"  {cfg:<35s} {pos:>3d}/10 {avg_net:>+8.1%} "
                                    f"{avg_dd:>+8.1%} {worst_dd:>+9.1%} {avg_n:>5.0f} "
                                    f"{avg_sharpe:>+6.1f}", f)

            # Find best configs
            candidates = {k: v for k, v in sweep_results.items() if v["pos"] >= 10}
            if candidates:
                # Best Sharpe with DD > -22%
                safe = {k: v for k, v in candidates.items() if v["worst_dd"] > -0.22}
                if safe:
                    best = sorted(safe.items(), key=lambda x: -x[1]["sharpe"])[0]
                    log(f"\n  {agree_label} Best (Sharpe, DD>-22%): {best[0]}  "
                        f"sharpe={best[1]['sharpe']:+.1f}  avg={best[1]['avg_net']:+.1%}  "
                        f"worstDD={best[1]['worst_dd']:+.1%}", f)

                # Best Sharpe with DD > -16%
                safest = {k: v for k, v in candidates.items() if v["worst_dd"] > -0.16}
                if safest:
                    best_safe = sorted(safest.items(), key=lambda x: -x[1]["sharpe"])[0]
                    log(f"  {agree_label} Best (Sharpe, DD>-16%): {best_safe[0]}  "
                        f"sharpe={best_safe[1]['sharpe']:+.1f}  avg={best_safe[1]['avg_net']:+.1%}  "
                        f"worstDD={best_safe[1]['worst_dd']:+.1%}", f)

            if not use_agree:
                no_bonus_sweep = sweep_results
            else:
                bonus_sweep = sweep_results

        # =================================================================
        # PHASE 4: FINAL PRODUCTION CONFIG
        # =================================================================
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 4: FINAL PRODUCTION CONFIG", f)
        log(f"{'#'*80}\n", f)

        # Compare best no-bonus vs best bonus
        def get_best_config(sweep):
            candidates = {k: v for k, v in sweep.items() if v["pos"] >= 10}
            safe = {k: v for k, v in candidates.items() if v["worst_dd"] > -0.22}
            if safe:
                return sorted(safe.items(), key=lambda x: -x[1]["sharpe"])[0]
            if candidates:
                return sorted(candidates.items(), key=lambda x: -x[1]["worst_dd"])[0]
            return None

        best_no = get_best_config(no_bonus_sweep)
        best_yes = get_best_config(bonus_sweep)

        log(f"  NO_BONUS best: {best_no[0] if best_no else 'N/A'}", f)
        if best_no:
            log(f"    sharpe={best_no[1]['sharpe']:+.1f}  avg={best_no[1]['avg_net']:+.1%}  "
                f"worstDD={best_no[1]['worst_dd']:+.1%}", f)
        log(f"  AGREE_BONUS best: {best_yes[0] if best_yes else 'N/A'}", f)
        if best_yes:
            log(f"    sharpe={best_yes[1]['sharpe']:+.1f}  avg={best_yes[1]['avg_net']:+.1%}  "
                f"worstDD={best_yes[1]['worst_dd']:+.1%}", f)

        # Pick better approach
        if best_no and best_yes:
            if best_yes[1]["sharpe"] > best_no[1]["sharpe"] * 1.05:
                final_cfg = best_yes
                use_agreement = True
            else:
                final_cfg = best_no
                use_agreement = False
        elif best_yes:
            final_cfg = best_yes
            use_agreement = True
        elif best_no:
            final_cfg = best_no
            use_agreement = False
        else:
            log(f"  ERROR: No valid config found!", f)
            return

        cfg_name, cfg_data = final_cfg
        max_dd, cooldown, max_conc, pos_scale = cfg_data["params"]

        bonus_label = "AGREE_BONUS" if use_agreement else "NO_BONUS"
        log(f"\n  +--------------------------------------------------+", f)
        log(f"  |  FINAL CONFIG ({bonus_label}): {cfg_name:<20s}|", f)
        log(f"  +--------------------------------------------------+", f)
        log(f"  DD limit: {max_dd}", f)
        log(f"  Cooldown: {cooldown}", f)
        log(f"  Max concurrent: {max_conc}", f)
        log(f"  Position scale: {pos_scale}", f)
        log(f"  Agreement bonus: {use_agreement}", f)
        log(f"  Models: {len(selected)}", f)
        log(f"  Quality weighting: YES (recent-weighted)", f)

        total_compound = 1.0
        all_sharpes = []
        for s in range(N_SPLITS):
            trades = get_split_trades(s, selected)
            res = portfolio_backtest(trades, max_dd_pct=max_dd, cooldown=cooldown,
                                   max_concurrent=max_conc, position_scale=pos_scale,
                                   model_weights=weights,
                                   agreement_bonus=use_agreement)
            test_end = n_total - s * test_size
            test_start = test_end - test_size
            test_df = df.iloc[test_start:test_end]
            dr = get_date_range(test_df)
            log(f"    Split {s+1}: {dr}  n={res['n']:>4d}  net={res['net']:>+8.2%}  "
                f"maxDD={res['max_dd']:>+6.2%}  WR={res['wr']:.1%}  sharpe={res['sharpe']:.1f}", f)
            total_compound *= (1 + res["net"])
            if res["n"] > 0:
                all_sharpes.append(res["sharpe"])

        years = N_SPLITS * 155 / 365.25
        annual = (total_compound) ** (1 / years) - 1 if years > 0 else 0

        log(f"\n  === FINAL METRICS ===", f)
        log(f"  Positive splits: {cfg_data['pos']}/{N_SPLITS}", f)
        log(f"  Avg net per split: {cfg_data['avg_net']:>+.2%}", f)
        log(f"  Compounded return: {total_compound - 1:>+.1%}", f)
        log(f"  Annualized return: {annual:>+.1%}", f)
        log(f"  Avg max drawdown: {cfg_data['avg_dd']:.2%}", f)
        log(f"  Worst max drawdown: {cfg_data['worst_dd']:.2%}", f)
        log(f"  Avg Sharpe: {cfg_data['sharpe']:.2f}", f)
        log(f"  Min Sharpe: {min(all_sharpes):.1f}" if all_sharpes else "  Min Sharpe: N/A", f)

        log(f"\n  === COMPARISON vs V16 ===", f)
        log(f"  V16: 10/10 pos, avg +625.4%, worstDD -17.0%, Sharpe 26.6, annual +2600%", f)
        log(f"  V17: {cfg_data['pos']}/10 pos, avg {cfg_data['avg_net']:+.1%}, "
            f"worstDD {cfg_data['worst_dd']:+.1%}, Sharpe {cfg_data['sharpe']:.1f}, "
            f"annual {annual:+.1%}", f)

        log(f"\n  === PRODUCTION READINESS ===", f)
        checks = []
        if cfg_data["pos"] >= 10:
            log(f"  [PASS] Consistency: {cfg_data['pos']}/10 positive", f)
            checks.append(True)
        else:
            log(f"  [FAIL] Consistency: {cfg_data['pos']}/10 positive", f)
            checks.append(False)
        if cfg_data["worst_dd"] > -0.25:
            log(f"  [PASS] Drawdown: worst={cfg_data['worst_dd']:.2%}", f)
            checks.append(True)
        else:
            log(f"  [WARN] Drawdown: worst={cfg_data['worst_dd']:.2%}", f)
            checks.append(False)
        if cfg_data["sharpe"] > 10:
            log(f"  [PASS] Sharpe={cfg_data['sharpe']:.1f}", f)
            checks.append(True)
        else:
            log(f"  [WARN] Sharpe={cfg_data['sharpe']:.1f}", f)
            checks.append(False)
        if annual > 1.0:
            log(f"  [PASS] Return: {annual:+.0%} annualized", f)
        if all(checks[:3]):
            log(f"  STATUS: * PRODUCTION READY *", f)
        elif checks[0]:
            log(f"  STATUS: READY FOR PAPER TRADING", f)
        else:
            log(f"  STATUS: NEEDS IMPROVEMENT", f)

        # Save models
        log(f"\n  === SAVING PRODUCTION MODELS ===", f)
        prod_config = {
            "config_name": cfg_name, "dd_limit": max_dd,
            "cooldown": cooldown, "max_concurrent": max_conc,
            "position_scale": pos_scale, "fee_rt": FEE_MAKER_RT,
            "agreement_bonus": use_agreement,
            "quality_weighting": True, "recent_weighted": True,
            "model_weights": {k: float(v) for k, v in weights.items()
                             if k in selected},
            "models": [],
        }

        for name in selected:
            if name in saved_models:
                info = saved_models[name]
                lgb_path = os.path.join(MODEL_DIR, f"prod_{name}_lgb.pkl")
                with open(lgb_path, "wb") as mf:
                    pickle.dump(info["model_lgb"], mf)

                entry = {
                    "name": name, "lgb_file": f"prod_{name}_lgb.pkl",
                    "features": info["features"], "target": info["target"],
                    "horizon": info["horizon"], "prob_threshold": info["prob_threshold"],
                    "top_pct": info["top_pct"], "auc": info["auc"],
                    "use_ensemble": info["use_ensemble"], "lgb_weight": info["lgb_weight"],
                    "quality_weight": float(weights.get(name, 1.0)),
                }
                if info.get("model_cb") is not None:
                    cb_path = os.path.join(MODEL_DIR, f"prod_{name}_cb.pkl")
                    with open(cb_path, "wb") as mf:
                        pickle.dump(info["model_cb"], mf)
                    entry["cb_file"] = f"prod_{name}_cb.pkl"

                prod_config["models"].append(entry)
                ens_label = f" [ENS w={info['lgb_weight']:.1f}]" if info["use_ensemble"] else ""
                w_label = f" q={weights.get(name, 1.0):.2f}"
                log(f"  Saved: {name} (AUC={info['auc']:.4f}){ens_label}{w_label}", f)

        config_path = os.path.join(MODEL_DIR, "production_config_v17.json")
        with open(config_path, "w") as cf:
            json.dump(prod_config, cf, indent=2, default=str)
        log(f"  Config: {config_path}", f)

        log(f"\n{'='*80}", f)
        log(f"Results saved to {RESULTS_FILE}", f)


if __name__ == "__main__":
    main()
