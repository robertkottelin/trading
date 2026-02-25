"""
BTC ML Training -- Iteration 21: AGGRESSIVE TARGETS + NEW 30-MIN TIMEFRAME
================================================================================
Building on v20 (Sharpe 39.9, 31 models). Key finding from v20: "up" family
targets dominate (5/5 passed), and up_48_0003 achieved 10/10 perfect consistency.
Higher threshold + longer horizon = untapped alpha.

This iteration explores:
  - up_48_0005, up_48_001: higher thresholds at 4h (since up_48_0003 was 10/10)
  - up_24_001, up_36_001: 1% moves at 2h/3h
  - up_6_0002, up_6_0003, up_6_0005: 30-min horizon (new timeframe family!)
  - 13 new candidate models + 31 existing from v20

Phases:
  0: Feature selection (per-target top-100)
  1: Train all models — existing 31 + 13 new candidates (10-split WF)
  2: Quality scoring (recent-weighted)
  3: Portfolio comparison: existing 31 vs expanded with new candidates
  4: Final production config

Usage:
  python train_model_v21.py
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
RESULTS_FILE = os.path.join(OUTPUT_DIR, "training_results_v21.txt")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models_v21")

FEE_MAKER_RT = 0.0004
N_SPLITS = 10

# V15 OPTIMIZED PARAMS (proven stable v15-v20)
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
                       model_weights=None):
    merged = []
    for name, trades, horizon in model_trades_list:
        w = model_weights.get(name, 1.0) if model_weights else 1.0
        for t in trades:
            merged.append({**t, "model": name, "horizon": horizon, "model_weight": w})
    merged.sort(key=lambda t: t["idx"])

    if not merged:
        return {"n": 0, "net": 0, "wr": 0, "max_dd": 0, "sharpe": 0, "model_counts": {}}

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


# Existing 31 models from v20
EXISTING_MODELS = [
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
    ("up_36_0002_p45t10",  "target_up_36_0002",  36, 0.45, 0.10, 2),
    ("up_24_0003_p45t10",  "target_up_24_0003",  24, 0.45, 0.10, 2),
    ("up_12_0002_p45all",  "target_up_12_0002",  12, 0.45, None, 2),
    ("up_48_0002_p35t20",  "target_up_48_0002",  48, 0.35, 0.20, 2),
    # v20 new models
    ("up_48_0003_p35t10",  "target_up_48_0003",  48, 0.35, 0.10, 2),
    ("up_48_0003_p40t10",  "target_up_48_0003",  48, 0.40, 0.10, 2),
    ("up_24_0005_p35t10",  "target_up_24_0005",  24, 0.35, 0.10, 2),
    ("up_24_0005_p40t10",  "target_up_24_0005",  24, 0.40, 0.10, 2),
    ("up_36_0005_p35t10",  "target_up_36_0005",  36, 0.35, 0.10, 2),
    ("fav_36_0005_p35t10", "target_favorable_36_0005", 36, 0.35, 0.10, 2),
]

# NEW candidate models — aggressive thresholds + 30min timeframe
NEW_CANDIDATES = [
    # 4h higher thresholds (up_48_0003 was 10/10 — push further)
    ("up_48_0005_p35t10",  "target_up_48_0005",  48, 0.35, 0.10, 2),
    ("up_48_0005_p40t10",  "target_up_48_0005",  48, 0.40, 0.10, 2),
    ("up_48_001_p35t10",   "target_up_48_001",   48, 0.35, 0.10, 2),
    # 2h/3h 1% threshold
    ("up_24_001_p35t10",   "target_up_24_001",   24, 0.35, 0.10, 2),
    ("up_24_001_p40t10",   "target_up_24_001",   24, 0.40, 0.10, 2),
    ("up_36_001_p35t10",   "target_up_36_001",   36, 0.35, 0.10, 2),
    ("up_36_001_p40t10",   "target_up_36_001",   36, 0.40, 0.10, 2),
    # 30-min horizon (entirely new timeframe family!)
    ("up_6_0002_p40t10",   "target_up_6_0002",   6, 0.40, 0.10, 2),
    ("up_6_0002_p35t20",   "target_up_6_0002",   6, 0.35, 0.20, 2),
    ("up_6_0003_p35t10",   "target_up_6_0003",   6, 0.35, 0.10, 2),
    ("up_6_0003_p40t10",   "target_up_6_0003",   6, 0.40, 0.10, 2),
    ("up_6_0005_p35t10",   "target_up_6_0005",   6, 0.35, 0.10, 2),
    ("up_6_0005_p40t10",   "target_up_6_0005",   6, 0.40, 0.10, 2),
]

ALL_MODELS = EXISTING_MODELS + NEW_CANDIDATES


def get_params(model_name):
    """Map model name to closest available Optuna params."""
    # 30-min targets → use closest 1h params
    if "up_6_0002" in model_name:
        return V15_PARAMS["up_12_0002"]
    if "up_6_0003" in model_name:
        return V15_PARAMS["up_12_0003"]
    if "up_6_0005" in model_name:
        return V15_PARAMS["up_12_0005"]
    # 4h higher thresholds → use up_48_0002
    if "up_48_0005" in model_name or "up_48_001" in model_name:
        return V15_PARAMS["up_48_0002"]
    if "up_48_0003" in model_name:
        return V15_PARAMS["up_48_0002"]
    # 1% thresholds → use highest available threshold at that horizon
    if "up_24_001" in model_name:
        return V15_PARAMS["up_24_0003"]
    if "up_36_001" in model_name:
        return V15_PARAMS["up_36_0003"]
    if "up_24_0005" in model_name:
        return V15_PARAMS["up_24_0003"]
    if "up_36_0005" in model_name:
        return V15_PARAMS["up_36_0003"]
    # Existing targets
    if "fav_12_0003" in model_name:
        return V15_PARAMS["fav_12_0003"]
    if "fav_12_0005" in model_name:
        return V15_PARAMS["fav_12_0005"]
    if "fav_36_0003" in model_name or "fav_36_0005" in model_name:
        return V15_PARAMS["up_36_0003"]
    for key in V15_PARAMS:
        if key.replace("_", "") in model_name.replace("_", ""):
            return V15_PARAMS[key]
    return V15_PARAMS["up_12_0002"]


def get_ens_config(model_name):
    """Map model name to ensemble config. All new 'up' targets are LGB-only."""
    if model_name in FORCE_LGB_ONLY:
        return (False, 1.0)
    # All new 'up' candidates: LGB-only (v20 showed up targets work LGB-only)
    for prefix in ["up_48_0005", "up_48_001", "up_24_001", "up_36_001",
                    "up_6_0002", "up_6_0003", "up_6_0005"]:
        if prefix in model_name:
            return (False, 1.0)
    # V20 new targets (LGB-only for up, ensemble for fav)
    if "up_48_0003" in model_name or "up_24_0005" in model_name or "up_36_0005" in model_name:
        return (False, 1.0)
    if "fav_36_0005" in model_name:
        return (True, 0.7)
    # Existing ensemble decisions
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

    new_candidate_names = {m[0] for m in NEW_CANDIDATES}

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        log(f"BTC ML Training -- Iteration 21: AGGRESSIVE TARGETS + NEW 30-MIN TIMEFRAME", f)
        log(f"{'='*80}", f)
        log(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", f)
        log(f"Data: {len(df):,} rows x {len(df.columns)} cols  Features: {len(all_feat_cols)}", f)
        log(f"CatBoost available: {HAS_CATBOOST}", f)
        log(f"Existing models: {len(EXISTING_MODELS)}, New candidates: {len(NEW_CANDIDATES)}", f)
        log(f"New targets: up_48_0005, up_48_001, up_24_001, up_36_001, up_6_0002, up_6_0003, up_6_0005", f)

        # Phase 0: Feature selection
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 0: FEATURE SELECTION (per-target top-100)", f)
        log(f"{'#'*80}\n", f)

        feat_cache = {}
        # All unique targets including new ones
        targets_for_feat = [
            ("up_12_0002", "target_up_12_0002"),
            ("up_12_0003", "target_up_12_0003"),
            ("up_12_0005", "target_up_12_0005"),
            ("up_24_0002", "target_up_24_0002"),
            ("up_24_0003", "target_up_24_0003"),
            ("up_24_0005", "target_up_24_0005"),
            ("up_36_0002", "target_up_36_0002"),
            ("up_36_0003", "target_up_36_0003"),
            ("up_36_0005", "target_up_36_0005"),
            ("up_48_0002", "target_up_48_0002"),
            ("up_48_0003", "target_up_48_0003"),
            ("fav_12_0003", "target_favorable_12_0003"),
            ("fav_12_0005", "target_favorable_12_0005"),
            ("fav_36_0003", "target_favorable_36_0003"),
            ("fav_36_0005", "target_favorable_36_0005"),
            # New targets for v21
            ("up_48_0005", "target_up_48_0005"),
            ("up_48_001",  "target_up_48_001"),
            ("up_24_001",  "target_up_24_001"),
            ("up_36_001",  "target_up_36_001"),
            ("up_6_0002",  "target_up_6_0002"),
            ("up_6_0003",  "target_up_6_0003"),
            ("up_6_0005",  "target_up_6_0005"),
        ]
        existing_targets = {"up_12_0002", "up_12_0003", "up_12_0005",
                           "up_24_0002", "up_24_0003", "up_24_0005",
                           "up_36_0002", "up_36_0003", "up_36_0005",
                           "up_48_0002", "up_48_0003",
                           "fav_12_0003", "fav_12_0005", "fav_36_0003", "fav_36_0005"}

        for tgt_name, tgt_col in targets_for_feat:
            if tgt_col not in df.columns:
                log(f"  {tgt_name}: {tgt_col} NOT FOUND in data -- skipping", f)
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
            tag = " [NEW]" if tgt_name not in existing_targets else ""
            log(f"  {tgt_name}: top5 = {list(top5['feat'].values)}{tag}", f)

        def get_feats(name, n_feats=100):
            # Match target name to cached features
            for key in feat_cache:
                if key in name:
                    return feat_cache[key][:n_feats]
            # Fallback: try loose matching
            for key in feat_cache:
                if key.replace("_", "") in name.replace("_", ""):
                    return feat_cache[key][:n_feats]
            return all_feat_cols[:n_feats]

        # Phase 1: Train all models
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
                model_results[s] = {"trades": trades, "auc": auc, "horizon": horizon}

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
            new_tag = " [NEW]" if name in new_candidate_names else ""
            passed = pos >= 7 and len(trade_nets) >= 7
            tag = " *" if passed else ""
            log(f"  {name:<25s} AUC={np.mean(aucs):.4f}+/-{np.std(aucs):.3f}  "
                f"{pos}/{len(trade_nets)} pos  avg={avg_net:>+.2%}  "
                f"sharpe={avg_sharpe:>+.1f}  med_trades={med_trades}  "
                f"t={elapsed:.0f}s{ens_tag}{forced_tag}{new_tag}{tag}", f)

        # Selection
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

        existing_selected = [n for n in selected if n not in new_candidate_names]
        new_selected = [n for n in selected if n in new_candidate_names]
        log(f"\n  Selected: {len(selected)} models ({len(existing_selected)} existing + {len(new_selected)} new)", f)
        if new_selected:
            log(f"  New models that passed: {new_selected}", f)

        # Phase 2: Quality scoring
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
            new_tag = " [NEW]" if name in new_candidate_names else ""
            log(f"    {name:<25s}  quality={quality[name]:.3f}  weight={weights[name]:.3f}{new_tag}", f)

        def get_split_trades(split_idx, model_names):
            trades_list = []
            for name in model_names:
                if split_idx in all_results.get(name, {}):
                    r = all_results[name][split_idx]
                    trades_list.append((name, r["trades"], r["horizon"]))
            return trades_list

        # Phase 3: Compare configs — existing vs expanded
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 3: PORTFOLIO COMPARISON (existing vs expanded)", f)
        log(f"{'#'*80}\n", f)

        configs_to_test = [
            (0.002, 10, 14, 1.0, "dd2m_cl10_c14_s100"),   # Best risk-adj
            (0.002, 10, 20, 0.8, "dd2m_cl10_c20_s80"),     # Best Sharpe
            (0.005, 10, 14, 1.0, "dd5m_cl10_c14_s100"),    # Best returns
            (0.003, 10, 14, 1.0, "dd3m_cl10_c14_s100"),    # Mid-range
            (0.002, 10, 16, 1.0, "dd2m_cl10_c16_s100"),    # Balanced
        ]

        log(f"  {'Config':<25s} {'Portfolio':<12s} {'Pos/N':>6s} {'Avg%':>9s} {'WorstDD%':>10s} "
            f"{'Sharpe':>7s} {'Models':>7s}", f)

        for max_dd, cooldown, max_conc, pos_scale, cfg_label in configs_to_test:
            for portfolio_name, model_set in [("EXISTING", existing_selected), ("EXPANDED", selected)]:
                split_res = []
                for s in range(N_SPLITS):
                    trades = get_split_trades(s, model_set)
                    res = portfolio_backtest(
                        trades, max_dd_pct=max_dd, cooldown=cooldown,
                        max_concurrent=max_conc, position_scale=pos_scale,
                        model_weights=weights)
                    split_res.append(res)

                pos = sum(1 for r in split_res if r["net"] > 0)
                avg_net = np.mean([r["net"] for r in split_res])
                valid = [r for r in split_res if r["n"] > 0]
                worst_dd = min(r["max_dd"] for r in valid) if valid else 0
                avg_sharpe = np.mean([r["sharpe"] for r in valid]) if valid else 0

                log(f"  {cfg_label:<25s} {portfolio_name:<12s} {pos:>3d}/10 {avg_net:>+8.1%} "
                    f"{worst_dd:>+9.1%} {avg_sharpe:>+6.1f} {len(model_set):>5d}", f)

        # Phase 4: Final config — pick the best
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 4: FINAL PRODUCTION CONFIG", f)
        log(f"{'#'*80}\n", f)

        best_sharpe = -1
        best_cfg = None
        best_params = None
        all_cfg_results = {}

        for max_dd, cooldown, max_conc, pos_scale, cfg_label in configs_to_test:
            split_res = []
            for s in range(N_SPLITS):
                trades = get_split_trades(s, selected)
                res = portfolio_backtest(
                    trades, max_dd_pct=max_dd, cooldown=cooldown,
                    max_concurrent=max_conc, position_scale=pos_scale,
                    model_weights=weights)
                split_res.append(res)

            pos = sum(1 for r in split_res if r["net"] > 0)
            avg_net = np.mean([r["net"] for r in split_res])
            valid = [r for r in split_res if r["n"] > 0]
            worst_dd = min(r["max_dd"] for r in valid) if valid else 0
            avg_dd = np.mean([r["max_dd"] for r in valid]) if valid else 0
            avg_sharpe = np.mean([r["sharpe"] for r in valid]) if valid else 0
            avg_n = np.mean([r["n"] for r in split_res])

            all_cfg_results[cfg_label] = {
                "pos": pos, "avg_net": avg_net, "avg_dd": avg_dd,
                "worst_dd": worst_dd, "avg_n": avg_n, "sharpe": avg_sharpe,
                "params": (max_dd, cooldown, max_conc, pos_scale),
                "split_res": split_res,
            }

            if pos >= 10 and worst_dd > -0.22 and avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_cfg = cfg_label
                best_params = (max_dd, cooldown, max_conc, pos_scale)

        if best_cfg is None:
            for cfg_label, data in all_cfg_results.items():
                if data["pos"] >= 10 and data["sharpe"] > best_sharpe:
                    best_sharpe = data["sharpe"]
                    best_cfg = cfg_label
                    best_params = data["params"]

        cfg_data = all_cfg_results[best_cfg]
        max_dd, cooldown, max_conc, pos_scale = best_params

        log(f"  +--------------------------------------------------+", f)
        log(f"  |  FINAL CONFIG: {best_cfg:<33s}|", f)
        log(f"  +--------------------------------------------------+", f)
        log(f"  DD limit: {max_dd}", f)
        log(f"  Cooldown: {cooldown}", f)
        log(f"  Max concurrent: {max_conc}", f)
        log(f"  Position scale: {pos_scale}", f)
        log(f"  Models: {len(selected)} ({len(existing_selected)} existing + {len(new_selected)} new)", f)
        log(f"  Quality weighting: YES (recent-weighted)", f)

        total_compound = 1.0
        all_sharpes = []
        for s in range(N_SPLITS):
            trades = get_split_trades(s, selected)
            res = portfolio_backtest(trades, max_dd_pct=max_dd, cooldown=cooldown,
                                   max_concurrent=max_conc, position_scale=pos_scale,
                                   model_weights=weights)
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

        log(f"\n  === COMPARISON vs V20 ===", f)
        log(f"  V20: 10/10 pos, avg +5770.2%, worstDD -20.2%, Sharpe 39.9, annual +11454%, minSharpe 32.5", f)
        log(f"  V21: {cfg_data['pos']}/10 pos, avg {cfg_data['avg_net']:+.1%}, "
            f"worstDD {cfg_data['worst_dd']:+.1%}, Sharpe {cfg_data['sharpe']:.1f}, "
            f"annual {annual:+.1%}, minSharpe {min(all_sharpes):.1f}" if all_sharpes else "", f)

        log(f"\n  === NEW CANDIDATES SUMMARY ===", f)
        for name in sorted(new_candidate_names):
            if name in all_results:
                results = all_results[name]
                trade_nets = []
                for r in results.values():
                    if r["trades"]:
                        trade_nets.append(np.prod([1 + t["net_ret"] for t in r["trades"]]) - 1)
                pos = sum(1 for n_ in trade_nets if n_ > 0)
                total = len(trade_nets)
                status = "SELECTED" if name in selected else "DROPPED"
                log(f"  {name:<25s}  {pos}/{total} pos  {status}", f)
            else:
                log(f"  {name:<25s}  TARGET NOT FOUND", f)

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
            "config_name": best_cfg, "dd_limit": max_dd,
            "cooldown": cooldown, "max_concurrent": max_conc,
            "position_scale": pos_scale, "fee_rt": FEE_MAKER_RT,
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
                    "is_new": name in new_candidate_names,
                }
                if info.get("model_cb") is not None:
                    cb_path = os.path.join(MODEL_DIR, f"prod_{name}_cb.pkl")
                    with open(cb_path, "wb") as mf:
                        pickle.dump(info["model_cb"], mf)
                    entry["cb_file"] = f"prod_{name}_cb.pkl"

                prod_config["models"].append(entry)
                ens_label = f" [ENS w={info['lgb_weight']:.1f}]" if info["use_ensemble"] else ""
                w_label = f" q={weights.get(name, 1.0):.2f}"
                new_label = " [NEW]" if name in new_candidate_names else ""
                log(f"  Saved: {name} (AUC={info['auc']:.4f}){ens_label}{w_label}{new_label}", f)

        config_path = os.path.join(MODEL_DIR, "production_config_v21.json")
        with open(config_path, "w") as cf:
            json.dump(prod_config, cf, indent=2, default=str)
        log(f"  Config: {config_path}", f)

        log(f"\n{'='*80}", f)
        log(f"Results saved to {RESULTS_FILE}", f)


if __name__ == "__main__":
    main()
