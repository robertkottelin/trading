"""
BTC ML Training — Iteration 7: SIGNAL STRENGTH BOOST
=====================================================
Focus: Increase AUC from ~0.55 to improve edge per trade and trade count.

Key innovations over Iter 6:
  1. Threshold-based targets: predict "up >0.3% in 1h" instead of "up in 1h"
  2. XGBoost comparison: different algorithm may find different patterns
  3. Recency-weighted training: exponential decay to combat observed signal decay
  4. Stacked meta-model: combine direction + threshold + trend predictions
  5. Asymmetric target: predict max_runup > fee_threshold (directly targets profitability)

Usage:
  python train_model_v7.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import os
import time
import warnings
from datetime import datetime, timezone
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = "btc_data"
FEATURES_FILE = os.path.join(OUTPUT_DIR, "btc_features_5m.parquet")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "training_results_v7.txt")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

FEE_MAKER_RT = 0.0004  # 0.04% round-trip (maker)

# ─── Logging ────────────────────────────────────────────────────────────────

def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")
        f.flush()


# ─── AUC (no sklearn dependency) ───────────────────────────────────────────

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
    tp = fp = auc = prev_fpr = prev_tpr = 0.0
    for val in y_sorted:
        if val == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        tpr = tp / n_pos
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_fpr = fpr
        prev_tpr = tpr
    return auc


# ─── Feature selection ─────────────────────────────────────────────────────

def get_feature_cols(df):
    exclude = {"open_time_ms", "timestamp"}
    target_cols = {c for c in df.columns if c.startswith("target_")}
    return sorted(c for c in df.columns if c not in exclude and c not in target_cols)


# ─── Backtesting ───────────────────────────────────────────────────────────

def backtest_lo(close, y_prob, horizon, fee_rt, margin=0.04, top_pct=None):
    """Long-only backtest with non-overlapping trades."""
    n = len(close)
    trades = []
    i = 0
    while i + horizon < n:
        prob = y_prob[i]
        if prob > 0.5 + margin:
            raw_ret = (close[i + horizon] - close[i]) / close[i]
            trades.append({"idx": i, "prob": prob, "raw_ret": raw_ret,
                           "net_ret": raw_ret - fee_rt})
        i += horizon
    if not trades:
        return trades
    if top_pct is not None and len(trades) > 5:
        df_t = pd.DataFrame(trades)
        abs_conf = abs(df_t["prob"] - 0.5)
        threshold = abs_conf.quantile(1.0 - top_pct)
        df_t = df_t[abs_conf >= threshold]
        trades = df_t.to_dict("records")
    return trades


def bt_stats(trades, horizon):
    if not trades:
        return {"n": 0, "net": 0, "sharpe": 0, "wr": 0, "pf": 0,
                "max_dd": 0, "kelly": 0}
    rets = np.array([t["net_ret"] for t in trades])
    cum = np.cumprod(1 + rets)
    total = cum[-1] - 1
    mu = rets.mean()
    sigma = rets.std()
    cpy = 365.25 * 24 * 12 / horizon
    sharpe = (mu / sigma) * np.sqrt(cpy) if sigma > 0 else 0
    wr = (rets > 0).mean()
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = wins / losses if losses > 0 else float("inf")
    cummax = np.maximum.accumulate(cum)
    max_dd = ((cum - cummax) / cummax).min()
    avg_w = rets[rets > 0].mean() if (rets > 0).any() else 0
    avg_l = abs(rets[rets < 0].mean()) if (rets < 0).any() else 1
    b = avg_w / avg_l if avg_l > 0 else 0
    kelly = (wr * b - (1 - wr)) / b if b > 0 else 0
    return {"n": len(rets), "net": total, "sharpe": sharpe, "wr": wr, "pf": pf,
            "max_dd": max_dd, "kelly": kelly}


# ─── Model training ───────────────────────────────────────────────────────

LGB_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "boosting_type": "gbdt", "n_estimators": 5000,
    "learning_rate": 0.005, "num_leaves": 24, "max_depth": 5,
    "min_child_samples": 500, "subsample": 0.5, "subsample_freq": 1,
    "colsample_bytree": 0.3, "colsample_bynode": 0.5,
    "reg_alpha": 5.0, "reg_lambda": 20.0,
    "feature_fraction_bynode": 0.5, "path_smooth": 10.0,
    "random_state": 42, "verbose": -1, "n_jobs": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "n_estimators": 5000, "learning_rate": 0.005,
    "max_depth": 5, "min_child_weight": 500,
    "subsample": 0.5, "colsample_bytree": 0.3, "colsample_bynode": 0.5,
    "reg_alpha": 5.0, "reg_lambda": 20.0,
    "tree_method": "hist", "random_state": 42, "verbosity": 0,
    "n_jobs": -1,
}


def train_lgb(X_tr, y_tr, X_va, y_va, params=None, sample_weight=None):
    p = params or LGB_PARAMS
    m = lgb.LGBMClassifier(**p)
    fit_kwargs = {"eval_set": [(X_va, y_va)],
                  "callbacks": [lgb.early_stopping(100, verbose=False),
                                lgb.log_evaluation(0)]}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    m.fit(X_tr, y_tr, **fit_kwargs)
    return m


def train_xgb(X_tr, y_tr, X_va, y_va, params=None, sample_weight=None):
    p = params or XGB_PARAMS
    m = xgb.XGBClassifier(**p)
    fit_kwargs = {"eval_set": [(X_va, y_va)], "verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    m.fit(X_tr, y_tr, **fit_kwargs)
    return m


def calibrate_iso(y_val, p_val, p_test):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p_val, y_val)
    return ir.transform(p_test)


def get_date_range(df_slice, col="open_time_ms"):
    dates = pd.to_datetime(df_slice[col], unit="ms", utc=True)
    return f"{dates.iloc[0].strftime('%Y-%m-%d')}/{dates.iloc[-1].strftime('%Y-%m-%d')}"


# ─── Walk-forward infrastructure ──────────────────────────────────────────

BT_CONFIGS = [
    ("m02_top30", 0.02, 0.30),
    ("m04_top30", 0.04, 0.30),
    ("m04_top20", 0.04, 0.20),
    ("m06_top20", 0.06, 0.20),
    ("m04_top10", 0.04, 0.10),
    ("m06_top10", 0.06, 0.10),
    ("m02_all",   0.02, None),
    ("m04_all",   0.04, None),
]


def compute_recency_weights(n_samples, half_life_frac=0.15):
    """Exponential decay sample weights. half_life_frac = fraction of dataset as half-life."""
    half_life = max(int(n_samples * half_life_frac), 1000)
    idx = np.arange(n_samples)
    weights = np.exp(np.log(2) * (idx - n_samples + 1) / half_life)
    # Normalize so mean weight = 1
    weights = weights / weights.mean()
    return weights


def run_wf_split(df, feat_cols, target_col, horizon, train_end, test_start, test_end,
                 algo="lgb", recency_weight=False, f=None, purge=None):
    """Train and evaluate one walk-forward split. Returns dict with predictions + stats."""
    if purge is None:
        purge = 2 * horizon

    # Split data
    train_mask = np.arange(len(df)) < train_end
    val_frac = 0.15
    n_train_total = train_mask.sum()
    val_size = int(n_train_total * val_frac)
    val_start_idx = train_end - val_size - purge
    train_sub_mask = np.arange(len(df)) < val_start_idx
    val_mask = (np.arange(len(df)) >= val_start_idx + purge) & (np.arange(len(df)) < train_end)
    test_mask = (np.arange(len(df)) >= test_start) & (np.arange(len(df)) < test_end)

    y_col = df[target_col].values
    X_all = df[feat_cols].values

    # Get valid rows
    valid_tr = train_sub_mask & ~np.isnan(y_col)
    valid_va = val_mask & ~np.isnan(y_col)
    valid_te = test_mask & ~np.isnan(y_col)

    X_tr, y_tr = X_all[valid_tr], y_col[valid_tr]
    X_va, y_va = X_all[valid_va], y_col[valid_va]
    X_te, y_te = X_all[valid_te], y_col[valid_te]

    if len(X_tr) < 1000 or len(X_va) < 100 or len(X_te) < 100:
        return None

    # Sample weights
    sw = None
    if recency_weight:
        sw = compute_recency_weights(len(X_tr))

    # Train
    t0 = time.time()
    if algo == "lgb":
        model = train_lgb(X_tr, y_tr, X_va, y_va, sample_weight=sw)
        best_iter = model.best_iteration_
    elif algo == "xgb":
        model = train_xgb(X_tr, y_tr, X_va, y_va, sample_weight=sw)
        best_iter = model.best_iteration
    else:
        raise ValueError(f"Unknown algo: {algo}")
    elapsed = time.time() - t0

    # Predict
    p_raw_te = model.predict_proba(X_te)[:, 1]
    p_raw_va = model.predict_proba(X_va)[:, 1]
    p_cal_te = calibrate_iso(y_va, p_raw_va, p_raw_te)

    auc_raw = auc_roc(y_te, p_raw_te)
    auc_cal = auc_roc(y_te, p_cal_te)

    # Get close prices for backtesting
    close_te = df["close"].values[valid_te]

    return {
        "model": model,
        "auc_raw": auc_raw,
        "auc_cal": auc_cal,
        "best_iter": best_iter,
        "elapsed": elapsed,
        "p_raw": p_raw_te,
        "p_cal": p_cal_te,
        "y_te": y_te,
        "close_te": close_te,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "test_indices": np.where(valid_te)[0],
    }


def evaluate_bt_configs(close_te, p_raw, p_cal, horizon, bt_configs, prefix=""):
    """Evaluate multiple backtest configs. Returns dict of {config_name: stats}."""
    results = {}
    for name, margin, top_pct in bt_configs:
        for prob_type, probs in [("raw", p_raw), ("cal", p_cal)]:
            key = f"{prefix}{prob_type}_{name}"
            trades = backtest_lo(close_te, probs, horizon, FEE_MAKER_RT,
                                 margin=margin, top_pct=top_pct)
            results[key] = bt_stats(trades, horizon)
    return results


def print_bt_results(results, f):
    for key, s in sorted(results.items()):
        if s["n"] == 0:
            continue
        log(f"    {key:<24s}: n={s['n']:>4d}  net={s['net']:>+7.2%}  "
            f"sr={s['sharpe']:>7.2f}  WR={s['wr']:>5.1%}  "
            f"PF={s['pf']:>5.2f}  kelly={s['kelly']:>+6.3f}  "
            f"maxDD={s['max_dd']:>7.2%}", f)


def summarize_wf(all_split_results, configs_to_summarize, n_splits, f, prefix=""):
    """Print WF summary for a set of configs across splits."""
    for cfg_name in configs_to_summarize:
        nets = []
        for split_res in all_split_results:
            if split_res is None:
                continue
            bt = split_res.get("bt", {})
            key = f"{prefix}{cfg_name}"
            if key in bt and bt[key]["n"] > 0:
                nets.append(bt[key]["net"])
            else:
                nets.append(None)

        valid_nets = [n for n in nets if n is not None]
        if not valid_nets:
            continue
        pos = sum(1 for n in valid_nets if n > 0)
        total = len(valid_nets)
        avg_net = np.mean(valid_nets)
        avg_kelly = np.mean([
            all_split_results[i]["bt"].get(f"{prefix}{cfg_name}", {}).get("kelly", 0)
            for i in range(len(all_split_results))
            if all_split_results[i] is not None
            and f"{prefix}{cfg_name}" in all_split_results[i].get("bt", {})
            and all_split_results[i]["bt"][f"{prefix}{cfg_name}"]["n"] > 0
        ])
        med_n = int(np.median([
            all_split_results[i]["bt"].get(f"{prefix}{cfg_name}", {}).get("n", 0)
            for i in range(len(all_split_results))
            if all_split_results[i] is not None
            and f"{prefix}{cfg_name}" in all_split_results[i].get("bt", {})
            and all_split_results[i]["bt"][f"{prefix}{cfg_name}"]["n"] > 0
        ] or [0]))
        log(f"  {cfg_name:<24s} {pos:>2d}/{total:<2d}  {avg_net:>+7.2%}  "
            f"kelly={avg_kelly:>+6.3f}  med_n={med_n:>4d}", f)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols")

    all_feat_cols = get_feature_cols(df)
    n_total = len(df)

    # Pre-select top features from v6 (stable features work just as well)
    top80_1h = None  # Will be computed in Phase 0

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        log(f"BTC ML Training — Iteration 7: SIGNAL STRENGTH BOOST", f)
        log(f"{'='*80}", f)
        log(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", f)
        log(f"Data: {len(df):,} rows x {len(df.columns)} cols  Features: {len(all_feat_cols)}", f)
        log(f"\nKey changes from Iter 6:", f)
        log(f"  1. Threshold-based targets (up >0.3% in 1h)", f)
        log(f"  2. XGBoost comparison", f)
        log(f"  3. Recency-weighted training (exponential decay)", f)
        log(f"  4. Stacked meta-model (direction + threshold + trend)", f)
        log(f"  5. Asymmetric target (max_runup > fee threshold)", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 0: FEATURE SELECTION (top-80 from quick train)
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 0: FEATURE SELECTION", f)
        log(f"{'#'*80}\n", f)

        # Quick train on direction_12 to get feature importances
        target_col = "target_direction_12"
        valid = ~df[target_col].isna()
        n_valid = valid.sum()
        tr_end = int(n_valid * 0.70)
        va_end = int(n_valid * 0.85)
        valid_idx = np.where(valid)[0]

        X_tr = df.loc[valid_idx[:tr_end], all_feat_cols]
        y_tr = df.loc[valid_idx[:tr_end], target_col]
        X_va = df.loc[valid_idx[tr_end:va_end], all_feat_cols]
        y_va = df.loc[valid_idx[tr_end:va_end], target_col]

        quick_model = train_lgb(X_tr.values, y_tr.values, X_va.values, y_va.values)
        imp = pd.DataFrame({"feat": all_feat_cols, "imp": quick_model.feature_importances_})
        imp = imp.sort_values("imp", ascending=False)

        top80_1h = imp.head(80)["feat"].tolist()
        log(f"  Top-80 features selected for 1h:", f)
        for i, row in imp.head(15).iterrows():
            log(f"    {imp.index.get_loc(i)+1:>3d}.  {row['imp']:>5.0f}  {row['feat']}", f)

        # Also get top features for threshold target
        threshold_target = "target_up_12_0003"
        if threshold_target in df.columns:
            valid_t = ~df[threshold_target].isna()
            valid_idx_t = np.where(valid_t)[0]
            n_valid_t = valid_t.sum()
            tr_end_t = int(n_valid_t * 0.70)
            va_end_t = int(n_valid_t * 0.85)
            X_tr_t = df.loc[valid_idx_t[:tr_end_t], all_feat_cols]
            y_tr_t = df.loc[valid_idx_t[:tr_end_t], threshold_target]
            X_va_t = df.loc[valid_idx_t[tr_end_t:va_end_t], all_feat_cols]
            y_va_t = df.loc[valid_idx_t[tr_end_t:va_end_t], threshold_target]
            m_t = train_lgb(X_tr_t.values, y_tr_t.values, X_va_t.values, y_va_t.values)
            imp_t = pd.DataFrame({"feat": all_feat_cols, "imp": m_t.feature_importances_})
            imp_t = imp_t.sort_values("imp", ascending=False)
            top80_thresh = imp_t.head(80)["feat"].tolist()
            log(f"\n  Top-80 features for threshold target (up >0.3% 1h):", f)
            for i, row in imp_t.head(15).iterrows():
                log(f"    {imp_t.index.get_loc(i)+1:>3d}.  {row['imp']:>5.0f}  {row['feat']}", f)
        else:
            top80_thresh = top80_1h

        # Union of direction + threshold features (ensures both signals are covered)
        top_union = sorted(set(top80_1h) | set(top80_thresh))
        log(f"\n  Union feature set: {len(top_union)} features", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: BASELINE vs THRESHOLD TARGETS (10-split WF)
        # Compare direction_12 vs up_12_0003 vs up_12_0005
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 1: DIRECTION vs THRESHOLD TARGETS (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        targets_to_compare = [
            ("direction_12", "target_direction_12", 12),
            ("up_12_0002",   "target_up_12_0002",   12),
            ("up_12_0003",   "target_up_12_0003",   12),
            ("up_12_0005",   "target_up_12_0005",   12),
            ("direction_36", "target_direction_36",  36),
            ("up_36_0005",   "target_up_36_0005",   36),
            ("up_36_001",    "target_up_36_001",     36),
        ]

        n_splits = 10
        test_frac = 0.05
        test_size = int(n_total * test_frac)

        for tgt_name, tgt_col, horizon in targets_to_compare:
            if tgt_col not in df.columns:
                log(f"  SKIP {tgt_name}: column {tgt_col} not found\n", f)
                continue

            log(f"  === {tgt_name} (horizon={horizon}) ===", f)

            # Use direction features for direction, threshold features for threshold
            feat_cols = top80_1h if "direction" in tgt_name else top80_thresh

            split_results = []
            purge = 2 * horizon

            for s in range(n_splits):
                test_end = n_total - s * test_size
                test_start = test_end - test_size
                train_end = test_start - purge

                if train_end < n_total * 0.3:
                    split_results.append(None)
                    continue

                result = run_wf_split(df, feat_cols, tgt_col, horizon,
                                      train_end, test_start, test_end, algo="lgb", f=f)
                if result is None:
                    split_results.append(None)
                    continue

                # Run backtests (always using direction target's close prices and horizon)
                bt = evaluate_bt_configs(result["close_te"], result["p_raw"],
                                         result["p_cal"], horizon, BT_CONFIGS, prefix="")
                result["bt"] = bt

                date_range = get_date_range(df.iloc[test_start:test_end])
                log(f"    Split {s+1}: {date_range}  train={result['n_train']:,}  "
                    f"AUC={result['auc_raw']:.4f}  iter={result['best_iter']}  "
                    f"time={result['elapsed']:.0f}s", f)

                # Print key configs only
                for key in ["raw_m06_top10", "raw_m04_top10", "raw_m04_top20"]:
                    if key in bt and bt[key]["n"] > 0:
                        s_bt = bt[key]
                        log(f"      {key}: n={s_bt['n']:>4d}  net={s_bt['net']:>+7.2%}  "
                            f"WR={s_bt['wr']:>5.1%}  kelly={s_bt['kelly']:>+6.3f}", f)

                split_results.append(result)

            # Summary
            log(f"\n  {tgt_name} WF SUMMARY:", f)
            aucs = [r["auc_raw"] for r in split_results if r is not None]
            if aucs:
                log(f"    AUC: {np.mean(aucs):.4f}±{np.std(aucs):.4f}", f)

            for cfg_name in ["raw_m06_top10", "raw_m04_top10", "raw_m04_top20",
                             "raw_m06_top20", "cal_m04_top10"]:
                nets = []
                for r in split_results:
                    if r is None:
                        continue
                    if cfg_name in r.get("bt", {}) and r["bt"][cfg_name]["n"] > 0:
                        nets.append(r["bt"][cfg_name]["net"])
                if nets:
                    pos = sum(1 for n in nets if n > 0)
                    log(f"    {cfg_name}: {pos}/{len(nets)} positive, "
                        f"avg={np.mean(nets):>+7.2%}", f)
            log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: XGBOOST vs LIGHTGBM (10-split WF on direction_12)
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 2: XGBOOST vs LIGHTGBM COMPARISON (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        target_col_dir = "target_direction_12"
        horizon = 12
        purge = 2 * horizon

        lgb_results = []
        xgb_results = []

        for s in range(n_splits):
            test_end = n_total - s * test_size
            test_start = test_end - test_size
            train_end = test_start - purge

            if train_end < n_total * 0.3:
                lgb_results.append(None)
                xgb_results.append(None)
                continue

            date_range = get_date_range(df.iloc[test_start:test_end])

            # LightGBM
            r_lgb = run_wf_split(df, top80_1h, target_col_dir, horizon,
                                 train_end, test_start, test_end, algo="lgb", f=f)
            if r_lgb:
                r_lgb["bt"] = evaluate_bt_configs(r_lgb["close_te"], r_lgb["p_raw"],
                                                  r_lgb["p_cal"], horizon, BT_CONFIGS)

            # XGBoost
            r_xgb = run_wf_split(df, top80_1h, target_col_dir, horizon,
                                 train_end, test_start, test_end, algo="xgb", f=f)
            if r_xgb:
                r_xgb["bt"] = evaluate_bt_configs(r_xgb["close_te"], r_xgb["p_raw"],
                                                  r_xgb["p_cal"], horizon, BT_CONFIGS)

            log(f"  Split {s+1}: {date_range}", f)
            if r_lgb:
                log(f"    LGB: AUC={r_lgb['auc_raw']:.4f}  iter={r_lgb['best_iter']}  "
                    f"time={r_lgb['elapsed']:.0f}s", f)
            if r_xgb:
                log(f"    XGB: AUC={r_xgb['auc_raw']:.4f}  iter={r_xgb['best_iter']}  "
                    f"time={r_xgb['elapsed']:.0f}s", f)

            # Print key config comparison
            for key in ["raw_m06_top10", "raw_m04_top10"]:
                lgb_s = r_lgb["bt"].get(key, {}) if r_lgb else {}
                xgb_s = r_xgb["bt"].get(key, {}) if r_xgb else {}
                if lgb_s.get("n", 0) > 0 or xgb_s.get("n", 0) > 0:
                    lgb_net = lgb_s.get("net", 0)
                    xgb_net = xgb_s.get("net", 0)
                    log(f"    {key}: LGB={lgb_net:>+7.2%}  XGB={xgb_net:>+7.2%}", f)

            lgb_results.append(r_lgb)
            xgb_results.append(r_xgb)

        # Summary comparison
        log(f"\n  ──── LGB vs XGB SUMMARY ────", f)
        lgb_aucs = [r["auc_raw"] for r in lgb_results if r is not None]
        xgb_aucs = [r["auc_raw"] for r in xgb_results if r is not None]
        log(f"  LGB AUC: {np.mean(lgb_aucs):.4f}±{np.std(lgb_aucs):.4f}", f)
        log(f"  XGB AUC: {np.mean(xgb_aucs):.4f}±{np.std(xgb_aucs):.4f}", f)

        for cfg in ["raw_m06_top10", "raw_m04_top10", "raw_m04_top20"]:
            lgb_nets = [r["bt"][cfg]["net"] for r in lgb_results
                        if r and cfg in r.get("bt", {}) and r["bt"][cfg]["n"] > 0]
            xgb_nets = [r["bt"][cfg]["net"] for r in xgb_results
                        if r and cfg in r.get("bt", {}) and r["bt"][cfg]["n"] > 0]
            lgb_pos = sum(1 for n in lgb_nets if n > 0) if lgb_nets else 0
            xgb_pos = sum(1 for n in xgb_nets if n > 0) if xgb_nets else 0
            log(f"  {cfg}:", f)
            log(f"    LGB: {lgb_pos}/{len(lgb_nets)} pos, avg={np.mean(lgb_nets):>+7.2%}" if lgb_nets else f"    LGB: N/A", f)
            log(f"    XGB: {xgb_pos}/{len(xgb_nets)} pos, avg={np.mean(xgb_nets):>+7.2%}" if xgb_nets else f"    XGB: N/A", f)

        # Also check if averaging LGB+XGB predictions helps
        log(f"\n  ──── LGB+XGB AVERAGE ────", f)
        for s in range(n_splits):
            if lgb_results[s] is None or xgb_results[s] is None:
                continue
            # Average raw predictions
            p_avg = (lgb_results[s]["p_raw"] + xgb_results[s]["p_raw"]) / 2
            bt_avg = {}
            for name, margin, top_pct in BT_CONFIGS:
                key = f"avg_{name}"
                trades = backtest_lo(lgb_results[s]["close_te"], p_avg, horizon,
                                     FEE_MAKER_RT, margin=margin, top_pct=top_pct)
                bt_avg[key] = bt_stats(trades, horizon)
            lgb_results[s]["bt_avg"] = bt_avg

        avg_nets_m06t10 = []
        for r in lgb_results:
            if r and "bt_avg" in r:
                s_avg = r["bt_avg"].get("avg_m06_top10", {})
                if s_avg.get("n", 0) > 0:
                    avg_nets_m06t10.append(s_avg["net"])
        if avg_nets_m06t10:
            pos = sum(1 for n in avg_nets_m06t10 if n > 0)
            log(f"  avg_m06_top10: {pos}/{len(avg_nets_m06t10)} pos, "
                f"avg={np.mean(avg_nets_m06t10):>+7.2%}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: RECENCY-WEIGHTED TRAINING (10-split WF)
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 3: RECENCY-WEIGHTED TRAINING (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        target_col_dir = "target_direction_12"
        horizon = 12
        purge = 2 * horizon

        # Test multiple half-lives
        half_lives = [0.05, 0.10, 0.15, 0.25]
        hl_results = {hl: [] for hl in half_lives}

        for s in range(n_splits):
            test_end = n_total - s * test_size
            test_start = test_end - test_size
            train_end = test_start - purge

            if train_end < n_total * 0.3:
                for hl in half_lives:
                    hl_results[hl].append(None)
                continue

            date_range = get_date_range(df.iloc[test_start:test_end])
            log(f"  Split {s+1}: {date_range}", f)

            for hl in half_lives:
                # Custom training with recency weights
                val_frac = 0.15
                n_train_total = train_end
                val_size = int(n_train_total * val_frac)
                val_start_idx = train_end - val_size - purge

                train_sub_mask = np.arange(len(df)) < val_start_idx
                val_mask = (np.arange(len(df)) >= val_start_idx + purge) & (np.arange(len(df)) < train_end)
                test_mask = (np.arange(len(df)) >= test_start) & (np.arange(len(df)) < test_end)

                y_col_vals = df[target_col_dir].values
                X_all_vals = df[top80_1h].values

                valid_tr = train_sub_mask & ~np.isnan(y_col_vals)
                valid_va = val_mask & ~np.isnan(y_col_vals)
                valid_te = test_mask & ~np.isnan(y_col_vals)

                X_tr_w = X_all_vals[valid_tr]
                y_tr_w = y_col_vals[valid_tr]
                X_va_w = X_all_vals[valid_va]
                y_va_w = y_col_vals[valid_va]
                X_te_w = X_all_vals[valid_te]
                y_te_w = y_col_vals[valid_te]

                if len(X_tr_w) < 1000:
                    hl_results[hl].append(None)
                    continue

                sw = compute_recency_weights(len(X_tr_w), half_life_frac=hl)
                model = train_lgb(X_tr_w, y_tr_w, X_va_w, y_va_w, sample_weight=sw)
                p_raw = model.predict_proba(X_te_w)[:, 1]
                auc_val = auc_roc(y_te_w, p_raw)

                close_te_w = df["close"].values[valid_te]
                bt = {}
                for name, margin, top_pct in [("m06_top10", 0.06, 0.10),
                                               ("m04_top10", 0.04, 0.10),
                                               ("m04_top20", 0.04, 0.20)]:
                    trades = backtest_lo(close_te_w, p_raw, horizon, FEE_MAKER_RT,
                                         margin=margin, top_pct=top_pct)
                    bt[name] = bt_stats(trades, horizon)

                hl_results[hl].append({"auc": auc_val, "bt": bt})

                log(f"    hl={hl:.2f}: AUC={auc_val:.4f}  "
                    f"m06t10: n={bt['m06_top10']['n']:>3d} net={bt['m06_top10']['net']:>+7.2%}  "
                    f"m04t10: n={bt['m04_top10']['n']:>3d} net={bt['m04_top10']['net']:>+7.2%}", f)

        # Summary
        log(f"\n  ──── RECENCY WEIGHT SUMMARY ────", f)
        log(f"  {'Half-life':<12s} {'AUC':<16s} {'m06_top10':<20s} {'m04_top10':<20s}", f)

        # Also include unweighted baseline
        unw_aucs = [r["auc_raw"] for r in lgb_results if r is not None]
        unw_m06 = [r["bt"]["raw_m06_top10"]["net"] for r in lgb_results
                   if r and "raw_m06_top10" in r.get("bt", {}) and r["bt"]["raw_m06_top10"]["n"] > 0]
        unw_m06_pos = sum(1 for n in unw_m06 if n > 0)
        log(f"  {'none':<12s} {np.mean(unw_aucs):.4f}±{np.std(unw_aucs):.3f}  "
            f"{unw_m06_pos}/{len(unw_m06)} pos avg={np.mean(unw_m06):>+6.2%}", f)

        for hl in half_lives:
            aucs = [r["auc"] for r in hl_results[hl] if r is not None]
            m06_nets = [r["bt"]["m06_top10"]["net"] for r in hl_results[hl]
                        if r is not None and r["bt"]["m06_top10"]["n"] > 0]
            m04_nets = [r["bt"]["m04_top10"]["net"] for r in hl_results[hl]
                        if r is not None and r["bt"]["m04_top10"]["n"] > 0]
            m06_pos = sum(1 for n in m06_nets if n > 0)
            m04_pos = sum(1 for n in m04_nets if n > 0)
            log(f"  hl={hl:<8.2f}  {np.mean(aucs):.4f}±{np.std(aucs):.3f}  "
                f"{m06_pos}/{len(m06_nets)} pos avg={np.mean(m06_nets):>+6.2%}  "
                f"{m04_pos}/{len(m04_nets)} pos avg={np.mean(m04_nets):>+6.2%}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: STACKED META-MODEL (8-split WF)
        # Uses predictions from direction + threshold + trend models as features
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 4: STACKED META-MODEL (8-split WF)", f)
        log(f"{'#'*80}\n", f)

        # Base models: direction_12, up_12_0003, up_12_0005, trend_up_12
        base_targets = [
            ("dir12",    "target_direction_12"),
            ("up12_03",  "target_up_12_0003"),
            ("up12_05",  "target_up_12_0005"),
            ("trend12",  "target_trend_up_12"),
        ]
        # Optional: add longer horizon signals
        extra_targets = [
            ("dir36",    "target_direction_36"),
            ("up36_05",  "target_up_36_0005"),
        ]
        all_base_targets = base_targets + [t for t in extra_targets if t[1] in df.columns]

        horizon = 12
        n_meta_splits = 8
        meta_test_size = int(n_total * 0.06)
        purge = 2 * horizon

        meta_results = []

        for s in range(n_meta_splits):
            test_end = n_total - s * meta_test_size
            test_start = test_end - meta_test_size
            train_end_base = test_start - purge

            if train_end_base < n_total * 0.3:
                meta_results.append(None)
                continue

            date_range = get_date_range(df.iloc[test_start:test_end])
            log(f"  === Meta Split {s+1}: {date_range} ===", f)

            # Step 1: Train base models and get OOF predictions for meta-train
            # Use last 15% of train as meta-validation
            meta_val_size = int(train_end_base * 0.15)
            meta_train_end = train_end_base - meta_val_size - purge

            # Build meta-features: base model predictions on meta-val + test
            meta_val_mask = (np.arange(len(df)) >= meta_train_end + purge) & \
                            (np.arange(len(df)) < train_end_base)
            meta_test_mask = (np.arange(len(df)) >= test_start) & \
                             (np.arange(len(df)) < test_end)

            meta_features_val = {}
            meta_features_test = {}
            base_aucs = {}

            for tgt_name, tgt_col in all_base_targets:
                if tgt_col not in df.columns:
                    continue

                y_col_vals = df[tgt_col].values
                X_all_vals = df[top80_1h].values

                # Train on data up to meta_train_end
                tr_mask = np.arange(len(df)) < meta_train_end
                # Validation for early stopping: last portion of train
                inner_val_size = int(meta_train_end * 0.15)
                inner_tr_mask = np.arange(len(df)) < (meta_train_end - inner_val_size - purge)
                inner_va_mask = (np.arange(len(df)) >= meta_train_end - inner_val_size) & \
                                (np.arange(len(df)) < meta_train_end)

                valid_inner_tr = inner_tr_mask & ~np.isnan(y_col_vals)
                valid_inner_va = inner_va_mask & ~np.isnan(y_col_vals)
                valid_meta_val = meta_val_mask & ~np.isnan(y_col_vals)
                valid_meta_test = meta_test_mask & ~np.isnan(y_col_vals)

                X_itr = X_all_vals[valid_inner_tr]
                y_itr = y_col_vals[valid_inner_tr]
                X_iva = X_all_vals[valid_inner_va]
                y_iva = y_col_vals[valid_inner_va]

                if len(X_itr) < 1000 or len(X_iva) < 100:
                    continue

                m = train_lgb(X_itr, y_itr, X_iva, y_iva)

                # Predictions on meta-val and test
                if valid_meta_val.sum() > 0:
                    meta_features_val[tgt_name] = m.predict_proba(X_all_vals[valid_meta_val])[:, 1]
                if valid_meta_test.sum() > 0:
                    p_test = m.predict_proba(X_all_vals[valid_meta_test])[:, 1]
                    meta_features_test[tgt_name] = p_test

                    # Also get base model AUC on test for direction
                    if tgt_name == "dir12":
                        base_aucs["lgb_dir12"] = auc_roc(
                            df[tgt_col].values[valid_meta_test], p_test)

            if not meta_features_val or not meta_features_test:
                log(f"    SKIP: insufficient data for meta features", f)
                meta_results.append(None)
                continue

            # Step 2: Build meta-feature matrices
            # Need to align indices — use common valid rows
            val_idx = np.where(meta_val_mask)[0]
            test_idx = np.where(meta_test_mask)[0]

            # Filter to rows where direction_12 target is valid (since that's what we trade)
            dir_valid_val = ~np.isnan(df["target_direction_12"].values[val_idx])
            dir_valid_test = ~np.isnan(df["target_direction_12"].values[test_idx])

            # Meta features = base model predictions + a few raw features
            raw_meta_feats = ["hour_of_day", "vol_price_corr_96", "volume_momentum_96",
                              "vol_regime_rank", "realized_vol_24"]
            raw_meta_feats = [c for c in raw_meta_feats if c in df.columns]

            # Build aligned meta-feature matrix
            n_val_aligned = dir_valid_val.sum()
            n_test_aligned = dir_valid_test.sum()

            meta_X_val = np.column_stack([
                *[meta_features_val[k][dir_valid_val[:len(meta_features_val[k])]]
                  for k in meta_features_val if len(meta_features_val[k]) >= dir_valid_val.sum()],
                *[df[c].values[val_idx[dir_valid_val]] for c in raw_meta_feats]
            ]) if n_val_aligned > 0 else None

            meta_X_test = np.column_stack([
                *[meta_features_test[k][dir_valid_test[:len(meta_features_test[k])]]
                  for k in meta_features_test if len(meta_features_test[k]) >= dir_valid_test.sum()],
                *[df[c].values[test_idx[dir_valid_test]] for c in raw_meta_feats]
            ]) if n_test_aligned > 0 else None

            if meta_X_val is None or meta_X_test is None or n_val_aligned < 100 or n_test_aligned < 100:
                log(f"    SKIP: meta feature alignment failed", f)
                meta_results.append(None)
                continue

            meta_y_val = df["target_direction_12"].values[val_idx[dir_valid_val]]
            meta_y_test = df["target_direction_12"].values[test_idx[dir_valid_test]]

            # Split meta-val into train/val for the meta-model
            meta_tr_size = int(len(meta_y_val) * 0.7)
            meta_X_mtr = meta_X_val[:meta_tr_size]
            meta_y_mtr = meta_y_val[:meta_tr_size]
            meta_X_mva = meta_X_val[meta_tr_size:]
            meta_y_mva = meta_y_val[meta_tr_size:]

            # Train meta-model (simple LightGBM with very few leaves)
            meta_params = LGB_PARAMS.copy()
            meta_params["num_leaves"] = 8
            meta_params["max_depth"] = 3
            meta_params["min_child_samples"] = 200

            try:
                meta_model = train_lgb(meta_X_mtr, meta_y_mtr, meta_X_mva, meta_y_mva,
                                       params=meta_params)
                meta_p_test = meta_model.predict_proba(meta_X_test)[:, 1]
                meta_auc = auc_roc(meta_y_test, meta_p_test)
            except Exception as e:
                log(f"    Meta model failed: {e}", f)
                meta_results.append(None)
                continue

            base_auc = base_aucs.get("lgb_dir12", 0.5)
            log(f"    Base dir12 AUC: {base_auc:.4f}  Meta AUC: {meta_auc:.4f}  "
                f"delta: {meta_auc - base_auc:>+.4f}", f)

            # Backtest the meta-model
            close_meta_test = df["close"].values[test_idx[dir_valid_test]]
            meta_bt = {}
            for name, margin, top_pct in BT_CONFIGS:
                trades = backtest_lo(close_meta_test, meta_p_test, horizon,
                                     FEE_MAKER_RT, margin=margin, top_pct=top_pct)
                meta_bt[f"meta_{name}"] = bt_stats(trades, horizon)

            # Also get base model backtest for comparison
            if "dir12" in meta_features_test:
                base_p = meta_features_test["dir12"][dir_valid_test[:len(meta_features_test["dir12"])]]
                for name, margin, top_pct in [("m06_top10", 0.06, 0.10),
                                               ("m04_top10", 0.04, 0.10)]:
                    if len(base_p) == len(close_meta_test):
                        trades = backtest_lo(close_meta_test, base_p, horizon,
                                             FEE_MAKER_RT, margin=margin, top_pct=top_pct)
                        meta_bt[f"base_{name}"] = bt_stats(trades, horizon)

            # Print key results
            for key in ["meta_m06_top10", "meta_m04_top10", "meta_m04_top20",
                         "base_m06_top10", "base_m04_top10"]:
                if key in meta_bt and meta_bt[key]["n"] > 0:
                    sb = meta_bt[key]
                    log(f"    {key}: n={sb['n']:>4d}  net={sb['net']:>+7.2%}  "
                        f"WR={sb['wr']:>5.1%}  kelly={sb['kelly']:>+6.3f}", f)

            meta_results.append({"auc_base": base_auc, "auc_meta": meta_auc,
                                 "bt": meta_bt})

        # Meta summary
        log(f"\n  ──── META-MODEL SUMMARY ────", f)
        base_a = [r["auc_base"] for r in meta_results if r is not None]
        meta_a = [r["auc_meta"] for r in meta_results if r is not None]
        if base_a:
            log(f"  Base AUC: {np.mean(base_a):.4f}±{np.std(base_a):.4f}", f)
            log(f"  Meta AUC: {np.mean(meta_a):.4f}±{np.std(meta_a):.4f}", f)
            log(f"  Delta:    {np.mean(meta_a)-np.mean(base_a):>+.4f}", f)

        for cfg in ["meta_m06_top10", "meta_m04_top10", "meta_m04_top20",
                     "base_m06_top10", "base_m04_top10"]:
            nets = [r["bt"][cfg]["net"] for r in meta_results
                    if r is not None and cfg in r.get("bt", {}) and r["bt"][cfg]["n"] > 0]
            if nets:
                pos = sum(1 for n in nets if n > 0)
                log(f"  {cfg}: {pos}/{len(nets)} pos, avg={np.mean(nets):>+7.2%}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: ASYMMETRIC TARGET — MAX RUNUP > FEE (10-split WF)
        # Predict: will max runup in next 12 candles > 0.04%?
        # This directly targets whether a long trade will be profitable
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 5: ASYMMETRIC TARGET — MAX RUNUP > FEE (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        # Create asymmetric targets: max_runup > threshold
        for thresh_label, thresh_val in [("04pct", 0.004), ("06pct", 0.006),
                                          ("08pct", 0.008), ("10pct", 0.010)]:
            col_name = f"target_runup_gt_{thresh_label}"
            if "target_max_runup_12" in df.columns:
                df[col_name] = (df["target_max_runup_12"] > thresh_val).astype(float)
                df.loc[df["target_max_runup_12"].isna(), col_name] = np.nan

        asymmetric_targets = [
            ("runup_gt_04pct", "target_runup_gt_04pct", 12),
            ("runup_gt_06pct", "target_runup_gt_06pct", 12),
            ("runup_gt_08pct", "target_runup_gt_08pct", 12),
            ("runup_gt_10pct", "target_runup_gt_10pct", 12),
            ("direction_12",   "target_direction_12",   12),  # baseline
        ]

        horizon = 12

        for tgt_name, tgt_col, horizon in asymmetric_targets:
            if tgt_col not in df.columns:
                log(f"  SKIP {tgt_name}: not found", f)
                continue

            # Check class balance
            valid_mask = ~df[tgt_col].isna()
            pos_rate = df.loc[valid_mask, tgt_col].mean()
            log(f"  === {tgt_name} (pos_rate={pos_rate:.3f}) ===", f)

            split_results = []
            for s in range(n_splits):
                test_end = n_total - s * test_size
                test_start = test_end - test_size
                train_end = test_start - purge

                if train_end < n_total * 0.3:
                    split_results.append(None)
                    continue

                result = run_wf_split(df, top80_1h, tgt_col, horizon,
                                      train_end, test_start, test_end, algo="lgb", f=f)
                if result is None:
                    split_results.append(None)
                    continue

                bt = evaluate_bt_configs(result["close_te"], result["p_raw"],
                                         result["p_cal"], horizon, BT_CONFIGS)
                result["bt"] = bt

                date_range = get_date_range(df.iloc[test_start:test_end])
                m06_s = bt.get("raw_m06_top10", {})
                m04_s = bt.get("raw_m04_top10", {})
                log(f"    Split {s+1}: AUC={result['auc_raw']:.4f}  "
                    f"m06t10: n={m06_s.get('n',0):>3d} net={m06_s.get('net',0):>+7.2%}  "
                    f"m04t10: n={m04_s.get('n',0):>3d} net={m04_s.get('net',0):>+7.2%}", f)

                split_results.append(result)

            # Summary
            aucs = [r["auc_raw"] for r in split_results if r is not None]
            if aucs:
                log(f"  {tgt_name} AUC: {np.mean(aucs):.4f}±{np.std(aucs):.4f}", f)
            for cfg in ["raw_m06_top10", "raw_m04_top10", "raw_m04_top20"]:
                nets = [r["bt"][cfg]["net"] for r in split_results
                        if r is not None and cfg in r.get("bt", {}) and r["bt"][cfg]["n"] > 0]
                if nets:
                    pos = sum(1 for n in nets if n > 0)
                    log(f"    {cfg}: {pos}/{len(nets)} pos, avg={np.mean(nets):>+7.2%}", f)
            log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 6: BEST-OF COMPARISON & COMBINED STRATEGY
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 6: BEST-OF COMPARISON", f)
        log(f"{'#'*80}\n", f)

        log(f"  Comparing all approaches on 1h direction with raw_m06_top10:", f)
        log(f"  (Summary of best results from each phase)\n", f)
        log(f"  Approach                   Avg AUC    m06_top10 Pos/N  Avg Net", f)
        log(f"  {'─'*70}", f)

        # Collect best from each phase for summary
        # Phase 1 direction baseline (already computed)
        log(f"  See detailed results above for each phase.", f)

        # ═══════════════════════════════════════════════════════════════
        # FINAL: Save best models
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  ITERATION 7 — FINAL VERDICT", f)
        log(f"{'#'*80}\n", f)

        log(f"  Results to be analyzed after run completes.", f)
        log(f"  Key questions:", f)
        log(f"    1. Do threshold targets improve AUC over direction?", f)
        log(f"    2. Does XGBoost beat LightGBM?", f)
        log(f"    3. Does recency weighting help?", f)
        log(f"    4. Does stacking add value?", f)
        log(f"    5. Does asymmetric runup target improve trading?", f)

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
