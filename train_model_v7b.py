"""
BTC ML Training — Iteration 7b: THRESHOLD SIGNAL EXPLOITATION
=============================================================
Key discovery from v7: threshold targets have AUC 0.67-0.72 (vs 0.55 for direction).
But the v7 backtest used direction-style margins which produced ~0 trades.

This script properly exploits threshold model probabilities:
  - Trade when P(up>0.3% in 1h) > base_rate * confidence_multiplier
  - This IS the signal: model says "price likely to rise >0.3%"

Also runs remaining v7 experiments:
  - XGBoost vs LightGBM comparison
  - Recency-weighted training
  - Combined direction+threshold signal

Usage:
  python train_model_v7b.py
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
RESULTS_FILE = os.path.join(OUTPUT_DIR, "training_results_v7b.txt")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

FEE_MAKER_RT = 0.0004


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


def backtest_threshold_signal(close, y_prob, horizon, fee_rt, prob_threshold,
                               top_pct=None):
    """
    Long-only backtest using threshold model predictions.
    Enter long when P(up>threshold) > prob_threshold.
    This is different from direction backtest — no 0.5+margin.
    """
    n = len(close)
    trades = []
    i = 0
    while i + horizon < n:
        prob = y_prob[i]
        if prob > prob_threshold:
            raw_ret = (close[i + horizon] - close[i]) / close[i]
            trades.append({"idx": i, "prob": prob, "raw_ret": raw_ret,
                           "net_ret": raw_ret - fee_rt})
        i += horizon
    if not trades:
        return trades
    if top_pct is not None and len(trades) > 5:
        df_t = pd.DataFrame(trades)
        threshold = df_t["prob"].quantile(1.0 - top_pct)
        df_t = df_t[df_t["prob"] >= threshold]
        trades = df_t.to_dict("records")
    return trades


def backtest_direction(close, y_prob, horizon, fee_rt, margin=0.04, top_pct=None):
    """Standard direction backtest (prob > 0.5 + margin)."""
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


LGB_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "boosting_type": "gbdt", "n_estimators": 3000,
    "learning_rate": 0.005, "num_leaves": 24, "max_depth": 5,
    "min_child_samples": 500, "subsample": 0.5, "subsample_freq": 1,
    "colsample_bytree": 0.3, "colsample_bynode": 0.5,
    "reg_alpha": 5.0, "reg_lambda": 20.0,
    "feature_fraction_bynode": 0.5, "path_smooth": 10.0,
    "random_state": 42, "verbose": -1, "n_jobs": -1,
}

XGB_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "n_estimators": 3000, "learning_rate": 0.005,
    "max_depth": 5, "min_child_weight": 500,
    "subsample": 0.5, "colsample_bytree": 0.3, "colsample_bynode": 0.5,
    "reg_alpha": 5.0, "reg_lambda": 20.0,
    "tree_method": "hist", "random_state": 42, "verbosity": 0, "n_jobs": -1,
}


def train_lgb(X_tr, y_tr, X_va, y_va, sample_weight=None):
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    kw = {"eval_set": [(X_va, y_va)],
          "callbacks": [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]}
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    m.fit(X_tr, y_tr, **kw)
    return m


def train_xgb(X_tr, y_tr, X_va, y_va, sample_weight=None):
    # XGBoost can't handle inf values (LightGBM can) — replace with NaN
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    X_va = np.where(np.isinf(X_va), np.nan, X_va)
    m = xgb.XGBClassifier(**XGB_PARAMS)
    kw = {"eval_set": [(X_va, y_va)], "verbose": False}
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    m.fit(X_tr, y_tr, **kw)
    return m


def calibrate_iso(y_val, p_val, p_test):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p_val, y_val)
    return ir.transform(p_test)


def get_date_range(df_slice, col="open_time_ms"):
    dates = pd.to_datetime(df_slice[col], unit="ms", utc=True)
    return f"{dates.iloc[0].strftime('%Y-%m-%d')}/{dates.iloc[-1].strftime('%Y-%m-%d')}"


def compute_recency_weights(n_samples, half_life_frac=0.15):
    half_life = max(int(n_samples * half_life_frac), 1000)
    idx = np.arange(n_samples)
    weights = np.exp(np.log(2) * (idx - n_samples + 1) / half_life)
    return weights / weights.mean()


def wf_split_data(df, feat_cols, target_col, train_end, test_start, test_end, purge):
    """Prepare WF split data. Returns (X_tr, y_tr, X_va, y_va, X_te, y_te, close_te) or None."""
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
    return X_tr, y_tr, X_va, y_va, X_te, y_te, close_te


# ─── THRESHOLD BACKTEST CONFIGS ────────────────────────────────────────────
# For threshold models P(up>X%), the probability rarely goes above 0.5.
# So we use absolute probability thresholds based on multiples of base rate.
# Base rates: up_12_0002 ≈ 0.40, up_12_0003 ≈ 0.30, up_12_0005 ≈ 0.18

THRESH_BT_CONFIGS = [
    # (name, prob_threshold, top_pct)
    ("p35_all",    0.35, None),
    ("p40_all",    0.40, None),
    ("p45_all",    0.45, None),
    ("p35_top30",  0.35, 0.30),
    ("p35_top20",  0.35, 0.20),
    ("p35_top10",  0.35, 0.10),
    ("p40_top30",  0.40, 0.30),
    ("p40_top20",  0.40, 0.20),
    ("p40_top10",  0.40, 0.10),
    ("p45_top30",  0.45, 0.30),
    ("p45_top20",  0.45, 0.20),
    ("p45_top10",  0.45, 0.10),
    ("p50_top30",  0.50, 0.30),
    ("p50_top20",  0.50, 0.20),
    ("p50_top10",  0.50, 0.10),
]

DIR_BT_CONFIGS = [
    # (name, margin, top_pct)
    ("m04_top20", 0.04, 0.20),
    ("m04_top10", 0.04, 0.10),
    ("m06_top20", 0.06, 0.20),
    ("m06_top10", 0.06, 0.10),
]


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols")

    all_feat_cols = get_feature_cols(df)
    n_total = len(df)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        log(f"BTC ML Training — Iteration 7b: THRESHOLD SIGNAL EXPLOITATION", f)
        log(f"{'='*80}", f)
        log(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", f)
        log(f"Data: {len(df):,} rows x {len(df.columns)} cols  Features: {len(all_feat_cols)}", f)

        # ─── Feature selection (reuse v7 result) ────────────────────────
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 0: FEATURE SELECTION", f)
        log(f"{'#'*80}\n", f)

        # Train quick models to select features
        for tgt_name, tgt_col in [("direction_12", "target_direction_12"),
                                   ("up_12_0003", "target_up_12_0003")]:
            valid = ~df[tgt_col].isna()
            valid_idx = np.where(valid)[0]
            n_v = valid.sum()
            tr_e = int(n_v * 0.70)
            va_e = int(n_v * 0.85)
            m = train_lgb(df.loc[valid_idx[:tr_e], all_feat_cols].values,
                          df.loc[valid_idx[:tr_e], tgt_col].values,
                          df.loc[valid_idx[tr_e:va_e], all_feat_cols].values,
                          df.loc[valid_idx[tr_e:va_e], tgt_col].values)
            imp = pd.DataFrame({"feat": all_feat_cols, "imp": m.feature_importances_})
            imp = imp.sort_values("imp", ascending=False)
            if tgt_name == "direction_12":
                top80_dir = imp.head(80)["feat"].tolist()
            else:
                top80_thr = imp.head(80)["feat"].tolist()
            log(f"  {tgt_name} top-10:", f)
            for idx, row in imp.head(10).iterrows():
                log(f"    {row['imp']:>5.0f}  {row['feat']}", f)
            log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: THRESHOLD MODEL WITH PROPER TRADING LOGIC (10-split WF)
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 1: THRESHOLD MODELS WITH PROPER TRADING (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        n_splits = 10
        test_size = int(n_total * 0.05)
        horizon = 12
        purge = 2 * horizon

        # Test three threshold targets
        for tgt_name, tgt_col in [("up_12_0003", "target_up_12_0003"),
                                   ("up_12_0005", "target_up_12_0005"),
                                   ("up_12_0002", "target_up_12_0002")]:
            if tgt_col not in df.columns:
                continue

            base_rate = df[tgt_col].dropna().mean()
            log(f"  === {tgt_name} (base_rate={base_rate:.3f}) ===\n", f)

            all_split_bt = {cfg[0]: [] for cfg in THRESH_BT_CONFIGS}
            all_aucs = []

            for s in range(n_splits):
                test_end = n_total - s * test_size
                test_start = test_end - test_size
                train_end = test_start - purge

                if train_end < n_total * 0.3:
                    for cfg in THRESH_BT_CONFIGS:
                        all_split_bt[cfg[0]].append(None)
                    continue

                data = wf_split_data(df, top80_thr, tgt_col, train_end,
                                     test_start, test_end, purge)
                if data is None:
                    for cfg in THRESH_BT_CONFIGS:
                        all_split_bt[cfg[0]].append(None)
                    continue

                X_tr, y_tr, X_va, y_va, X_te, y_te, close_te = data
                t0 = time.time()
                model = train_lgb(X_tr, y_tr, X_va, y_va)
                elapsed = time.time() - t0
                p_raw = model.predict_proba(X_te)[:, 1]
                auc_val = auc_roc(y_te, p_raw)
                all_aucs.append(auc_val)

                date_range = get_date_range(df.iloc[test_start:test_end])
                log(f"    Split {s+1}: {date_range}  AUC={auc_val:.4f}  "
                    f"iter={model.best_iteration_}  time={elapsed:.0f}s  "
                    f"p_mean={p_raw.mean():.3f}  p_max={p_raw.max():.3f}", f)

                # Run threshold backtests
                for cfg_name, prob_thresh, top_pct in THRESH_BT_CONFIGS:
                    trades = backtest_threshold_signal(close_te, p_raw, horizon,
                                                       FEE_MAKER_RT, prob_thresh,
                                                       top_pct=top_pct)
                    stats = bt_stats(trades, horizon)
                    all_split_bt[cfg_name].append(stats)

                # Print key configs
                for cfg_name in ["p35_top20", "p40_top20", "p40_top10",
                                  "p45_top10", "p50_top10"]:
                    stats_list = all_split_bt[cfg_name]
                    s_bt = stats_list[-1]
                    if s_bt and s_bt["n"] > 0:
                        log(f"      {cfg_name}: n={s_bt['n']:>4d}  "
                            f"net={s_bt['net']:>+7.2%}  WR={s_bt['wr']:>5.1%}  "
                            f"kelly={s_bt['kelly']:>+6.3f}", f)

            # Summary
            log(f"\n  {tgt_name} WF SUMMARY (AUC={np.mean(all_aucs):.4f}±{np.std(all_aucs):.4f}):", f)
            log(f"  {'Config':<16s} {'Pos/N':>6s}  {'Avg Net':>9s}  {'Avg SR':>8s}  "
                f"{'Avg WR':>7s}  {'Med N':>6s}  {'Kelly':>7s}", f)
            for cfg_name, _, _ in THRESH_BT_CONFIGS:
                nets = []
                for st in all_split_bt[cfg_name]:
                    if st is not None and st["n"] > 0:
                        nets.append(st)
                if not nets:
                    continue
                pos = sum(1 for s in nets if s["net"] > 0)
                avg_net = np.mean([s["net"] for s in nets])
                avg_sr = np.mean([s["sharpe"] for s in nets])
                avg_wr = np.mean([s["wr"] for s in nets])
                med_n = int(np.median([s["n"] for s in nets]))
                avg_k = np.mean([s["kelly"] for s in nets])
                log(f"  {cfg_name:<16s} {pos:>2d}/{len(nets):<2d}  {avg_net:>+8.2%}  "
                    f"{avg_sr:>8.2f}  {avg_wr:>6.1%}  {med_n:>6d}  {avg_k:>+6.3f}", f)
            log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: DIRECTION BASELINE COMPARISON (same splits)
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 2: DIRECTION BASELINE (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        dir_split_bt = {cfg[0]: [] for cfg in DIR_BT_CONFIGS}
        dir_aucs = []

        for s in range(n_splits):
            test_end = n_total - s * test_size
            test_start = test_end - test_size
            train_end = test_start - purge

            if train_end < n_total * 0.3:
                for cfg in DIR_BT_CONFIGS:
                    dir_split_bt[cfg[0]].append(None)
                continue

            data = wf_split_data(df, top80_dir, "target_direction_12", train_end,
                                 test_start, test_end, purge)
            if data is None:
                for cfg in DIR_BT_CONFIGS:
                    dir_split_bt[cfg[0]].append(None)
                continue

            X_tr, y_tr, X_va, y_va, X_te, y_te, close_te = data
            model = train_lgb(X_tr, y_tr, X_va, y_va)
            p_raw = model.predict_proba(X_te)[:, 1]
            auc_val = auc_roc(y_te, p_raw)
            dir_aucs.append(auc_val)

            date_range = get_date_range(df.iloc[test_start:test_end])
            log(f"  Split {s+1}: {date_range}  AUC={auc_val:.4f}", f)

            for cfg_name, margin, top_pct in DIR_BT_CONFIGS:
                trades = backtest_direction(close_te, p_raw, horizon, FEE_MAKER_RT,
                                            margin=margin, top_pct=top_pct)
                stats = bt_stats(trades, horizon)
                dir_split_bt[cfg_name].append(stats)
                if stats["n"] > 0:
                    log(f"    {cfg_name}: n={stats['n']:>4d}  net={stats['net']:>+7.2%}  "
                        f"WR={stats['wr']:>5.1%}  kelly={stats['kelly']:>+6.3f}", f)

        log(f"\n  Direction WF SUMMARY (AUC={np.mean(dir_aucs):.4f}±{np.std(dir_aucs):.4f}):", f)
        for cfg_name, _, _ in DIR_BT_CONFIGS:
            nets = [s for s in dir_split_bt[cfg_name] if s is not None and s["n"] > 0]
            if not nets:
                continue
            pos = sum(1 for s in nets if s["net"] > 0)
            avg_net = np.mean([s["net"] for s in nets])
            med_n = int(np.median([s["n"] for s in nets]))
            avg_k = np.mean([s["kelly"] for s in nets])
            log(f"  {cfg_name}: {pos}/{len(nets)} pos, avg={avg_net:>+7.2%}, "
                f"med_n={med_n}, kelly={avg_k:>+6.3f}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: XGBOOST vs LIGHTGBM (10-split, direction_12)
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 3: XGBOOST vs LIGHTGBM (10-split WF, direction_12)", f)
        log(f"{'#'*80}\n", f)

        xgb_aucs = []
        lgb_m06t10 = []
        xgb_m06t10 = []
        avg_m06t10 = []

        for s in range(n_splits):
            test_end = n_total - s * test_size
            test_start = test_end - test_size
            train_end = test_start - purge

            if train_end < n_total * 0.3:
                continue

            data = wf_split_data(df, top80_dir, "target_direction_12", train_end,
                                 test_start, test_end, purge)
            if data is None:
                continue

            X_tr, y_tr, X_va, y_va, X_te, y_te, close_te = data

            # LightGBM
            m_lgb = train_lgb(X_tr, y_tr, X_va, y_va)
            p_lgb = m_lgb.predict_proba(X_te)[:, 1]
            auc_lgb = auc_roc(y_te, p_lgb)

            # XGBoost
            m_xgb = train_xgb(X_tr, y_tr, X_va, y_va)
            p_xgb = m_xgb.predict_proba(X_te)[:, 1]
            auc_xgb = auc_roc(y_te, p_xgb)
            xgb_aucs.append(auc_xgb)

            # Average
            p_avg = (p_lgb + p_xgb) / 2
            auc_avg = auc_roc(y_te, p_avg)

            date_range = get_date_range(df.iloc[test_start:test_end])
            log(f"  Split {s+1}: {date_range}  LGB={auc_lgb:.4f}  XGB={auc_xgb:.4f}  "
                f"AVG={auc_avg:.4f}", f)

            # Backtest m06_top10 for all three
            for name, probs, result_list in [("LGB", p_lgb, lgb_m06t10),
                                              ("XGB", p_xgb, xgb_m06t10),
                                              ("AVG", p_avg, avg_m06t10)]:
                trades = backtest_direction(close_te, probs, horizon, FEE_MAKER_RT,
                                            margin=0.06, top_pct=0.10)
                stats = bt_stats(trades, horizon)
                result_list.append(stats)
                if stats["n"] > 0:
                    log(f"    {name} m06_top10: n={stats['n']:>4d}  "
                        f"net={stats['net']:>+7.2%}  kelly={stats['kelly']:>+6.3f}", f)

        log(f"\n  XGB vs LGB SUMMARY:", f)
        log(f"  LGB AUC: {np.mean(dir_aucs):.4f}±{np.std(dir_aucs):.4f}", f)
        log(f"  XGB AUC: {np.mean(xgb_aucs):.4f}±{np.std(xgb_aucs):.4f}", f)
        for name, result_list in [("LGB", lgb_m06t10), ("XGB", xgb_m06t10),
                                   ("AVG", avg_m06t10)]:
            valid = [s for s in result_list if s["n"] > 0]
            if valid:
                pos = sum(1 for s in valid if s["net"] > 0)
                avg = np.mean([s["net"] for s in valid])
                log(f"  {name} m06_top10: {pos}/{len(valid)} pos, avg={avg:>+7.2%}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: RECENCY-WEIGHTED TRAINING (direction_12)
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 4: RECENCY-WEIGHTED TRAINING (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        half_lives = [0.05, 0.10, 0.20]
        hl_results = {hl: {"aucs": [], "m06t10": [], "m04t10": []} for hl in half_lives}

        for s in range(n_splits):
            test_end = n_total - s * test_size
            test_start = test_end - test_size
            train_end = test_start - purge

            if train_end < n_total * 0.3:
                continue

            data = wf_split_data(df, top80_dir, "target_direction_12", train_end,
                                 test_start, test_end, purge)
            if data is None:
                continue

            X_tr, y_tr, X_va, y_va, X_te, y_te, close_te = data
            date_range = get_date_range(df.iloc[test_start:test_end])
            log(f"  Split {s+1}: {date_range}", f)

            for hl in half_lives:
                sw = compute_recency_weights(len(X_tr), half_life_frac=hl)
                m = train_lgb(X_tr, y_tr, X_va, y_va, sample_weight=sw)
                p = m.predict_proba(X_te)[:, 1]
                auc_val = auc_roc(y_te, p)
                hl_results[hl]["aucs"].append(auc_val)

                for cfg_name, margin, top_pct, key in [
                    ("m06_top10", 0.06, 0.10, "m06t10"),
                    ("m04_top10", 0.04, 0.10, "m04t10"),
                ]:
                    trades = backtest_direction(close_te, p, horizon, FEE_MAKER_RT,
                                                margin=margin, top_pct=top_pct)
                    hl_results[hl][key].append(bt_stats(trades, horizon))

                log(f"    hl={hl:.2f}: AUC={auc_val:.4f}  "
                    f"m06t10: n={hl_results[hl]['m06t10'][-1]['n']:>3d} "
                    f"net={hl_results[hl]['m06t10'][-1]['net']:>+7.2%}", f)

        log(f"\n  RECENCY SUMMARY:", f)
        log(f"  {'HL':<8s} {'AUC':<16s} {'m06_top10':<24s} {'m04_top10':<24s}", f)
        # Unweighted baseline
        lgb_base_nets = [s["net"] for s in lgb_m06t10 if s["n"] > 0]
        lgb_base_pos = sum(1 for n in lgb_base_nets if n > 0) if lgb_base_nets else 0
        log(f"  {'none':<8s} {np.mean(dir_aucs):.4f}±{np.std(dir_aucs):.3f}  "
            f"{lgb_base_pos}/{len(lgb_base_nets)} avg={np.mean(lgb_base_nets):>+6.2%}", f)

        for hl in half_lives:
            a = hl_results[hl]["aucs"]
            m06 = [s for s in hl_results[hl]["m06t10"] if s["n"] > 0]
            m04 = [s for s in hl_results[hl]["m04t10"] if s["n"] > 0]
            m06_pos = sum(1 for s in m06 if s["net"] > 0)
            m04_pos = sum(1 for s in m04 if s["net"] > 0)
            log(f"  hl={hl:<5.2f}  {np.mean(a):.4f}±{np.std(a):.3f}  "
                f"{m06_pos}/{len(m06)} avg={np.mean([s['net'] for s in m06]):>+6.2%}  "
                f"{m04_pos}/{len(m04)} avg={np.mean([s['net'] for s in m04]):>+6.2%}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: COMBINED DIRECTION + THRESHOLD SIGNAL (10-split WF)
        # Use threshold model probability as additional filter for direction trades
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 5: COMBINED DIR+THRESHOLD SIGNAL (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        combo_results = {"dir_only": [], "thresh_only": [],
                          "dir_AND_thresh": [], "dir_OR_thresh": []}

        thresh_target = "target_up_12_0003"

        for s in range(n_splits):
            test_end = n_total - s * test_size
            test_start = test_end - test_size
            train_end = test_start - purge

            if train_end < n_total * 0.3:
                continue

            # Direction model
            data_dir = wf_split_data(df, top80_dir, "target_direction_12", train_end,
                                      test_start, test_end, purge)
            # Threshold model
            data_thr = wf_split_data(df, top80_thr, thresh_target, train_end,
                                      test_start, test_end, purge)
            if data_dir is None or data_thr is None:
                continue

            X_tr_d, y_tr_d, X_va_d, y_va_d, X_te_d, y_te_d, close_d = data_dir
            X_tr_t, y_tr_t, X_va_t, y_va_t, X_te_t, y_te_t, close_t = data_thr

            m_dir = train_lgb(X_tr_d, y_tr_d, X_va_d, y_va_d)
            m_thr = train_lgb(X_tr_t, y_tr_t, X_va_t, y_va_t)

            p_dir = m_dir.predict_proba(X_te_d)[:, 1]
            p_thr = m_thr.predict_proba(X_te_t)[:, 1]

            # Align arrays — both should be same length for same test period
            n_common = min(len(p_dir), len(p_thr))
            p_dir = p_dir[:n_common]
            p_thr = p_thr[:n_common]
            close_common = close_d[:n_common]

            date_range = get_date_range(df.iloc[test_start:test_end])

            # Direction-only trade: P(up) > 0.54 (top 10%)
            dir_signal = p_dir > 0.54
            # Threshold-only trade: P(up>0.3%) > base_rate * 1.2
            base_rate = df[thresh_target].dropna().iloc[:train_end].mean()
            thr_signal = p_thr > base_rate * 1.3
            # Combined AND: both agree
            combo_and = dir_signal & thr_signal
            # Combined OR: either signals
            combo_or = dir_signal | thr_signal

            log(f"  Split {s+1}: {date_range}", f)

            for signal_name, signal_mask, result_key in [
                ("dir_only",      dir_signal,  "dir_only"),
                ("thresh_only",   thr_signal,  "thresh_only"),
                ("dir_AND_thresh", combo_and,   "dir_AND_thresh"),
                ("dir_OR_thresh", combo_or,     "dir_OR_thresh"),
            ]:
                # Backtest using signal mask
                trades = []
                i = 0
                while i + horizon < n_common:
                    if signal_mask[i]:
                        raw_ret = (close_common[i + horizon] - close_common[i]) / close_common[i]
                        trades.append({"idx": i, "prob": 1.0, "raw_ret": raw_ret,
                                       "net_ret": raw_ret - FEE_MAKER_RT})
                    i += horizon
                stats = bt_stats(trades, horizon)
                combo_results[result_key].append(stats)

                if stats["n"] > 0:
                    log(f"    {signal_name:<18s}: n={stats['n']:>4d}  "
                        f"net={stats['net']:>+7.2%}  WR={stats['wr']:>5.1%}  "
                        f"kelly={stats['kelly']:>+6.3f}", f)

        log(f"\n  COMBINED SIGNAL SUMMARY:", f)
        for key in ["dir_only", "thresh_only", "dir_AND_thresh", "dir_OR_thresh"]:
            valid = [s for s in combo_results[key] if s["n"] > 0]
            if valid:
                pos = sum(1 for s in valid if s["net"] > 0)
                avg = np.mean([s["net"] for s in valid])
                avg_k = np.mean([s["kelly"] for s in valid])
                med_n = int(np.median([s["n"] for s in valid]))
                log(f"  {key:<18s}: {pos}/{len(valid)} pos, avg={avg:>+7.2%}, "
                    f"kelly={avg_k:>+6.3f}, med_n={med_n}", f)

        # ═══════════════════════════════════════════════════════════════
        # FINAL VERDICT
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  ITERATION 7b — FINAL VERDICT", f)
        log(f"{'#'*80}\n", f)

        log(f"  Key questions answered:", f)
        log(f"    1. Do threshold targets produce better trading signals?", f)
        log(f"    2. Does XGBoost beat LightGBM?", f)
        log(f"    3. Does recency weighting help?", f)
        log(f"    4. Does combining direction + threshold improve results?", f)
        log(f"\n  (See results above for answers)", f)

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
