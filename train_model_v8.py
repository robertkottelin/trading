"""
BTC ML Training — Iteration 8: MULTI-MODEL PORTFOLIO & 3H THRESHOLD
====================================================================
Building on v7b breakthrough: threshold targets >> direction targets.
Best configs so far:
  - up_12_0002 p45_top20: 10/10 WF positive, +4.71% avg
  - up_12_0005 p45_all: 9/9 WF positive, +3.34% avg, Kelly 0.632
  - up_12_0003 p35_top10: 9/10 WF positive, +4.45% avg

Key experiments:
  Phase 1: 3h threshold targets (up_36_0003, up_36_0005) — new independent signal
  Phase 2: Short-side (down_12_0003, down_12_0005) — double trade count
  Phase 3: LGB+XGB ensemble on threshold targets — boost 9/10 → 10/10
  Phase 4: Regime filtering on threshold models
  Phase 5: Multi-model portfolio backtest — combined equity curve

Usage:
  python train_model_v8.py
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
RESULTS_FILE = os.path.join(OUTPUT_DIR, "training_results_v8.txt")
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


def backtest_short_threshold(close, y_prob, horizon, fee_rt, prob_threshold,
                              top_pct=None):
    """Short-side backtest: enter SHORT when P(down>threshold) > prob_threshold."""
    n = len(close)
    trades = []
    i = 0
    while i + horizon < n:
        prob = y_prob[i]
        if prob > prob_threshold:
            raw_ret = (close[i] - close[i + horizon]) / close[i]  # short = inverted
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


def train_lgb(X_tr, y_tr, X_va, y_va):
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return m


def train_xgb(X_tr, y_tr, X_va, y_va):
    X_tr = np.where(np.isinf(X_tr), np.nan, X_tr)
    X_va = np.where(np.isinf(X_va), np.nan, X_va)
    m = xgb.XGBClassifier(**XGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def get_date_range(df_slice, col="open_time_ms"):
    dates = pd.to_datetime(df_slice[col], unit="ms", utc=True)
    return f"{dates.iloc[0].strftime('%Y-%m-%d')}/{dates.iloc[-1].strftime('%Y-%m-%d')}"


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
    return X_tr, y_tr, X_va, y_va, X_te, y_te, close_te


# Best configs from v7b for each target
BEST_CONFIGS = {
    "up_12_0002": [("p45_top20", 0.45, 0.20), ("p45_all", 0.45, None)],
    "up_12_0003": [("p35_top10", 0.35, 0.10), ("p40_top10", 0.40, 0.10)],
    "up_12_0005": [("p45_all", 0.45, None), ("p35_top20", 0.35, 0.20)],
}

# Comprehensive configs for new targets
THRESH_BT_CONFIGS = [
    ("p35_all",    0.35, None),
    ("p40_all",    0.40, None),
    ("p45_all",    0.45, None),
    ("p35_top20",  0.35, 0.20),
    ("p35_top10",  0.35, 0.10),
    ("p40_top20",  0.40, 0.20),
    ("p40_top10",  0.40, 0.10),
    ("p45_top20",  0.45, 0.20),
    ("p45_top10",  0.45, 0.10),
    ("p50_top20",  0.50, 0.20),
    ("p50_top10",  0.50, 0.10),
]


def run_threshold_wf(df, feat_cols, tgt_name, tgt_col, horizon, n_splits,
                      test_size, purge, bt_configs, f, backtest_fn=backtest_threshold_signal):
    """Run 10-split WF for a threshold target. Returns per-split AUC and trade stats."""
    if tgt_col not in df.columns:
        log(f"  {tgt_col} not found — skipping", f)
        return None

    n_total = len(df)
    base_rate = df[tgt_col].dropna().mean()
    log(f"  === {tgt_name} (base_rate={base_rate:.3f}) ===\n", f)

    all_split_bt = {cfg[0]: [] for cfg in bt_configs}
    all_aucs = []

    for s in range(n_splits):
        test_end = n_total - s * test_size
        test_start = test_end - test_size
        train_end = test_start - purge

        if train_end < 10000:
            continue

        data = wf_split_data(df, feat_cols, tgt_col, train_end, test_start, test_end, purge)
        if data is None:
            continue

        X_tr, y_tr, X_va, y_va, X_te, y_te, close_te = data

        t0 = time.time()
        model = train_lgb(X_tr, y_tr, X_va, y_va)
        elapsed = time.time() - t0
        p_test = model.predict_proba(X_te)[:, 1]
        auc = auc_roc(y_te, p_test)
        all_aucs.append(auc)

        iters = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.n_estimators
        test_df = df.iloc[np.where((np.arange(n_total) >= test_start) &
                                    (np.arange(n_total) < test_end) &
                                    ~np.isnan(df[tgt_col].values))[0]]
        dr = get_date_range(test_df)

        log(f"    Split {s+1}: {dr}  AUC={auc:.4f}  iter={iters}  time={elapsed:.0f}s  "
            f"p_mean={p_test.mean():.3f}  p_max={p_test.max():.3f}", f)

        for cfg_name, prob_thresh, top_pct in bt_configs:
            trades = backtest_fn(close_te, p_test, horizon, FEE_MAKER_RT,
                                  prob_thresh, top_pct)
            stats = bt_stats(trades, horizon)
            all_split_bt[cfg_name].append(stats)
            if stats["n"] > 0:
                log(f"      {cfg_name}: n={stats['n']:>4d}  net={stats['net']:>+7.2%}  "
                    f"WR={stats['wr']:.1%}  kelly={stats['kelly']:>+.3f}", f)

    if not all_aucs:
        return None

    # Summary
    auc_mean = np.mean(all_aucs)
    auc_std = np.std(all_aucs)
    log(f"\n  {tgt_name} WF SUMMARY (AUC={auc_mean:.4f}\u00b1{auc_std:.4f}):", f)
    log(f"  {'Config':<18s} {'Pos/N':>8s} {'Avg Net':>10s} {'Avg WR':>8s} {'Med N':>7s} {'Kelly':>8s}", f)

    results = {}
    for cfg_name, _, _ in bt_configs:
        splits = all_split_bt[cfg_name]
        with_trades = [s for s in splits if s["n"] > 0]
        if not with_trades:
            continue
        pos = sum(1 for s in with_trades if s["net"] > 0)
        n_total_s = len(with_trades)
        avg_net = np.mean([s["net"] for s in with_trades])
        avg_wr = np.mean([s["wr"] for s in with_trades])
        med_n = int(np.median([s["n"] for s in with_trades]))
        avg_kelly = np.mean([s["kelly"] for s in with_trades])
        log(f"  {cfg_name:<18s} {pos:>3d}/{n_total_s:<3d}  {avg_net:>+8.2%}  "
            f"{avg_wr:>7.1%}  {med_n:>6d}  {avg_kelly:>+.3f}", f)
        results[cfg_name] = {"pos": pos, "total": n_total_s, "avg_net": avg_net,
                             "avg_wr": avg_wr, "med_n": med_n, "kelly": avg_kelly}

    return {"auc_mean": auc_mean, "auc_std": auc_std, "configs": results,
            "split_aucs": all_aucs, "split_bt": all_split_bt}


def run_ensemble_wf(df, feat_cols, tgt_name, tgt_col, horizon, n_splits,
                     test_size, purge, bt_configs, f,
                     backtest_fn=backtest_threshold_signal):
    """Run 10-split WF with LGB+XGB ensemble for a threshold target."""
    if tgt_col not in df.columns:
        return None

    n_total = len(df)
    base_rate = df[tgt_col].dropna().mean()
    log(f"  === {tgt_name} ENSEMBLE (base_rate={base_rate:.3f}) ===\n", f)

    all_split_bt = {f"{cfg[0]}_{src}": [] for cfg in bt_configs for src in ["lgb", "xgb", "avg"]}
    all_aucs = {"lgb": [], "xgb": [], "avg": []}

    for s in range(n_splits):
        test_end = n_total - s * test_size
        test_start = test_end - test_size
        train_end = test_start - purge

        if train_end < 10000:
            continue

        data = wf_split_data(df, feat_cols, tgt_col, train_end, test_start, test_end, purge)
        if data is None:
            continue

        X_tr, y_tr, X_va, y_va, X_te, y_te, close_te = data

        t0 = time.time()
        m_lgb = train_lgb(X_tr, y_tr, X_va, y_va)
        m_xgb = train_xgb(X_tr, y_tr, X_va, y_va)
        elapsed = time.time() - t0

        p_lgb = m_lgb.predict_proba(X_te)[:, 1]
        X_te_clean = np.where(np.isinf(X_te), np.nan, X_te)
        p_xgb = m_xgb.predict_proba(X_te_clean)[:, 1]
        p_avg = (p_lgb + p_xgb) / 2

        auc_lgb = auc_roc(y_te, p_lgb)
        auc_xgb = auc_roc(y_te, p_xgb)
        auc_avg = auc_roc(y_te, p_avg)
        all_aucs["lgb"].append(auc_lgb)
        all_aucs["xgb"].append(auc_xgb)
        all_aucs["avg"].append(auc_avg)

        test_df = df.iloc[np.where((np.arange(n_total) >= test_start) &
                                    (np.arange(n_total) < test_end) &
                                    ~np.isnan(df[tgt_col].values))[0]]
        dr = get_date_range(test_df)

        log(f"    Split {s+1}: {dr}  LGB={auc_lgb:.4f}  XGB={auc_xgb:.4f}  AVG={auc_avg:.4f}  time={elapsed:.0f}s", f)

        for cfg_name, prob_thresh, top_pct in bt_configs:
            for src, p in [("lgb", p_lgb), ("xgb", p_xgb), ("avg", p_avg)]:
                trades = backtest_fn(close_te, p, horizon, FEE_MAKER_RT,
                                      prob_thresh, top_pct)
                stats = bt_stats(trades, horizon)
                all_split_bt[f"{cfg_name}_{src}"].append(stats)

        # Show best config per source
        for src, p in [("lgb", p_lgb), ("xgb", p_xgb), ("avg", p_avg)]:
            best_cfg = bt_configs[0]
            best_trades = backtest_fn(close_te, p, horizon, FEE_MAKER_RT,
                                       best_cfg[1], best_cfg[2])
            best_stats = bt_stats(best_trades, horizon)
            if best_stats["n"] > 0:
                log(f"      {src.upper()} {best_cfg[0]}: n={best_stats['n']:>4d}  "
                    f"net={best_stats['net']:>+7.2%}  kelly={best_stats['kelly']:>+.3f}", f)

    # Summary
    log(f"\n  ENSEMBLE SUMMARY:", f)
    for src in ["lgb", "xgb", "avg"]:
        if all_aucs[src]:
            auc_m = np.mean(all_aucs[src])
            auc_s = np.std(all_aucs[src])
            log(f"    {src.upper()} AUC: {auc_m:.4f}\u00b1{auc_s:.4f}", f)

    log(f"  {'Config':<22s} {'Pos/N':>8s} {'Avg Net':>10s} {'Med N':>7s} {'Kelly':>8s}", f)
    for cfg_name, _, _ in bt_configs:
        for src in ["lgb", "xgb", "avg"]:
            key = f"{cfg_name}_{src}"
            splits = all_split_bt[key]
            with_trades = [s for s in splits if s["n"] > 0]
            if not with_trades:
                continue
            pos = sum(1 for s in with_trades if s["net"] > 0)
            n_s = len(with_trades)
            avg_net = np.mean([s["net"] for s in with_trades])
            med_n = int(np.median([s["n"] for s in with_trades]))
            avg_kelly = np.mean([s["kelly"] for s in with_trades])
            log(f"  {cfg_name}_{src:<3s}  {pos:>3d}/{n_s:<3d}  {avg_net:>+8.2%}  "
                f"{med_n:>6d}  {avg_kelly:>+.3f}", f)

    return all_aucs, all_split_bt


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols")

    all_feat_cols = get_feature_cols(df)
    n_total = len(df)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        log(f"BTC ML Training — Iteration 8: MULTI-MODEL PORTFOLIO & 3H THRESHOLD", f)
        log(f"{'='*80}", f)
        log(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", f)
        log(f"Data: {len(df):,} rows x {len(df.columns)} cols  Features: {len(all_feat_cols)}", f)

        # ─── Feature selection ──────────────────────────────────────────
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 0: FEATURE SELECTION", f)
        log(f"{'#'*80}\n", f)

        # Select top-80 features for each target type
        feat_cache = {}
        for tgt_name, tgt_col in [("up_12_0003", "target_up_12_0003"),
                                    ("up_36_0003", "target_up_36_0003"),
                                    ("down_12_0003", "target_down_12_0003")]:
            if tgt_col not in df.columns:
                log(f"  {tgt_col} not found — skipping feature selection", f)
                continue
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
            feat_cache[tgt_name] = imp.head(80)["feat"].tolist()
            log(f"  {tgt_name} top-10:", f)
            for idx, row in imp.head(10).iterrows():
                log(f"    {row['imp']:>5.0f}  {row['feat']}", f)
            log("", f)

        # Use up_12_0003 features for all 1h targets, up_36_0003 for 3h
        feat_1h = feat_cache.get("up_12_0003", all_feat_cols[:80])
        feat_3h = feat_cache.get("up_36_0003", all_feat_cols[:80])
        feat_down = feat_cache.get("down_12_0003", all_feat_cols[:80])

        n_splits = 10
        test_size = int(n_total * 0.05)
        purge_1h = 2 * 12
        purge_3h = 2 * 36

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: 3H THRESHOLD TARGETS (NEW)
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 1: 3H THRESHOLD TARGETS (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        results_3h = {}
        for tgt_name, tgt_col in [("up_36_0003", "target_up_36_0003"),
                                    ("up_36_0005", "target_up_36_0005"),
                                    ("up_36_0002", "target_up_36_0002")]:
            res = run_threshold_wf(df, feat_3h, tgt_name, tgt_col, 36,
                                   n_splits, test_size, purge_3h, THRESH_BT_CONFIGS, f)
            if res:
                results_3h[tgt_name] = res
            log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: SHORT-SIDE THRESHOLD TARGETS
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 2: SHORT-SIDE THRESHOLD TARGETS (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        results_short = {}
        for tgt_name, tgt_col in [("down_12_0003", "target_down_12_0003"),
                                    ("down_12_0005", "target_down_12_0005")]:
            res = run_threshold_wf(df, feat_down, tgt_name, tgt_col, 12,
                                   n_splits, test_size, purge_1h, THRESH_BT_CONFIGS, f,
                                   backtest_fn=backtest_short_threshold)
            if res:
                results_short[tgt_name] = res
            log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: LGB+XGB ENSEMBLE ON BEST THRESHOLD TARGETS
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 3: LGB+XGB ENSEMBLE ON THRESHOLD TARGETS (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        # Test ensemble on the three best 1h targets
        ensemble_configs = [("p45_top20", 0.45, 0.20), ("p35_top10", 0.35, 0.10),
                            ("p45_all", 0.45, None), ("p40_top10", 0.40, 0.10)]
        for tgt_name, tgt_col in [("up_12_0002", "target_up_12_0002"),
                                    ("up_12_0003", "target_up_12_0003"),
                                    ("up_12_0005", "target_up_12_0005")]:
            run_ensemble_wf(df, feat_1h, tgt_name, tgt_col, 12,
                           n_splits, test_size, purge_1h, ensemble_configs, f)
            log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: REGIME FILTERING ON THRESHOLD MODELS
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 4: REGIME FILTERING ON THRESHOLD MODELS (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        # Compute regime indicators for each test split
        # Use vol_mom96 (strongest predictor from v6)
        vol_mom96 = df["volume_momentum_96"].values if "volume_momentum_96" in df.columns else None

        if vol_mom96 is not None:
            tgt_name = "up_12_0003"
            tgt_col = "target_up_12_0003"
            base_rate = df[tgt_col].dropna().mean()
            log(f"  === {tgt_name} with vol_mom96 gating ===\n", f)

            # Run WF with regime gating
            for gate_name, gate_fn in [("no_gate", lambda v: np.ones(len(v), dtype=bool)),
                                       ("low_volmom", lambda v: v < np.nanmedian(v)),
                                       ("high_volmom", lambda v: v >= np.nanmedian(v))]:
                gated_bt = {cfg[0]: [] for cfg in THRESH_BT_CONFIGS[:6]}  # Top 6 configs
                gated_aucs = []

                for s in range(n_splits):
                    test_end = n_total - s * test_size
                    test_start = test_end - test_size
                    train_end = test_start - purge_1h

                    if train_end < 10000:
                        continue

                    data = wf_split_data(df, feat_1h, tgt_col, train_end, test_start, test_end, purge_1h)
                    if data is None:
                        continue

                    X_tr, y_tr, X_va, y_va, X_te, y_te, close_te = data
                    model = train_lgb(X_tr, y_tr, X_va, y_va)
                    p_test = model.predict_proba(X_te)[:, 1]
                    auc = auc_roc(y_te, p_test)

                    # Get vol_mom96 for test period
                    test_mask = (np.arange(n_total) >= test_start) & (np.arange(n_total) < test_end) & ~np.isnan(df[tgt_col].values)
                    vm96_test = vol_mom96[test_mask]

                    # Apply gate
                    gate_mask = gate_fn(vm96_test)

                    # Backtest with gated predictions (zero out non-gated)
                    p_gated = p_test.copy()
                    p_gated[~gate_mask] = 0.0

                    gated_aucs.append(auc)

                    for cfg_name, prob_thresh, top_pct in THRESH_BT_CONFIGS[:6]:
                        trades = backtest_threshold_signal(close_te, p_gated, 12, FEE_MAKER_RT,
                                                           prob_thresh, top_pct)
                        stats = bt_stats(trades, 12)
                        gated_bt[cfg_name].append(stats)

                log(f"  Gate: {gate_name}", f)
                for cfg_name, _, _ in THRESH_BT_CONFIGS[:6]:
                    splits = gated_bt[cfg_name]
                    with_trades = [s for s in splits if s["n"] > 0]
                    if not with_trades:
                        continue
                    pos = sum(1 for s in with_trades if s["net"] > 0)
                    n_s = len(with_trades)
                    avg_net = np.mean([s["net"] for s in with_trades])
                    med_n = int(np.median([s["n"] for s in with_trades]))
                    log(f"    {cfg_name:<15s} {pos:>3d}/{n_s:<3d}  avg={avg_net:>+7.2%}  med_n={med_n:>4d}", f)
                log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: MULTI-MODEL PORTFOLIO BACKTEST
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 5: MULTI-MODEL PORTFOLIO BACKTEST (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        # Simulate running 3 models in parallel on each WF split
        portfolio_models = [
            ("up_12_0002", "target_up_12_0002", 12, feat_1h, 0.45, 0.20, purge_1h),
            ("up_12_0003", "target_up_12_0003", 12, feat_1h, 0.35, 0.10, purge_1h),
            ("up_12_0005", "target_up_12_0005", 12, feat_1h, 0.45, None, purge_1h),
        ]

        # Also test 3h models if they worked well in Phase 1
        for tgt_name in ["up_36_0003", "up_36_0005"]:
            res = results_3h.get(tgt_name)
            if res:
                best_cfg = max(res["configs"].items(),
                               key=lambda x: x[1]["pos"] / max(x[1]["total"], 1)
                               if x[1]["total"] > 5 else 0)
                if best_cfg[1]["pos"] / max(best_cfg[1]["total"], 1) >= 0.7:
                    # Parse config
                    for cn, pt, tp in THRESH_BT_CONFIGS:
                        if cn == best_cfg[0]:
                            portfolio_models.append((tgt_name, f"target_{tgt_name}", 36,
                                                    feat_3h, pt, tp, purge_3h))
                            log(f"  Including {tgt_name} {best_cfg[0]} in portfolio", f)
                            break

        log(f"  Portfolio models: {[m[0] for m in portfolio_models]}\n", f)

        portfolio_results = []
        for s in range(n_splits):
            test_end = n_total - s * test_size
            test_start = test_end - test_size
            split_trades = []
            split_model_trades = {}

            for tgt_name, tgt_col, horizon, feat, prob_thresh, top_pct, purge in portfolio_models:
                train_end = test_start - purge

                if train_end < 10000:
                    continue

                data = wf_split_data(df, feat, tgt_col, train_end, test_start, test_end, purge)
                if data is None:
                    continue

                X_tr, y_tr, X_va, y_va, X_te, y_te, close_te = data
                model = train_lgb(X_tr, y_tr, X_va, y_va)
                p_test = model.predict_proba(X_te)[:, 1]

                trades = backtest_threshold_signal(close_te, p_test, horizon,
                                                    FEE_MAKER_RT, prob_thresh, top_pct)
                stats = bt_stats(trades, horizon)
                split_model_trades[tgt_name] = stats

                # Tag trades with model name and global idx for portfolio
                for t in trades:
                    t["model"] = tgt_name
                    t["horizon"] = horizon
                split_trades.extend(trades)

            # Compute portfolio stats
            if split_trades:
                # Sort by idx to simulate time-ordered execution
                split_trades.sort(key=lambda t: t["idx"])
                all_rets = [t["net_ret"] for t in split_trades]
                cum = np.cumprod(1 + np.array(all_rets))
                total_net = cum[-1] - 1
                wr = np.mean([r > 0 for r in all_rets])
                n_trades = len(split_trades)
            else:
                total_net = 0
                wr = 0
                n_trades = 0

            test_df = df.iloc[test_start:test_end]
            dr = get_date_range(test_df)

            log(f"  Split {s+1}: {dr}  portfolio: n={n_trades:>4d}  net={total_net:>+7.2%}  WR={wr:.1%}", f)
            for tgt_name in [m[0] for m in portfolio_models]:
                if tgt_name in split_model_trades:
                    st = split_model_trades[tgt_name]
                    log(f"    {tgt_name:<15s}: n={st['n']:>4d}  net={st['net']:>+7.2%}  kelly={st['kelly']:>+.3f}", f)

            portfolio_results.append({"n": n_trades, "net": total_net, "wr": wr,
                                       "model_trades": split_model_trades})

        # Portfolio summary
        pos = sum(1 for r in portfolio_results if r["net"] > 0)
        avg_net = np.mean([r["net"] for r in portfolio_results])
        avg_n = np.mean([r["n"] for r in portfolio_results])
        avg_wr = np.mean([r["wr"] for r in portfolio_results if r["n"] > 0])

        log(f"\n  PORTFOLIO SUMMARY:", f)
        log(f"  Positive splits: {pos}/{len(portfolio_results)}", f)
        log(f"  Avg net per split: {avg_net:>+.2%}", f)
        log(f"  Avg trades per split: {avg_n:.0f}", f)
        log(f"  Avg win rate: {avg_wr:.1%}", f)
        log(f"  Models: {[m[0] for m in portfolio_models]}", f)

        # ═══════════════════════════════════════════════════════════════
        # FINAL VERDICT
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  ITERATION 8 — FINAL VERDICT", f)
        log(f"{'#'*80}\n", f)

        log(f"  Key questions answered:", f)
        log(f"    1. Do 3h threshold targets provide independent signals?", f)
        log(f"    2. Can we trade the short side with down_12 targets?", f)
        log(f"    3. Does LGB+XGB ensemble improve threshold models?", f)
        log(f"    4. Does vol_mom96 regime gating help threshold models?", f)
        log(f"    5. Does a multi-model portfolio increase trade count?", f)
        log(f"\n  (See results above for answers)", f)

        log(f"\nResults saved to {RESULTS_FILE}", f)


if __name__ == "__main__":
    main()
