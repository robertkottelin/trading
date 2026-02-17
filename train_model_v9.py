"""
BTC ML Training — Iteration 9: PRODUCTION-GRADE PORTFOLIO
==========================================================
Building on v7b/v8 breakthroughs. Key experiments:

Phase 1: Enhanced feature selection (new v3 volatility features)
Phase 2: Take-profit/stop-loss exits (exit when target hit, not fixed hold)
Phase 3: Multi-seed ensemble (3 LGB seeds averaged)
Phase 4: Favorable risk-reward targets (new asymmetric targets)
Phase 5: Portfolio backtest — 1h + 3h models combined
Phase 6: Production metrics — equity curve, annual return, max DD, Kelly sizing

Usage:
  python train_model_v9.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = "btc_data"
FEATURES_FILE = os.path.join(OUTPUT_DIR, "btc_features_5m.parquet")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "training_results_v9.txt")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

FEE_MAKER_RT = 0.0004  # 0.04% round-trip


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


# ─── BACKTEST FUNCTIONS ──────────────────────────────────────────────────

def backtest_threshold(close, y_prob, horizon, fee_rt, prob_threshold, top_pct=None):
    """Long-only backtest: enter when P(up>threshold) > prob_threshold, hold for horizon."""
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


def backtest_take_profit(close_arr, y_prob, horizon, fee_rt, prob_threshold,
                          take_profit, stop_loss=None, top_pct=None):
    """
    Take-profit/stop-loss backtest.
    Enter long when P > prob_threshold. Exit when:
      - Price hits take_profit (e.g., +0.3%)
      - Price hits stop_loss (e.g., -0.15%)
      - Max hold time reached (horizon candles)
    """
    n = len(close_arr)
    all_entries = []
    i = 0
    while i < n:
        prob = y_prob[i]
        if prob > prob_threshold and i + 1 < n:
            all_entries.append({"idx": i, "prob": prob, "entry_price": close_arr[i]})
        i += 1  # Check every candle, not every horizon

    if not all_entries:
        return []

    # Top-pct filtering
    if top_pct is not None and len(all_entries) > 5:
        probs = np.array([e["prob"] for e in all_entries])
        threshold = np.quantile(probs, 1.0 - top_pct)
        all_entries = [e for e in all_entries if e["prob"] >= threshold]

    # Non-overlapping execution
    trades = []
    next_available = 0
    for entry in all_entries:
        idx = entry["idx"]
        if idx < next_available:
            continue

        entry_price = entry["entry_price"]
        exit_idx = min(idx + horizon, n - 1)
        exit_price = close_arr[exit_idx]
        exit_reason = "timeout"

        # Scan forward for TP/SL
        for j in range(idx + 1, min(idx + horizon + 1, n)):
            price = close_arr[j]
            ret = (price - entry_price) / entry_price
            if take_profit is not None and ret >= take_profit:
                exit_price = price
                exit_idx = j
                exit_reason = "tp"
                break
            if stop_loss is not None and ret <= -stop_loss:
                exit_price = price
                exit_idx = j
                exit_reason = "sl"
                break

        raw_ret = (exit_price - entry_price) / entry_price
        net_ret = raw_ret - fee_rt
        trades.append({"idx": idx, "prob": entry["prob"], "raw_ret": raw_ret,
                       "net_ret": net_ret, "exit_reason": exit_reason,
                       "hold_candles": exit_idx - idx})
        next_available = exit_idx + 1

    return trades


def bt_stats(trades, horizon):
    if not trades:
        return {"n": 0, "net": 0, "sharpe": 0, "wr": 0, "pf": 0,
                "max_dd": 0, "kelly": 0, "avg_hold": 0}
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
    avg_hold = np.mean([t.get("hold_candles", horizon) for t in trades])
    return {"n": len(rets), "net": total, "sharpe": sharpe, "wr": wr, "pf": pf,
            "max_dd": max_dd, "kelly": kelly, "avg_hold": avg_hold}


# ─── TRAINING ────────────────────────────────────────────────────────────

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


def train_lgb(X_tr, y_tr, X_va, y_va, seed=42):
    params = dict(LGB_PARAMS)
    params["random_state"] = seed
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return m


def train_ensemble(X_tr, y_tr, X_va, y_va, n_seeds=3):
    """Train multi-seed ensemble and return averaged predict function."""
    seeds = [42, 123, 7][:n_seeds]
    models = []
    for seed in seeds:
        m = train_lgb(X_tr, y_tr, X_va, y_va, seed=seed)
        models.append(m)
    return models


def predict_ensemble(models, X):
    """Average predictions from multiple models."""
    probs = [m.predict_proba(X)[:, 1] for m in models]
    return np.mean(probs, axis=0)


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


# ─── CONFIGS ─────────────────────────────────────────────────────────────

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
]

TP_CONFIGS = [
    # (name, prob_thresh, top_pct, take_profit, stop_loss)
    ("tp3_sl15_p40",  0.40, None, 0.003, 0.0015),
    ("tp3_sl15_p45",  0.45, None, 0.003, 0.0015),
    ("tp3_none_p40",  0.40, None, 0.003, None),
    ("tp5_sl25_p40",  0.40, None, 0.005, 0.0025),
    ("tp5_sl25_p45",  0.45, None, 0.005, 0.0025),
    ("tp2_sl10_p45",  0.45, None, 0.002, 0.0010),
    ("tp3_sl15_p40t20", 0.40, 0.20, 0.003, 0.0015),
    ("tp3_sl15_p45t20", 0.45, 0.20, 0.003, 0.0015),
]


def run_wf_threshold(df, feat_cols, tgt_name, tgt_col, horizon, n_splits,
                      test_size, purge, f, use_ensemble=False):
    """Run walk-forward with threshold backtesting."""
    if tgt_col not in df.columns:
        log(f"  {tgt_col} not found — skipping", f)
        return None

    n_total = len(df)
    base_rate = df[tgt_col].dropna().mean()
    ens_label = " (ensemble)" if use_ensemble else ""
    log(f"  === {tgt_name}{ens_label} (base_rate={base_rate:.3f}) ===\n", f)

    all_split_bt = {cfg[0]: [] for cfg in THRESH_BT_CONFIGS}
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
        if use_ensemble:
            models = train_ensemble(X_tr, y_tr, X_va, y_va, n_seeds=3)
            p_test = predict_ensemble(models, X_te)
            iters = models[0].best_iteration_
        else:
            model = train_lgb(X_tr, y_tr, X_va, y_va)
            p_test = model.predict_proba(X_te)[:, 1]
            iters = model.best_iteration_
        elapsed = time.time() - t0

        auc = auc_roc(y_te, p_test)
        all_aucs.append(auc)

        test_df = df.iloc[np.where((np.arange(n_total) >= test_start) &
                                    (np.arange(n_total) < test_end) &
                                    ~np.isnan(df[tgt_col].values))[0]]
        dr = get_date_range(test_df)

        log(f"    Split {s+1}: {dr}  AUC={auc:.4f}  iter={iters}  time={elapsed:.0f}s  "
            f"p_mean={p_test.mean():.3f}  p_max={p_test.max():.3f}", f)

        for cfg_name, prob_thresh, top_pct in THRESH_BT_CONFIGS:
            trades = backtest_threshold(close_te, p_test, horizon, FEE_MAKER_RT,
                                         prob_thresh, top_pct)
            stats = bt_stats(trades, horizon)
            all_split_bt[cfg_name].append(stats)

        # Print key configs
        for cfg_name in ["p40_top20", "p40_top10", "p45_top20", "p45_top10"]:
            s_bt = all_split_bt[cfg_name][-1]
            if s_bt["n"] > 0:
                log(f"      {cfg_name}: n={s_bt['n']:>4d}  net={s_bt['net']:>+7.2%}  "
                    f"WR={s_bt['wr']:.1%}  kelly={s_bt['kelly']:>+.3f}", f)

    if not all_aucs:
        return None

    auc_mean = np.mean(all_aucs)
    auc_std = np.std(all_aucs)
    log(f"\n  {tgt_name} WF SUMMARY (AUC={auc_mean:.4f}±{auc_std:.4f}):", f)
    log(f"  {'Config':<18s} {'Pos/N':>8s} {'Avg Net':>10s} {'Avg WR':>8s} {'Med N':>7s} {'Kelly':>8s}", f)

    results = {}
    for cfg_name, _, _ in THRESH_BT_CONFIGS:
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
                             "avg_wr": avg_wr, "med_n": med_n, "kelly": avg_kelly,
                             "split_stats": all_split_bt[cfg_name]}

    return {"auc_mean": auc_mean, "auc_std": auc_std, "configs": results,
            "split_aucs": all_aucs}


def run_wf_take_profit(df, feat_cols, tgt_name, tgt_col, horizon, n_splits,
                        test_size, purge, f):
    """Run walk-forward with take-profit/stop-loss backtesting."""
    if tgt_col not in df.columns:
        return None

    n_total = len(df)
    base_rate = df[tgt_col].dropna().mean()
    log(f"  === {tgt_name} TAKE-PROFIT (base_rate={base_rate:.3f}) ===\n", f)

    all_tp_bt = {cfg[0]: [] for cfg in TP_CONFIGS}
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

        model = train_lgb(X_tr, y_tr, X_va, y_va)
        p_test = model.predict_proba(X_te)[:, 1]
        auc = auc_roc(y_te, p_test)
        all_aucs.append(auc)

        test_df = df.iloc[np.where((np.arange(n_total) >= test_start) &
                                    (np.arange(n_total) < test_end) &
                                    ~np.isnan(df[tgt_col].values))[0]]
        dr = get_date_range(test_df)
        log(f"    Split {s+1}: {dr}  AUC={auc:.4f}", f)

        for cfg_name, prob_thresh, top_pct, tp, sl in TP_CONFIGS:
            trades = backtest_take_profit(close_te, p_test, horizon, FEE_MAKER_RT,
                                           prob_thresh, tp, sl, top_pct)
            stats = bt_stats(trades, horizon)
            all_tp_bt[cfg_name].append(stats)
            if stats["n"] > 0:
                tp_count = sum(1 for t in trades if t.get("exit_reason") == "tp")
                sl_count = sum(1 for t in trades if t.get("exit_reason") == "sl")
                to_count = sum(1 for t in trades if t.get("exit_reason") == "timeout")
                log(f"      {cfg_name}: n={stats['n']:>4d}  net={stats['net']:>+7.2%}  "
                    f"WR={stats['wr']:.1%}  kelly={stats['kelly']:>+.3f}  "
                    f"tp/sl/to={tp_count}/{sl_count}/{to_count}  "
                    f"avg_hold={stats['avg_hold']:.1f}", f)

    if not all_aucs:
        return None

    log(f"\n  {tgt_name} TP SUMMARY (AUC={np.mean(all_aucs):.4f}±{np.std(all_aucs):.4f}):", f)
    log(f"  {'Config':<22s} {'Pos/N':>8s} {'Avg Net':>10s} {'Med N':>7s} {'Kelly':>8s} {'AvgHold':>8s}", f)

    results = {}
    for cfg_name, _, _, _, _ in TP_CONFIGS:
        splits = all_tp_bt[cfg_name]
        with_trades = [s for s in splits if s["n"] > 0]
        if not with_trades:
            continue
        pos = sum(1 for s in with_trades if s["net"] > 0)
        n_s = len(with_trades)
        avg_net = np.mean([s["net"] for s in with_trades])
        med_n = int(np.median([s["n"] for s in with_trades]))
        avg_kelly = np.mean([s["kelly"] for s in with_trades])
        avg_hold = np.mean([s["avg_hold"] for s in with_trades])
        log(f"  {cfg_name:<22s} {pos:>3d}/{n_s:<3d}  {avg_net:>+8.2%}  {med_n:>6d}  "
            f"{avg_kelly:>+.3f}  {avg_hold:>7.1f}", f)
        results[cfg_name] = {"pos": pos, "total": n_s, "avg_net": avg_net,
                             "med_n": med_n, "kelly": avg_kelly,
                             "split_stats": all_tp_bt[cfg_name]}

    return results


def run_portfolio_backtest(df, model_configs, n_splits, test_size, f):
    """
    Portfolio backtest: run multiple models in parallel, combine trades.
    Each model_config: (name, tgt_col, horizon, feat_cols, prob_thresh, top_pct, purge)
    """
    n_total = len(df)
    log(f"  Portfolio models: {[m[0] for m in model_configs]}\n", f)

    portfolio_results = []
    for s in range(n_splits):
        test_end = n_total - s * test_size
        test_start = test_end - test_size
        split_trades = []
        split_model_stats = {}

        for name, tgt_col, horizon, feat, prob_thresh, top_pct, purge in model_configs:
            train_end = test_start - purge

            if train_end < 10000:
                continue

            data = wf_split_data(df, feat, tgt_col, train_end, test_start, test_end, purge)
            if data is None:
                continue

            X_tr, y_tr, X_va, y_va, X_te, y_te, close_te = data
            models = train_ensemble(X_tr, y_tr, X_va, y_va, n_seeds=3)
            p_test = predict_ensemble(models, X_te)

            trades = backtest_threshold(close_te, p_test, horizon, FEE_MAKER_RT,
                                         prob_thresh, top_pct)
            stats = bt_stats(trades, horizon)
            split_model_stats[name] = stats

            for t in trades:
                t["model"] = name
                t["horizon"] = horizon
            split_trades.extend(trades)

        # Portfolio equity curve
        if split_trades:
            split_trades.sort(key=lambda t: t["idx"])
            all_rets = np.array([t["net_ret"] for t in split_trades])
            cum = np.cumprod(1 + all_rets)
            total_net = cum[-1] - 1
            wr = (all_rets > 0).mean()
            cummax = np.maximum.accumulate(cum)
            max_dd = ((cum - cummax) / cummax).min()
            n_trades = len(split_trades)
            # Per-model trade counts
            model_counts = {}
            for t in split_trades:
                model_counts[t["model"]] = model_counts.get(t["model"], 0) + 1
        else:
            total_net = 0
            wr = 0
            max_dd = 0
            n_trades = 0
            model_counts = {}

        test_df = df.iloc[test_start:test_end]
        dr = get_date_range(test_df)

        log(f"  Split {s+1}: {dr}  portfolio: n={n_trades:>4d}  net={total_net:>+7.2%}  "
            f"WR={wr:.1%}  maxDD={max_dd:.2%}", f)
        for name in [m[0] for m in model_configs]:
            if name in split_model_stats:
                st = split_model_stats[name]
                log(f"    {name:<22s}: n={st['n']:>4d}  net={st['net']:>+7.2%}  "
                    f"kelly={st['kelly']:>+.3f}", f)

        portfolio_results.append({
            "n": n_trades, "net": total_net, "wr": wr, "max_dd": max_dd,
            "model_stats": split_model_stats, "model_counts": model_counts
        })

    return portfolio_results


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols")

    all_feat_cols = get_feature_cols(df)
    n_total = len(df)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        log(f"BTC ML Training — Iteration 9: PRODUCTION-GRADE PORTFOLIO", f)
        log(f"{'='*80}", f)
        log(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", f)
        log(f"Data: {len(df):,} rows x {len(df.columns)} cols  Features: {len(all_feat_cols)}", f)

        # Check for new features
        new_feats = [c for c in all_feat_cols if any(c.startswith(p) for p in
                     ["parkinson_", "garman_klass_", "ewma_vol_", "intraday_vol_",
                      "vol_term_", "park_vs_rv_", "atr_", "hit_rate_",
                      "upside_vol_", "downside_vol_", "vol_skew_",
                      "ret_autocorr_", "trail_dd_", "trail_runup_", "hurst_"])]
        log(f"  New v3 features: {len(new_feats)}", f)
        log(f"  Total features: {len(all_feat_cols)}", f)

        # ─── Feature selection ──────────────────────────────────────────
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 0: FEATURE SELECTION (with v3 features)", f)
        log(f"{'#'*80}\n", f)

        feat_cache = {}
        for tgt_name, tgt_col in [("up_12_0002", "target_up_12_0002"),
                                    ("up_12_0003", "target_up_12_0003"),
                                    ("up_12_0005", "target_up_12_0005"),
                                    ("up_36_0002", "target_up_36_0002"),
                                    ("up_36_0003", "target_up_36_0003")]:
            if tgt_col not in df.columns:
                log(f"  {tgt_col} not found", f)
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
            log(f"  {tgt_name} top-15:", f)
            for idx, row in imp.head(15).iterrows():
                is_new = "***" if row['feat'] in new_feats else ""
                log(f"    {row['imp']:>5.0f}  {row['feat']} {is_new}", f)
            log("", f)

            # Check how many new features in top-80
            top80 = set(imp.head(80)["feat"].tolist())
            new_in_top80 = len(top80 & set(new_feats))
            log(f"  New v3 features in top-80: {new_in_top80}/{len(new_feats)}\n", f)

        feat_1h = feat_cache.get("up_12_0003", all_feat_cols[:80])
        feat_3h = feat_cache.get("up_36_0003", all_feat_cols[:80])

        n_splits = 10
        test_size = int(n_total * 0.05)
        purge_1h = 2 * 12
        purge_3h = 2 * 36

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: CORE THRESHOLD MODELS (v3 features comparison)
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 1: CORE THRESHOLD MODELS (v3 features, 10-split WF)", f)
        log(f"{'#'*80}\n", f)

        core_results = {}
        # 1h models
        for tgt_name, tgt_col in [("up_12_0002", "target_up_12_0002"),
                                    ("up_12_0003", "target_up_12_0003"),
                                    ("up_12_0005", "target_up_12_0005")]:
            feat = feat_cache.get(tgt_name, feat_1h)
            res = run_wf_threshold(df, feat, tgt_name, tgt_col, 12,
                                    n_splits, test_size, purge_1h, f)
            if res:
                core_results[tgt_name] = res
            log("", f)

        # 3h models
        for tgt_name, tgt_col in [("up_36_0002", "target_up_36_0002"),
                                    ("up_36_0003", "target_up_36_0003")]:
            feat = feat_cache.get(tgt_name, feat_3h)
            res = run_wf_threshold(df, feat, tgt_name, tgt_col, 36,
                                    n_splits, test_size, purge_3h, f)
            if res:
                core_results[tgt_name] = res
            log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: TAKE-PROFIT/STOP-LOSS EXITS
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 2: TAKE-PROFIT/STOP-LOSS EXITS (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        tp_results = {}
        for tgt_name, tgt_col, horizon, feat in [
            ("up_12_0003", "target_up_12_0003", 12, feat_cache.get("up_12_0003", feat_1h)),
            ("up_12_0005", "target_up_12_0005", 12, feat_cache.get("up_12_0005", feat_1h)),
        ]:
            res = run_wf_take_profit(df, feat, tgt_name, tgt_col, horizon,
                                      n_splits, test_size, purge_1h, f)
            if res:
                tp_results[tgt_name] = res
            log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: MULTI-SEED ENSEMBLE ON BEST TARGETS
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 3: MULTI-SEED ENSEMBLE (3 seeds, 10-split WF)", f)
        log(f"{'#'*80}\n", f)

        ens_results = {}
        for tgt_name, tgt_col, horizon, feat, purge in [
            ("up_12_0002", "target_up_12_0002", 12, feat_cache.get("up_12_0002", feat_1h), purge_1h),
            ("up_12_0003", "target_up_12_0003", 12, feat_cache.get("up_12_0003", feat_1h), purge_1h),
        ]:
            res = run_wf_threshold(df, feat, tgt_name, tgt_col, horizon,
                                    n_splits, test_size, purge, f, use_ensemble=True)
            if res:
                ens_results[tgt_name] = res
            log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: FAVORABLE RISK-REWARD TARGETS
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 4: FAVORABLE RISK-REWARD TARGETS (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        fav_results = {}
        for tgt_name, tgt_col, horizon, purge in [
            ("favorable_12_0003", "target_favorable_12_0003", 12, purge_1h),
            ("favorable_12_0005", "target_favorable_12_0005", 12, purge_1h),
            ("favorable_36_0003", "target_favorable_36_0003", 36, purge_3h),
        ]:
            feat = feat_cache.get("up_12_0003", feat_1h) if horizon == 12 else feat_cache.get("up_36_0003", feat_3h)
            res = run_wf_threshold(df, feat, tgt_name, tgt_col, horizon,
                                    n_splits, test_size, purge, f)
            if res:
                fav_results[tgt_name] = res
            log("", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: MULTI-MODEL PORTFOLIO BACKTEST
        # ═══════════════════════════════════════════════════════════════
        log(f"{'#'*80}", f)
        log(f"  PHASE 5: MULTI-MODEL PORTFOLIO BACKTEST (10-split WF)", f)
        log(f"{'#'*80}\n", f)

        # Build portfolio from best-performing models
        # Select configs that had >= 8/10 positive splits
        portfolio_models = []

        # Always include the proven 1h models
        portfolio_models.append(("up_12_0002_p45t20", "target_up_12_0002", 12,
                                  feat_cache.get("up_12_0002", feat_1h), 0.45, 0.20, purge_1h))
        portfolio_models.append(("up_12_0003_p35t10", "target_up_12_0003", 12,
                                  feat_cache.get("up_12_0003", feat_1h), 0.35, 0.10, purge_1h))
        portfolio_models.append(("up_12_0005_p45all", "target_up_12_0005", 12,
                                  feat_cache.get("up_12_0005", feat_1h), 0.45, None, purge_1h))

        # Add 3h models if they performed well
        for tgt_name in ["up_36_0002", "up_36_0003"]:
            if tgt_name in core_results:
                res = core_results[tgt_name]
                best_cfg = None
                best_ratio = 0
                for cfg_name, cfg_data in res["configs"].items():
                    if cfg_data["total"] >= 7:
                        ratio = cfg_data["pos"] / cfg_data["total"]
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_cfg = (cfg_name, cfg_data)
                if best_cfg and best_ratio >= 0.7:
                    cfg_name = best_cfg[0]
                    # Parse config name to get params
                    for cn, pt, tp in THRESH_BT_CONFIGS:
                        if cn == cfg_name:
                            portfolio_models.append(
                                (f"{tgt_name}_{cfg_name}", f"target_{tgt_name}", 36,
                                 feat_cache.get(tgt_name, feat_3h), pt, tp, purge_3h))
                            log(f"  Including {tgt_name} {cfg_name} in portfolio "
                                f"({best_cfg[1]['pos']}/{best_cfg[1]['total']} positive)", f)
                            break

        # Add favorable targets if they work
        for tgt_name in ["favorable_12_0003", "favorable_12_0005"]:
            if tgt_name in fav_results:
                res = fav_results[tgt_name]
                for cfg_name, cfg_data in res["configs"].items():
                    if cfg_data["total"] >= 7 and cfg_data["pos"] / cfg_data["total"] >= 0.8:
                        for cn, pt, tp in THRESH_BT_CONFIGS:
                            if cn == cfg_name:
                                portfolio_models.append(
                                    (f"{tgt_name}_{cfg_name}", f"target_{tgt_name}", 12,
                                     feat_cache.get("up_12_0003", feat_1h), pt, tp, purge_1h))
                                log(f"  Including {tgt_name} {cfg_name} in portfolio", f)
                                break
                        break  # Just use the best config

        log(f"\n  Total portfolio models: {len(portfolio_models)}", f)
        portfolio_res = run_portfolio_backtest(df, portfolio_models, n_splits, test_size, f)

        # Portfolio summary
        pos = sum(1 for r in portfolio_res if r["net"] > 0)
        valid = [r for r in portfolio_res if r["n"] > 0]
        avg_net = np.mean([r["net"] for r in portfolio_res]) if portfolio_res else 0
        avg_n = np.mean([r["n"] for r in portfolio_res]) if portfolio_res else 0
        avg_wr = np.mean([r["wr"] for r in valid]) if valid else 0
        avg_dd = np.mean([r["max_dd"] for r in valid]) if valid else 0

        log(f"\n  PORTFOLIO SUMMARY:", f)
        log(f"  Positive splits: {pos}/{len(portfolio_res)}", f)
        log(f"  Avg net per split: {avg_net:>+.2%}", f)
        log(f"  Avg trades per split: {avg_n:.0f}", f)
        log(f"  Avg win rate: {avg_wr:.1%}", f)
        log(f"  Avg max drawdown: {avg_dd:.2%}", f)

        # Annualized performance estimate
        if portfolio_res:
            # Each split is ~155 days
            days_per_split = 155
            annual_factor = 365.25 / days_per_split
            cumulative = 1.0
            for r in portfolio_res:
                cumulative *= (1 + r["net"])
            total_return = cumulative - 1
            n_splits_actual = len(portfolio_res)
            years = n_splits_actual * days_per_split / 365.25
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

            log(f"\n  Compounded return over {n_splits_actual} splits: {total_return:>+.2%}", f)
            log(f"  Annualized return estimate: {annual_return:>+.2%}", f)
            log(f"  Years covered: {years:.1f}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 6: COMPARISON TABLE
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 6: COMPREHENSIVE COMPARISON", f)
        log(f"{'#'*80}\n", f)

        log(f"  === Single Model Results (best config each) ===", f)
        log(f"  {'Target':<22s} {'AUC':>8s} {'Config':<16s} {'Pos/N':>8s} {'Avg Net':>10s} {'Kelly':>8s}", f)

        for tgt_name, res in core_results.items():
            best_cfg = None
            best_score = -999
            for cfg_name, cfg_data in res["configs"].items():
                if cfg_data["total"] >= 7:
                    score = (cfg_data["pos"] / cfg_data["total"]) * 10 + cfg_data["avg_net"] * 100
                    if score > best_score:
                        best_score = score
                        best_cfg = (cfg_name, cfg_data)
            if best_cfg:
                cn, cd = best_cfg
                log(f"  {tgt_name:<22s} {res['auc_mean']:>7.4f}  {cn:<16s} "
                    f"{cd['pos']:>3d}/{cd['total']:<3d}  {cd['avg_net']:>+8.2%}  {cd['kelly']:>+.3f}", f)

        log(f"\n  === Ensemble vs Single (up_12_0002, up_12_0003) ===", f)
        for tgt_name in ["up_12_0002", "up_12_0003"]:
            if tgt_name in core_results and tgt_name in ens_results:
                single = core_results[tgt_name]
                ens = ens_results[tgt_name]
                log(f"  {tgt_name}:", f)
                log(f"    Single AUC: {single['auc_mean']:.4f}±{single['auc_std']:.4f}", f)
                log(f"    Ensemble AUC: {ens['auc_mean']:.4f}±{ens['auc_std']:.4f}", f)
                # Compare best configs
                for cfg_name in ["p45_top20", "p40_top10", "p35_top10"]:
                    s_data = single["configs"].get(cfg_name, {})
                    e_data = ens["configs"].get(cfg_name, {})
                    if s_data and e_data:
                        log(f"    {cfg_name}: single {s_data.get('pos',0)}/{s_data.get('total',0)} "
                            f"avg={s_data.get('avg_net',0):+.2%} | "
                            f"ensemble {e_data.get('pos',0)}/{e_data.get('total',0)} "
                            f"avg={e_data.get('avg_net',0):+.2%}", f)

        # ═══════════════════════════════════════════════════════════════
        # FINAL VERDICT
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  ITERATION 9 — FINAL VERDICT", f)
        log(f"{'#'*80}\n", f)

        log(f"  Key questions answered:", f)
        log(f"    1. Do v3 volatility features improve threshold models?", f)
        log(f"    2. Do take-profit/stop-loss exits improve returns?", f)
        log(f"    3. Does multi-seed ensemble help threshold models?", f)
        log(f"    4. Do favorable risk-reward targets outperform simple up targets?", f)
        log(f"    5. How does the combined portfolio perform?", f)

        # Production readiness assessment
        log(f"\n  === PRODUCTION READINESS ===", f)
        production_ready = True
        for tgt_name, res in core_results.items():
            for cfg_name, cfg_data in res["configs"].items():
                if cfg_data["total"] >= 8 and cfg_data["pos"] >= 8:
                    log(f"  [PASS] {tgt_name} {cfg_name}: {cfg_data['pos']}/{cfg_data['total']} "
                        f"positive, avg={cfg_data['avg_net']:+.2%}", f)

        if portfolio_res:
            if pos >= 8:
                log(f"  [PASS] Portfolio: {pos}/{len(portfolio_res)} positive, "
                    f"avg={avg_net:+.2%}", f)
            else:
                log(f"  [NEEDS WORK] Portfolio: {pos}/{len(portfolio_res)} positive", f)
                production_ready = False

        if production_ready:
            log(f"\n  STATUS: PRODUCTION CANDIDATE", f)
        else:
            log(f"\n  STATUS: NEEDS FURTHER IMPROVEMENT", f)

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
