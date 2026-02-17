"""
BTC LightGBM Training — Iteration 6
=====================================
Focus: Validate Iter 5 findings, close remaining gaps, improve robustness.

Key improvements over Iter 5:
  1. 10-split walk-forward (vs 6) for more statistical confidence
  2. Regime analysis: WHY does certain splits fail? Correlate with vol/trend/calendar
  3. Regime-gated walk-forward: skip trading in historically weak regimes
  4. Multi-horizon consensus walk-forward validation (untested in Iter 5)
  5. Feature stability analysis: which features are consistently important?
  6. Stable-features-only WF: does reducing to stable features improve robustness?
  7. Dynamic confidence thresholds: rolling percentile-based margins

Usage:
  python train_model_v6.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import time
import warnings
from datetime import datetime, timezone
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = "btc_data"
FEATURES_FILE = os.path.join(OUTPUT_DIR, "btc_features_5m.parquet")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "training_results_v6.txt")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

FEE_MAKER_RT = 0.0004  # 0.04% round-trip (maker)
FEE_TAKER_RT = 0.0008  # 0.08% round-trip (taker)


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


def get_feature_cols(df):
    exclude = {"open_time_ms", "timestamp"}
    target_cols = {c for c in df.columns if c.startswith("target_")}
    return sorted(c for c in df.columns if c not in exclude and c not in target_cols)


def backtest_lo(close, y_prob, horizon, fee_rt, margin=0.04, top_pct=None):
    """Long-only backtest with non-overlapping trades. Returns list of trade dicts."""
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
                "max_dd": 0, "kelly": 0, "calmar": 0, "avg_ret": 0}
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
    calmar = total / abs(max_dd) if abs(max_dd) > 1e-8 else 0
    return {"n": len(rets), "net": total, "sharpe": sharpe, "wr": wr, "pf": pf,
            "max_dd": max_dd, "kelly": kelly, "calmar": calmar, "avg_ret": mu}


def train_lgb(X_tr, y_tr, X_va, y_va, params):
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return m


def calibrate_iso(y_val, p_val, p_test):
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(p_val, y_val)
    return ir.transform(p_test)


BASE_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "boosting_type": "gbdt", "n_estimators": 5000,
    "learning_rate": 0.005, "num_leaves": 24, "max_depth": 5,
    "min_child_samples": 500, "subsample": 0.5, "subsample_freq": 1,
    "colsample_bytree": 0.3, "colsample_bynode": 0.5,
    "reg_alpha": 5.0, "reg_lambda": 20.0,
    "feature_fraction_bynode": 0.5, "path_smooth": 10.0,
    "random_state": 42, "verbose": -1, "n_jobs": -1,
}

# Backtest configs to test in walk-forward
BT_CONFIGS = [
    ("m02_top30", 0.02, 0.30),
    ("m04_top30", 0.04, 0.30),
    ("m04_top20", 0.04, 0.20),
    ("m06_top20", 0.06, 0.20),
    ("m04_top10", 0.04, 0.10),
    ("m06_top10", 0.06, 0.10),
    ("m02_all", 0.02, None),
    ("m04_all", 0.04, None),
]


def get_date_range(df_slice, col="open_time_ms"):
    dates = pd.to_datetime(df_slice[col], unit="ms", utc=True)
    return f"{dates.iloc[0].strftime('%Y-%m-%d')}/{dates.iloc[-1].strftime('%Y-%m-%d')}"


def get_regime_stats(df_slice):
    """Extract regime summary statistics for a data slice."""
    stats = {}
    for col in ["realized_vol_24", "vol_regime_rank", "return_288",
                 "trend_strength_48", "vol_price_corr_24", "volume_momentum_96"]:
        if col in df_slice.columns:
            vals = df_slice[col].dropna()
            if len(vals) > 0:
                stats[col] = {"mean": vals.mean(), "std": vals.std(),
                              "median": vals.median()}
    return stats


def get_feature_importance(model, feat_cols):
    imp = pd.DataFrame({"feat": feat_cols, "imp": model.feature_importances_})
    return imp.sort_values("imp", ascending=False)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols")

    all_feat_cols = get_feature_cols(df)
    n_total = len(df)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        log(f"BTC LightGBM Training — Iteration 6", f)
        log(f"{'='*80}", f)
        log(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", f)
        log(f"Data: {len(df):,} rows x {len(df.columns)} cols  Features: {len(all_feat_cols)}", f)
        log(f"\nKey changes from Iter 5:", f)
        log(f"  1. 10-split WF (vs 6) for more confidence", f)
        log(f"  2. Regime analysis per WF split", f)
        log(f"  3. Regime-gated walk-forward trading", f)
        log(f"  4. Multi-horizon consensus walk-forward", f)
        log(f"  5. Feature stability analysis", f)
        log(f"  6. Stable-features-only WF comparison", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: FEATURE IMPORTANCE (quick train for feature selection)
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 0: FEATURE IMPORTANCE BASELINE", f)
        log(f"{'#'*80}", f)

        # Train a quick model for feature importance
        horizon_feats = {}  # will store top features per horizon
        for horizon, h_label in [(12, "1h"), (36, "3h"), (48, "4h")]:
            target = f"target_direction_{horizon}"
            tr_end = int(n_total * 0.70)
            va_end = int(n_total * 0.85)
            y_tr = df.iloc[:tr_end][target].dropna()
            y_va = df.iloc[tr_end:va_end][target].dropna()
            X_tr = df.iloc[:tr_end].loc[y_tr.index, all_feat_cols]
            X_va = df.iloc[tr_end:va_end].loc[y_va.index, all_feat_cols]

            m0 = train_lgb(X_tr, y_tr, X_va, y_va, BASE_PARAMS)
            imp = get_feature_importance(m0, all_feat_cols)
            top80 = imp.head(80)["feat"].tolist()
            horizon_feats[h_label] = top80

            log(f"\n  {h_label} top-15 features:", f)
            for rank, (_, r) in enumerate(imp.head(15).iterrows(), 1):
                log(f"    {rank:>2d}. {r['imp']:5.0f}  {r['feat']}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: 10-SPLIT WALK-FORWARD (1h + 3h)
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 1: 10-SPLIT WALK-FORWARD VALIDATION", f)
        log(f"{'#'*80}", f)

        n_splits = 10
        test_frac = 0.05  # 5% per split

        wf_all_results = {}  # {h_label: [split_results]}

        for horizon, h_label in [(12, "1h"), (36, "3h")]:
            target = f"target_direction_{horizon}"
            purge_gap = horizon * 2
            feat_cols = horizon_feats[h_label]
            test_size = int(n_total * test_frac)

            log(f"\n{'='*80}", f)
            log(f"  {h_label} WALK-FORWARD ({n_splits} splits, top-80 features)", f)
            log(f"  Test size: {test_size:,} candles per split (~{test_size*5/60/24:.0f} days)", f)
            log(f"{'='*80}", f)

            split_results = []

            for i in range(n_splits):
                te_end = n_total - i * test_size
                te_start = te_end - test_size
                if te_start < 0:
                    break
                tr_end_i = te_start - purge_gap
                if tr_end_i < test_size * 2:
                    break

                df_te = df.iloc[te_start:te_end]
                df_tr = df.iloc[:tr_end_i]

                y_tr_i = df_tr[target].dropna()
                y_te_i = df_te[target].dropna()
                if len(y_tr_i) < 5000 or len(y_te_i) < 200:
                    continue

                X_tr_full = df_tr.loc[y_tr_i.index, feat_cols]
                X_te_i = df_te.loc[y_te_i.index, feat_cols]

                val_cut = int(len(X_tr_full) * 0.85)
                X_tr_s = X_tr_full.iloc[:val_cut]
                y_tr_s = y_tr_i.iloc[:val_cut]
                X_va_s = X_tr_full.iloc[val_cut:]
                y_va_s = y_tr_i.iloc[val_cut:]

                t0 = time.time()
                model = train_lgb(X_tr_s, y_tr_s, X_va_s, y_va_s, BASE_PARAMS)
                elapsed = time.time() - t0

                y_prob = model.predict_proba(X_te_i)[:, 1]
                raw_auc = auc_roc(y_te_i.values, y_prob)

                # Calibrated predictions
                y_prob_val = model.predict_proba(X_va_s)[:, 1]
                y_prob_cal = calibrate_iso(y_va_s.values, y_prob_val, y_prob)
                cal_auc = auc_roc(y_te_i.values, y_prob_cal)

                close_te = df_te.loc[y_te_i.index, "close"].values
                date_str = get_date_range(df_te)

                # Regime stats for this test period
                regime = get_regime_stats(df_te)

                # Backtest all configs (raw + calibrated)
                bt_res = {}
                for cname, margin, top_pct in BT_CONFIGS:
                    # Raw predictions
                    trades = backtest_lo(close_te, y_prob, horizon, FEE_MAKER_RT,
                                         margin=margin, top_pct=top_pct)
                    stats = bt_stats(trades, horizon)
                    bt_res[f"raw_{cname}"] = stats

                    # Calibrated predictions
                    trades_cal = backtest_lo(close_te, y_prob_cal, horizon, FEE_MAKER_RT,
                                             margin=margin, top_pct=top_pct)
                    stats_cal = bt_stats(trades_cal, horizon)
                    bt_res[f"cal_{cname}"] = stats_cal

                split_result = {
                    "split": i + 1, "raw_auc": raw_auc, "cal_auc": cal_auc,
                    "bt": bt_res, "regime": regime, "date_str": date_str,
                    "train_size": len(X_tr_s), "test_size": len(y_te_i),
                    "best_iter": model.best_iteration_, "elapsed": elapsed,
                    "te_start": te_start, "te_end": te_end,
                    "model": model, "feat_cols": feat_cols,
                }
                split_results.append(split_result)

                # Log per-split details
                vol_rank = regime.get("vol_regime_rank", {}).get("mean", 0)
                ret288 = regime.get("return_288", {}).get("mean", 0)
                trend = regime.get("trend_strength_48", {}).get("mean", 0)

                log(f"\n  Split {i+1}: {date_str}  train={len(X_tr_s):,}  test={len(y_te_i):,}", f)
                log(f"    AUC: raw={raw_auc:.4f}  cal={cal_auc:.4f}  iter={model.best_iteration_}  time={elapsed:.0f}s", f)
                log(f"    Regime: vol_rank={vol_rank:.2f}  ret288={ret288:+.4f}  trend={trend:.3f}", f)

                # Show top configs
                for cname, stats in bt_res.items():
                    if stats["n"] >= 3 and (stats["net"] > 0 or stats["n"] >= 20):
                        pf_str = f"{stats['pf']:.2f}" if stats['pf'] < 100 else "inf"
                        log(f"    {cname:<20s}: n={stats['n']:>4d}  net={stats['net']:>+7.2%}  "
                            f"sr={stats['sharpe']:>6.2f}  WR={stats['wr']:>5.1%}  "
                            f"PF={pf_str:>5s}  kelly={stats['kelly']:>+.3f}  "
                            f"maxDD={stats['max_dd']:>7.2%}", f)

            wf_all_results[h_label] = split_results

            # ── WF Summary ──
            if split_results:
                log(f"\n  {'─'*70}", f)
                log(f"  {h_label} WF SUMMARY ({len(split_results)} splits)", f)
                log(f"  {'─'*70}", f)

                # Aggregate by config
                log(f"  {'Config':<20s} {'Pos/N':>6s} {'Avg Net':>9s} {'Avg SR':>8s} "
                    f"{'Avg WR':>7s} {'Med N':>6s} {'Avg Kelly':>10s}", f)

                config_summaries = {}
                all_config_names = set()
                for sr in split_results:
                    all_config_names.update(sr["bt"].keys())

                for cname in sorted(all_config_names):
                    nets, sharpes, wrs, trade_counts, kellys = [], [], [], [], []
                    for sr in split_results:
                        bt = sr["bt"].get(cname, {"n": 0})
                        if bt["n"] > 0:
                            nets.append(bt["net"])
                            sharpes.append(bt["sharpe"])
                            wrs.append(bt["wr"])
                            trade_counts.append(bt["n"])
                            kellys.append(bt["kelly"])

                    if not nets:
                        continue

                    n_pos = sum(1 for n in nets if n > 0)
                    ratio = f"{n_pos}/{len(nets)}"
                    avg_net = np.mean(nets)
                    avg_sr = np.mean(sharpes)
                    avg_wr = np.mean(wrs)
                    med_n = int(np.median(trade_counts))
                    avg_kelly = np.mean(kellys)

                    log(f"  {cname:<20s} {ratio:>6s} {avg_net:>+8.2%} "
                        f"{avg_sr:>7.2f} {avg_wr:>6.1%} {med_n:>6d} {avg_kelly:>+9.3f}", f)

                    config_summaries[cname] = {
                        "n_pos": n_pos, "n_splits": len(nets),
                        "avg_net": avg_net, "avg_sharpe": avg_sr,
                        "avg_wr": avg_wr, "med_trades": med_n,
                        "avg_kelly": avg_kelly, "nets": nets,
                    }

                # Best config by score: positive_ratio * 10 + avg_net * 50 + avg_kelly * 20
                best_cfg = None
                best_score = -999
                for cname, summary in config_summaries.items():
                    score = (summary["n_pos"] / max(summary["n_splits"], 1)) * 10 + \
                            summary["avg_net"] * 50 + summary["avg_kelly"] * 20
                    if score > best_score and summary["med_trades"] >= 5:
                        best_score = score
                        best_cfg = cname

                if best_cfg:
                    s = config_summaries[best_cfg]
                    log(f"\n  ★ BEST {h_label} WF CONFIG: {best_cfg}", f)
                    log(f"    Positive: {s['n_pos']}/{s['n_splits']}  "
                        f"Avg net: {s['avg_net']:+.2%}  Avg Sharpe: {s['avg_sharpe']:.2f}  "
                        f"Avg Kelly: {s['avg_kelly']:+.3f}", f)
                    log(f"    Per-split nets: {' '.join(f'{n:+.2%}' for n in s['nets'])}", f)

                # AUC summary
                raw_aucs = [sr["raw_auc"] for sr in split_results]
                cal_aucs = [sr["cal_auc"] for sr in split_results]
                log(f"\n  AUC: raw_mean={np.mean(raw_aucs):.4f}±{np.std(raw_aucs):.4f}  "
                    f"cal_mean={np.mean(cal_aucs):.4f}±{np.std(cal_aucs):.4f}", f)
                log(f"  Raw AUCs: {' '.join(f'{a:.4f}' for a in raw_aucs)}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: REGIME ANALYSIS (WHY DO CERTAIN SPLITS FAIL?)
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 2: REGIME ANALYSIS — WHAT PREDICTS WF FAILURE?", f)
        log(f"{'#'*80}", f)

        for h_label, split_results in wf_all_results.items():
            if not split_results:
                continue

            log(f"\n  --- {h_label} Regime Analysis ---", f)

            # Find the best raw config for this horizon
            best_raw_cfg = None
            best_raw_score = -999
            all_cfgs = set()
            for sr in split_results:
                all_cfgs.update(sr["bt"].keys())
            for cname in all_cfgs:
                if not cname.startswith("raw_"):
                    continue
                nets = [sr["bt"].get(cname, {"n": 0, "net": 0})["net"]
                        for sr in split_results if sr["bt"].get(cname, {"n": 0})["n"] > 0]
                if nets:
                    n_pos = sum(1 for n in nets if n > 0)
                    score = n_pos / len(nets) * 10 + np.mean(nets) * 50
                    if score > best_raw_score:
                        best_raw_score = score
                        best_raw_cfg = cname

            if not best_raw_cfg:
                best_raw_cfg = "raw_m04_top20"

            log(f"  Using config: {best_raw_cfg}", f)
            log(f"\n  {'Split':>5s} {'Period':>23s} {'AUC':>6s} {'Net':>8s} {'WR':>6s} "
                f"{'VolRank':>8s} {'Ret288':>8s} {'Trend':>7s} {'VolMom96':>9s} {'VPC24':>7s}", f)

            for sr in split_results:
                bt = sr["bt"].get(best_raw_cfg, {"n": 0, "net": 0, "wr": 0})
                vol_rank = sr["regime"].get("vol_regime_rank", {}).get("mean", 0)
                ret288 = sr["regime"].get("return_288", {}).get("mean", 0)
                trend = sr["regime"].get("trend_strength_48", {}).get("mean", 0)
                vol_mom = sr["regime"].get("volume_momentum_96", {}).get("mean", 0)
                vpc = sr["regime"].get("vol_price_corr_24", {}).get("mean", 0)

                status = "✓" if bt["net"] > 0 else "✗"
                log(f"  {sr['split']:>4d}{status} {sr['date_str']:>23s} {sr['raw_auc']:>5.4f} "
                    f"{bt['net']:>+7.2%} {bt['wr']:>5.1%} "
                    f"{vol_rank:>7.2f} {ret288:>+7.4f} {trend:>6.3f} "
                    f"{vol_mom:>+8.4f} {vpc:>+6.3f}", f)

            # Correlation between regime features and performance
            log(f"\n  Regime-Performance Correlation:", f)
            nets = []
            aucs = []
            vol_ranks = []
            ret288s = []
            trends = []
            vol_moms = []
            vpcs = []
            for sr in split_results:
                bt = sr["bt"].get(best_raw_cfg, {"n": 0, "net": 0})
                if bt["n"] > 0:
                    nets.append(bt["net"])
                    aucs.append(sr["raw_auc"])
                    vol_ranks.append(sr["regime"].get("vol_regime_rank", {}).get("mean", 0))
                    ret288s.append(sr["regime"].get("return_288", {}).get("mean", 0))
                    trends.append(sr["regime"].get("trend_strength_48", {}).get("mean", 0))
                    vol_moms.append(sr["regime"].get("volume_momentum_96", {}).get("mean", 0))
                    vpcs.append(sr["regime"].get("vol_price_corr_24", {}).get("mean", 0))

            if len(nets) >= 4:
                nets_a = np.array(nets)
                for name, vals in [("vol_rank", vol_ranks), ("ret288", ret288s),
                                   ("trend", trends), ("vol_mom96", vol_moms),
                                   ("vpc24", vpcs), ("auc", aucs)]:
                    corr = np.corrcoef(nets_a, np.array(vals))[0, 1]
                    log(f"    corr(net, {name}) = {corr:+.3f}", f)

                # Which regime predicts failure?
                pos_mask = nets_a > 0
                neg_mask = nets_a <= 0
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    log(f"\n  Winning splits vs Losing splits:", f)
                    for name, vals in [("vol_rank", vol_ranks), ("ret288", ret288s),
                                       ("trend", trends), ("vol_mom96", vol_moms)]:
                        vals_a = np.array(vals)
                        win_mean = vals_a[pos_mask].mean()
                        lose_mean = vals_a[neg_mask].mean()
                        log(f"    {name:<12s}: winners={win_mean:+.4f}  losers={lose_mean:+.4f}  "
                            f"diff={win_mean-lose_mean:+.4f}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: REGIME-GATED WALK-FORWARD
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 3: REGIME-GATED WALK-FORWARD TRADING", f)
        log(f"{'#'*80}", f)

        for h_label, split_results in wf_all_results.items():
            if not split_results:
                continue

            horizon = 12 if h_label == "1h" else 36

            log(f"\n  --- {h_label} Regime-Gated WF ---", f)

            # For each split, re-run backtest but GATE on regime
            regime_configs = [
                ("no_gate", lambda df_te: np.ones(len(df_te), dtype=bool)),
            ]

            # Add regime gates based on available columns
            if "vol_regime_rank" in df.columns:
                regime_configs.extend([
                    ("low_vol(<0.4)", lambda df_te: df_te["vol_regime_rank"].values < 0.4),
                    ("mid_vol(0.3-0.7)", lambda df_te: (df_te["vol_regime_rank"].values >= 0.3) & (df_te["vol_regime_rank"].values < 0.7)),
                    ("high_vol(>0.6)", lambda df_te: df_te["vol_regime_rank"].values >= 0.6),
                    ("not_extreme(<0.8)", lambda df_te: df_te["vol_regime_rank"].values < 0.8),
                ])

            if "trend_strength_48" in df.columns:
                regime_configs.extend([
                    ("trending(>0.5)", lambda df_te: df_te["trend_strength_48"].values > 0.5),
                    ("ranging(<0.3)", lambda df_te: df_te["trend_strength_48"].values < 0.3),
                ])

            for margin_cfg in ["m04_top20", "m06_top20"]:
                margin = float(margin_cfg.split("_")[0][1:]) / 100
                top_pct_str = margin_cfg.split("_")[1]
                top_pct = float(top_pct_str.replace("top", "").replace("%", "")) / 100 if "top" in top_pct_str else None

                log(f"\n  Config: raw_{margin_cfg}", f)
                log(f"  {'Gate':<20s} {'Pos/N':>6s} {'Avg Net':>9s} {'Avg SR':>8s} "
                    f"{'Avg WR':>7s} {'Med N':>6s}", f)

                for gate_name, gate_fn in regime_configs:
                    gate_nets = []
                    gate_sharpes = []
                    gate_wrs = []
                    gate_trades = []

                    for sr in split_results:
                        te_start = sr["te_start"]
                        te_end = sr["te_end"]
                        df_te = df.iloc[te_start:te_end]
                        target = f"target_direction_{horizon}"
                        y_te = df_te[target].dropna()

                        if len(y_te) < 50:
                            continue

                        model = sr["model"]
                        feat_cols = sr["feat_cols"]
                        X_te = df_te.loc[y_te.index, feat_cols]
                        y_prob = model.predict_proba(X_te)[:, 1]
                        close_te = df_te.loc[y_te.index, "close"].values

                        # Apply regime gate
                        try:
                            gate_mask = gate_fn(df_te.loc[y_te.index])
                        except Exception:
                            continue

                        if gate_mask.sum() < 20:
                            continue

                        close_gated = close_te[gate_mask]
                        prob_gated = y_prob[gate_mask]

                        trades = backtest_lo(close_gated, prob_gated, horizon,
                                             FEE_MAKER_RT, margin=margin, top_pct=top_pct)
                        stats = bt_stats(trades, horizon)

                        if stats["n"] > 0:
                            gate_nets.append(stats["net"])
                            gate_sharpes.append(stats["sharpe"])
                            gate_wrs.append(stats["wr"])
                            gate_trades.append(stats["n"])

                    if gate_nets:
                        n_pos = sum(1 for n in gate_nets if n > 0)
                        ratio = f"{n_pos}/{len(gate_nets)}"
                        log(f"  {gate_name:<20s} {ratio:>6s} {np.mean(gate_nets):>+8.2%} "
                            f"{np.mean(gate_sharpes):>7.2f} {np.mean(gate_wrs):>6.1%} "
                            f"{int(np.median(gate_trades)):>6d}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: FEATURE STABILITY ANALYSIS
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 4: FEATURE STABILITY ACROSS WF SPLITS", f)
        log(f"{'#'*80}", f)

        stable_features = {}  # {h_label: [stable feature names]}

        for h_label, split_results in wf_all_results.items():
            if not split_results or len(split_results) < 3:
                continue

            log(f"\n  --- {h_label} Feature Stability ---", f)

            # Collect feature importances from all WF split models
            all_imps = []
            for sr in split_results:
                model = sr["model"]
                feat_cols = sr["feat_cols"]
                imp = pd.Series(model.feature_importances_, index=feat_cols)
                all_imps.append(imp)

            imp_df = pd.DataFrame(all_imps).fillna(0)
            mean_imp = imp_df.mean()
            std_imp = imp_df.std()
            cv_imp = std_imp / (mean_imp + 1e-10)

            # Rank consistency: for each feature, what's the average rank across splits?
            rank_df = imp_df.rank(axis=1, ascending=False)
            avg_rank = rank_df.mean()
            rank_std = rank_df.std()

            # Stability score: high importance + low CV + consistent rank
            stability_score = mean_imp * (1 / (cv_imp + 0.1)) * (1 / (avg_rank + 1))
            top_stable = stability_score.nlargest(30)

            log(f"\n  Top 25 STABLE features (high importance, low variability):", f)
            log(f"  {'Rank':>4s} {'Feature':<35s} {'MeanImp':>8s} {'CV':>6s} {'AvgRank':>8s} {'Score':>8s}", f)
            for rank, (feat, score) in enumerate(top_stable.head(25).items(), 1):
                log(f"  {rank:>4d} {feat:<35s} {mean_imp[feat]:>7.0f} {cv_imp[feat]:>5.2f} "
                    f"{avg_rank[feat]:>7.1f} {score:>7.0f}", f)

            # Store top-50 stable features for Phase 5
            stable_features[h_label] = stability_score.nlargest(50).index.tolist()

            # Most UNSTABLE features (high importance but high variance)
            significant = mean_imp[mean_imp > 10]
            if len(significant) > 0:
                unstable = cv_imp[significant.index].nlargest(10)
                log(f"\n  Top 10 UNSTABLE features (high importance, high CV):", f)
                for feat, cv in unstable.items():
                    log(f"    {feat:<35s}  mean_imp={mean_imp[feat]:6.0f}  cv={cv:.2f}  "
                        f"avg_rank={avg_rank[feat]:.1f}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: STABLE-FEATURES-ONLY WALK-FORWARD
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 5: STABLE-FEATURES-ONLY WALK-FORWARD", f)
        log(f"{'#'*80}", f)

        for h_label in ["1h", "3h"]:
            if h_label not in stable_features or h_label not in wf_all_results:
                continue

            horizon = 12 if h_label == "1h" else 36
            target = f"target_direction_{horizon}"
            feat_cols_stable = stable_features[h_label]
            purge_gap = horizon * 2
            test_size = int(n_total * test_frac)

            log(f"\n  --- {h_label} with {len(feat_cols_stable)} stable features ---", f)

            stable_split_results = []
            for i in range(n_splits):
                te_end = n_total - i * test_size
                te_start = te_end - test_size
                if te_start < 0:
                    break
                tr_end_i = te_start - purge_gap
                if tr_end_i < test_size * 2:
                    break

                df_te = df.iloc[te_start:te_end]
                df_tr = df.iloc[:tr_end_i]

                y_tr_i = df_tr[target].dropna()
                y_te_i = df_te[target].dropna()
                if len(y_tr_i) < 5000 or len(y_te_i) < 200:
                    continue

                X_tr_full = df_tr.loc[y_tr_i.index, feat_cols_stable]
                X_te_i = df_te.loc[y_te_i.index, feat_cols_stable]

                val_cut = int(len(X_tr_full) * 0.85)
                X_tr_s = X_tr_full.iloc[:val_cut]
                y_tr_s = y_tr_i.iloc[:val_cut]
                X_va_s = X_tr_full.iloc[val_cut:]
                y_va_s = y_tr_i.iloc[val_cut:]

                model = train_lgb(X_tr_s, y_tr_s, X_va_s, y_va_s, BASE_PARAMS)
                y_prob = model.predict_proba(X_te_i)[:, 1]
                raw_auc = auc_roc(y_te_i.values, y_prob)
                close_te = df_te.loc[y_te_i.index, "close"].values

                bt_res = {}
                for cname, margin, top_pct in BT_CONFIGS:
                    trades = backtest_lo(close_te, y_prob, horizon, FEE_MAKER_RT,
                                         margin=margin, top_pct=top_pct)
                    stats = bt_stats(trades, horizon)
                    bt_res[cname] = stats

                stable_split_results.append({
                    "split": i + 1, "auc": raw_auc, "bt": bt_res,
                    "date_str": get_date_range(df_te),
                })

            # Compare stable vs all features
            if stable_split_results:
                log(f"\n  {h_label} STABLE FEATURES WF SUMMARY:", f)
                log(f"  {'Config':<20s} {'Pos/N':>6s} {'Avg Net':>9s} {'Avg SR':>8s} "
                    f"{'Avg WR':>7s}", f)

                for cname, _, _ in BT_CONFIGS:
                    nets, sharpes, wrs = [], [], []
                    for sr in stable_split_results:
                        bt = sr["bt"].get(cname, {"n": 0})
                        if bt["n"] > 0:
                            nets.append(bt["net"])
                            sharpes.append(bt["sharpe"])
                            wrs.append(bt["wr"])
                    if nets:
                        n_pos = sum(1 for n in nets if n > 0)
                        log(f"  {cname:<20s} {n_pos}/{len(nets):>3d} {np.mean(nets):>+8.2%} "
                            f"{np.mean(sharpes):>7.2f} {np.mean(wrs):>6.1%}", f)

                # AUC comparison
                stable_aucs = [sr["auc"] for sr in stable_split_results]
                orig_aucs = [sr["raw_auc"] for sr in wf_all_results.get(h_label, [])]
                if orig_aucs and stable_aucs:
                    log(f"\n  AUC comparison:", f)
                    log(f"    All features (80):    {np.mean(orig_aucs):.4f}±{np.std(orig_aucs):.4f}", f)
                    log(f"    Stable features ({len(feat_cols_stable)}): {np.mean(stable_aucs):.4f}±{np.std(stable_aucs):.4f}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 6: MULTI-HORIZON CONSENSUS WALK-FORWARD
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 6: MULTI-HORIZON CONSENSUS WALK-FORWARD", f)
        log(f"{'#'*80}", f)

        consensus_horizons = [(12, "1h"), (36, "3h"), (48, "4h")]
        n_wf_consensus = 8
        test_frac_c = 0.06

        log(f"\n  Horizons: {[h for h, _ in consensus_horizons]}", f)
        log(f"  WF splits: {n_wf_consensus}, test fraction: {test_frac_c}", f)

        test_size_c = int(n_total * test_frac_c)
        consensus_wf_results = []

        for split_i in range(n_wf_consensus):
            te_end = n_total - split_i * test_size_c
            te_start = te_end - test_size_c
            if te_start < 0:
                break

            max_purge = max(h * 2 for h, _ in consensus_horizons)
            tr_end = te_start - max_purge
            if tr_end < test_size_c * 2:
                break

            df_te = df.iloc[te_start:te_end]
            df_tr = df.iloc[:tr_end]
            date_str = get_date_range(df_te)

            log(f"\n  === Consensus Split {split_i+1}: {date_str} ===", f)

            # Train model for each horizon
            horizon_probs = {}
            for horizon, h_label in consensus_horizons:
                target = f"target_direction_{horizon}"
                feat_cols = horizon_feats.get(h_label, all_feat_cols[:80])

                y_tr_h = df_tr[target].dropna()
                y_te_h = df_te[target].dropna()
                if len(y_tr_h) < 5000 or len(y_te_h) < 200:
                    continue

                X_tr_full = df_tr.loc[y_tr_h.index, feat_cols]
                val_cut = int(len(X_tr_full) * 0.85)

                m = train_lgb(
                    X_tr_full.iloc[:val_cut], y_tr_h.iloc[:val_cut],
                    X_tr_full.iloc[val_cut:], y_tr_h.iloc[val_cut:],
                    BASE_PARAMS
                )
                prob = m.predict_proba(df_te.loc[y_te_h.index, feat_cols])[:, 1]
                _auc = auc_roc(y_te_h.values, prob)

                # Map prob to full test index
                full_prob = np.full(len(df_te), np.nan)
                for j, idx in enumerate(y_te_h.index):
                    local_pos = df_te.index.get_loc(idx)
                    if isinstance(local_pos, (int, np.integer)):
                        full_prob[local_pos] = prob[j]
                    elif isinstance(local_pos, np.ndarray):
                        full_prob[local_pos[0]] = prob[j]

                horizon_probs[horizon] = full_prob
                log(f"    {h_label}: AUC={_auc:.4f}  iter={m.best_iteration_}", f)

            close_te = df_te["close"].values

            # Single-model baselines
            single_stats = {}
            for horizon, h_label in consensus_horizons:
                if horizon not in horizon_probs:
                    continue
                prob = horizon_probs[horizon]
                valid_mask = ~np.isnan(prob)
                if valid_mask.sum() < 50:
                    continue
                close_valid = close_te[valid_mask]
                prob_valid = prob[valid_mask]

                for margin in [0.04, 0.06]:
                    for top_pct in [0.20, 0.10]:
                        trades = backtest_lo(close_valid, prob_valid, horizon,
                                             FEE_MAKER_RT, margin=margin, top_pct=top_pct)
                        stats = bt_stats(trades, horizon)
                        if stats["n"] >= 3:
                            cfg = f"{h_label}_m{int(margin*100):02d}_top{int(top_pct*100)}%"
                            single_stats[cfg] = stats
                            log(f"    Single {cfg}: n={stats['n']}  net={stats['net']:+.2%}  "
                                f"sr={stats['sharpe']:.2f}  WR={stats['wr']:.1%}", f)

            # Consensus: trade when N+ horizons agree
            log(f"    CONSENSUS:", f)
            consensus_results_split = {}
            for min_agree in [2, 3]:
                for exec_h in [12, 36]:
                    for margin in [0.02, 0.04, 0.06]:
                        trades = []
                        i = 0
                        while i + exec_h < len(close_te):
                            bullish = 0
                            valid = 0
                            for h in horizon_probs:
                                p = horizon_probs[h][i]
                                if np.isnan(p):
                                    continue
                                valid += 1
                                if p > 0.5 + margin:
                                    bullish += 1

                            if valid >= min_agree and bullish >= min_agree:
                                raw_ret = (close_te[i + exec_h] - close_te[i]) / close_te[i]
                                net_ret = raw_ret - FEE_MAKER_RT
                                trades.append({"idx": i, "net_ret": net_ret,
                                               "raw_ret": raw_ret, "prob": bullish / valid})
                            i += exec_h

                        if trades:
                            stats = bt_stats(trades, exec_h)
                            cfg = f"agree{min_agree}_exec{exec_h}_m{int(margin*100):02d}"
                            consensus_results_split[cfg] = stats
                            if stats["n"] >= 3:
                                log(f"      {cfg}: n={stats['n']}  net={stats['net']:+.2%}  "
                                    f"sr={stats['sharpe']:.2f}  WR={stats['wr']:.1%}", f)

            consensus_wf_results.append({
                "split": split_i + 1, "date_str": date_str,
                "single": single_stats,
                "consensus": consensus_results_split,
            })

        # Consensus WF Summary
        if consensus_wf_results:
            log(f"\n  {'─'*70}", f)
            log(f"  CONSENSUS WF SUMMARY ({len(consensus_wf_results)} splits)", f)
            log(f"  {'─'*70}", f)

            # Aggregate consensus results
            all_consensus_cfgs = set()
            for cwr in consensus_wf_results:
                all_consensus_cfgs.update(cwr["consensus"].keys())

            log(f"\n  {'Config':<25s} {'Pos/N':>6s} {'Avg Net':>9s} {'Avg SR':>8s} "
                f"{'Avg WR':>7s} {'Med N':>6s}", f)

            for cfg in sorted(all_consensus_cfgs):
                nets, sharpes, wrs, trade_counts = [], [], [], []
                for cwr in consensus_wf_results:
                    bt = cwr["consensus"].get(cfg, {"n": 0})
                    if bt["n"] > 0:
                        nets.append(bt["net"])
                        sharpes.append(bt["sharpe"])
                        wrs.append(bt["wr"])
                        trade_counts.append(bt["n"])
                if nets and np.median(trade_counts) >= 3:
                    n_pos = sum(1 for n in nets if n > 0)
                    log(f"  {cfg:<25s} {n_pos}/{len(nets):>3d} {np.mean(nets):>+8.2%} "
                        f"{np.mean(sharpes):>7.2f} {np.mean(wrs):>6.1%} "
                        f"{int(np.median(trade_counts)):>6d}", f)

            # Also summarize single model baselines across splits
            log(f"\n  Single-model baselines across consensus splits:", f)
            all_single_cfgs = set()
            for cwr in consensus_wf_results:
                all_single_cfgs.update(cwr["single"].keys())

            for cfg in sorted(all_single_cfgs):
                nets, sharpes, wrs = [], [], []
                for cwr in consensus_wf_results:
                    bt = cwr["single"].get(cfg, {"n": 0})
                    if bt["n"] > 0:
                        nets.append(bt["net"])
                        sharpes.append(bt["sharpe"])
                        wrs.append(bt["wr"])
                if nets:
                    n_pos = sum(1 for n in nets if n > 0)
                    log(f"  {cfg:<25s} {n_pos}/{len(nets):>3d} {np.mean(nets):>+8.2%} "
                        f"{np.mean(sharpes):>7.2f} {np.mean(wrs):>6.1%}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 7: DYNAMIC CONFIDENCE THRESHOLDS
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 7: DYNAMIC CONFIDENCE THRESHOLDS (ROLLING PERCENTILE)", f)
        log(f"{'#'*80}", f)

        # Instead of fixed margin (e.g., prob > 0.54), use rolling percentile
        # of the model's own prediction distribution as the threshold.
        # This adapts to changing prediction distributions across regimes.

        for h_label, split_results in wf_all_results.items():
            if not split_results:
                continue

            horizon = 12 if h_label == "1h" else 36
            target = f"target_direction_{horizon}"

            log(f"\n  --- {h_label} Dynamic Thresholds ---", f)

            for sr in split_results[:6]:  # first 6 splits for speed
                te_start = sr["te_start"]
                te_end = sr["te_end"]
                df_te = df.iloc[te_start:te_end]
                y_te = df_te[target].dropna()

                if len(y_te) < 50:
                    continue

                model = sr["model"]
                feat_cols = sr["feat_cols"]
                X_te = df_te.loc[y_te.index, feat_cols]
                y_prob = model.predict_proba(X_te)[:, 1]
                close_te = df_te.loc[y_te.index, "close"].values

                # Dynamic threshold: use rolling 288-period (1-day) percentile
                # Trade when prob is in top X% of recent predictions
                log(f"\n  Split {sr['split']} ({sr['date_str']}):", f)

                for pctile in [90, 85, 80, 75, 70]:
                    trades = []
                    i = 0
                    lookback = 288  # 1 day of 5m candles
                    while i + horizon < len(close_te):
                        if i < lookback:
                            # Not enough history, use fixed threshold
                            threshold = 0.5 + 0.04
                        else:
                            # Rolling threshold: X-th percentile of recent predictions
                            recent_probs = y_prob[max(0, i - lookback):i]
                            threshold = np.percentile(recent_probs, pctile)

                        if y_prob[i] > threshold:
                            raw_ret = (close_te[i + horizon] - close_te[i]) / close_te[i]
                            trades.append({"idx": i, "net_ret": raw_ret - FEE_MAKER_RT,
                                           "raw_ret": raw_ret, "prob": y_prob[i]})
                        i += horizon

                    stats = bt_stats(trades, horizon)
                    if stats["n"] >= 3:
                        log(f"    p{pctile}: n={stats['n']:>4d}  net={stats['net']:>+7.2%}  "
                            f"sr={stats['sharpe']:>6.2f}  WR={stats['wr']:>5.1%}  "
                            f"kelly={stats['kelly']:>+.3f}", f)

        # ═══════════════════════════════════════════════════════════════
        # FINAL VERDICT
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  ITERATION 6 — FINAL VERDICT", f)
        log(f"{'#'*80}", f)

        log(f"\n  === 10-Split Walk-Forward Results ===", f)
        for h_label, split_results in wf_all_results.items():
            if not split_results:
                continue
            raw_aucs = [sr["raw_auc"] for sr in split_results]
            log(f"  {h_label}: {len(split_results)} splits  AUC={np.mean(raw_aucs):.4f}±{np.std(raw_aucs):.4f}", f)

            # Find best config
            all_cfgs = set()
            for sr in split_results:
                all_cfgs.update(sr["bt"].keys())

            best_cfg = None
            best_score = -999
            for cname in sorted(all_cfgs):
                nets = [sr["bt"].get(cname, {"n": 0, "net": 0})["net"]
                        for sr in split_results if sr["bt"].get(cname, {"n": 0})["n"] > 0]
                if len(nets) >= 3:
                    n_pos = sum(1 for n in nets if n > 0)
                    score = (n_pos / len(nets)) * 10 + np.mean(nets) * 50
                    if score > best_score:
                        best_score = score
                        best_cfg = cname

            if best_cfg:
                nets = [sr["bt"].get(best_cfg, {"n": 0, "net": 0})["net"]
                        for sr in split_results if sr["bt"].get(best_cfg, {"n": 0})["n"] > 0]
                n_pos = sum(1 for n in nets if n > 0)
                log(f"    Best config: {best_cfg}  Positive: {n_pos}/{len(nets)}  "
                    f"Avg net: {np.mean(nets):+.2%}", f)

        log(f"\n  === Key Findings ===", f)
        log(f"  (To be filled after analyzing results)", f)

        log(f"\n  === Next Steps ===", f)
        log(f"  (To be determined based on results above)", f)

        # Save models from most recent WF split
        for h_label, split_results in wf_all_results.items():
            if split_results:
                horizon = 12 if h_label == "1h" else 36
                model = split_results[0]["model"]  # most recent split
                model_path = os.path.join(MODEL_DIR, f"v6_{h_label}_{horizon}.txt")
                model.booster_.save_model(model_path)
                log(f"\n  Saved {model_path}", f)

    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
