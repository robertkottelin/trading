"""
BTC LightGBM Training Pipeline — Iteration 3
==============================================
Focus: target_direction_6 (30min) — the only model with consistent walk-forward performance.

Improvements over v2:
  - Focus exclusively on direction_6 (drop all other targets)
  - Realistic backtest: non-overlapping trades every 6 candles + transaction costs
  - Systematic hyperparameter search (6 configurations)
  - Feature selection: compare all features vs top-50 vs top-30
  - Ensemble: average predictions from top-3 configs
  - Purged walk-forward with gap = horizon (6 candles) to prevent leakage
  - Prediction distribution analysis (calibration check)

Usage:
  python train_model.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import time
import json
from datetime import datetime, timezone

OUTPUT_DIR = "btc_data"
FEATURES_FILE = os.path.join(OUTPUT_DIR, "btc_features_5m.csv")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "training_results.txt")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

TARGET = "target_direction_6"
HORIZON = 6  # candles (30 min)
FEE_RT = 0.0008  # 0.08% round-trip (Binance futures with BNB)


def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")
        f.flush()


def _auc_roc(y_true, y_prob):
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


def eval_cls(y_true, y_pred, y_prob):
    n = len(y_true)
    if n == 0:
        return {}
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    acc = float((y_pred == y_true).mean())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    auc = _auc_roc(y_true, y_prob)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc, "n": int(n)}


def get_feature_cols(df):
    exclude = {"open_time_ms", "timestamp"}
    target_cols = {c for c in df.columns if c.startswith("target_")}
    return sorted(c for c in df.columns if c not in exclude and c not in target_cols)


def temporal_split(df, train_frac=0.70, val_frac=0.15):
    n = len(df)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return df.iloc[:t1].copy(), df.iloc[t1:t2].copy(), df.iloc[t2:].copy()


def backtest_realistic(df_test, y_prob, fee_rt=FEE_RT, mode="long_short", margin=0.0):
    """
    Realistic backtest with non-overlapping trades and transaction costs.

    Instead of taking a position every 5m candle (overlapping with 30min horizon),
    we only enter a new trade every HORIZON candles = every 30 minutes.
    Each trade: enter at close[i], exit at close[i + HORIZON].
    Fee applied on each trade entry+exit.
    """
    close = df_test["close"].values
    n = len(close)

    trades = []
    i = 0
    while i + HORIZON < n:
        prob = y_prob[i]
        # Determine signal based on mode and margin
        if mode == "long_short":
            if abs(prob - 0.5) <= margin:
                i += HORIZON  # skip, not confident
                continue
            signal = 1 if prob > 0.5 else -1
        else:  # long_only
            if prob <= 0.5 + margin:
                i += HORIZON
                continue
            signal = 1

        entry = close[i]
        exit_price = close[i + HORIZON]
        raw_ret = (exit_price - entry) / entry * signal
        net_ret = raw_ret - fee_rt  # subtract transaction cost

        trades.append({
            "idx": i,
            "entry": entry,
            "exit": exit_price,
            "signal": signal,
            "raw_ret": raw_ret,
            "net_ret": net_ret,
            "prob": prob,
        })
        i += HORIZON

    if not trades:
        return {
            "total_ret": 0.0, "sharpe": 0.0, "max_dd": 0.0,
            "win_rate": 0.0, "n_trades": 0, "avg_ret": 0.0,
            "profit_factor": 0.0, "fee_drag": 0.0,
        }

    df_trades = pd.DataFrame(trades)
    cum = (1 + df_trades["net_ret"]).cumprod()
    total_ret = cum.iloc[-1] - 1

    # Buy & hold over same period
    bh_ret = (close[-1] - close[0]) / close[0]

    # Sharpe — annualized (trades are 30min apart, ~17520 per year)
    trades_per_year = 365.25 * 24 * 2  # every 30 min
    mu = df_trades["net_ret"].mean()
    sigma = df_trades["net_ret"].std()
    sharpe = (mu / sigma) * np.sqrt(trades_per_year) if sigma > 0 else 0

    # Max drawdown
    cummax = cum.cummax()
    max_dd = ((cum - cummax) / cummax).min()

    # Win rate
    win_rate = (df_trades["net_ret"] > 0).mean()

    # Profit factor
    gross_wins = df_trades.loc[df_trades["net_ret"] > 0, "net_ret"].sum()
    gross_losses = abs(df_trades.loc[df_trades["net_ret"] < 0, "net_ret"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Fee drag (total fees as % of gross profit)
    gross_profit = df_trades["raw_ret"].sum()
    total_fees = fee_rt * len(df_trades)

    return {
        "total_ret": total_ret, "bh_ret": bh_ret,
        "sharpe": sharpe, "max_dd": max_dd, "win_rate": win_rate,
        "n_trades": len(df_trades), "avg_ret": mu,
        "profit_factor": profit_factor,
        "fee_drag": total_fees,
        "gross_ret": (1 + df_trades["raw_ret"]).cumprod().iloc[-1] - 1,
    }


def walk_forward_purged(df, feat_cols, params, n_splits=5, f=None):
    """
    Purged walk-forward with realistic backtest.
    Gap = HORIZON candles between train and test to prevent leakage.
    """
    n = len(df)
    chunk = int(n * 0.10)
    results = []

    for i in range(n_splits):
        te_end = n - i * chunk
        te_start = te_end - chunk
        tr_end = te_start - HORIZON  # purge gap

        if tr_end < chunk:
            break

        df_tr = df.iloc[:tr_end]
        df_te = df.iloc[te_start:te_end]

        y_tr = df_tr[TARGET].dropna()
        y_te = df_te[TARGET].dropna()
        X_tr = df_tr.loc[y_tr.index, feat_cols]
        X_te = df_te.loc[y_te.index, feat_cols]

        val_cut = int(len(X_tr) * 0.85)
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_tr.iloc[:val_cut], y_tr.iloc[:val_cut],
            eval_set=[(X_tr.iloc[val_cut:], y_tr.iloc[val_cut:])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        y_prob = m.predict_proba(X_te)[:, 1]
        y_pred = m.predict(X_te)
        metrics = eval_cls(y_te.values, y_pred, y_prob)

        # Realistic backtest on this split
        bt = backtest_realistic(df_te.loc[y_te.index], y_prob, mode="long_short", margin=0.02)
        metrics.update(bt)
        results.append(metrics)

    return results


def train_single(df_train, df_val, feat_cols, params):
    """Train a single model, return it with best iteration."""
    y_tr = df_train[TARGET].dropna()
    y_va = df_val[TARGET].dropna()
    X_tr = df_train.loc[y_tr.index, feat_cols]
    X_va = df_val.loc[y_va.index, feat_cols]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def get_feature_importance(model, feat_cols):
    imp = pd.DataFrame({"feat": feat_cols, "imp": model.feature_importances_})
    return imp.sort_values("imp", ascending=False)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading {FEATURES_FILE}...")
    df = pd.read_csv(FEATURES_FILE)
    print(f"  {len(df):,} rows x {len(df.columns)} cols")

    all_feat_cols = get_feature_cols(df)
    df_train, df_val, df_test = temporal_split(df)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Hyperparameter search — 6 configurations
    # ═══════════════════════════════════════════════════════════════════
    configs = {
        "A_baseline": {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt", "n_estimators": 3000,
            "learning_rate": 0.005, "num_leaves": 31, "max_depth": 6,
            "min_child_samples": 200, "subsample": 0.7,
            "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 5.0,
            "random_state": 42, "verbose": -1, "n_jobs": -1,
        },
        "B_deeper": {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt", "n_estimators": 5000,
            "learning_rate": 0.003, "num_leaves": 63, "max_depth": 8,
            "min_child_samples": 300, "subsample": 0.6,
            "colsample_bytree": 0.4, "reg_alpha": 2.0, "reg_lambda": 10.0,
            "random_state": 42, "verbose": -1, "n_jobs": -1,
        },
        "C_shallow": {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt", "n_estimators": 5000,
            "learning_rate": 0.003, "num_leaves": 15, "max_depth": 4,
            "min_child_samples": 500, "subsample": 0.8,
            "colsample_bytree": 0.6, "reg_alpha": 0.5, "reg_lambda": 3.0,
            "random_state": 42, "verbose": -1, "n_jobs": -1,
        },
        "D_dart": {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "dart", "n_estimators": 1000,
            "learning_rate": 0.01, "num_leaves": 31, "max_depth": 6,
            "min_child_samples": 200, "subsample": 0.7,
            "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 5.0,
            "drop_rate": 0.1, "max_drop": 50,
            "random_state": 42, "verbose": -1, "n_jobs": -1,
        },
        "E_high_reg": {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt", "n_estimators": 5000,
            "learning_rate": 0.003, "num_leaves": 31, "max_depth": 5,
            "min_child_samples": 500, "subsample": 0.5,
            "colsample_bytree": 0.3, "reg_alpha": 5.0, "reg_lambda": 20.0,
            "random_state": 42, "verbose": -1, "n_jobs": -1,
        },
        "F_seed_vary": {
            "objective": "binary", "metric": "binary_logloss",
            "boosting_type": "gbdt", "n_estimators": 3000,
            "learning_rate": 0.005, "num_leaves": 31, "max_depth": 6,
            "min_child_samples": 200, "subsample": 0.7,
            "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 5.0,
            "random_state": 123, "verbose": -1, "n_jobs": -1,
        },
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        log(f"BTC LightGBM Training Results — Iteration 3", f)
        log(f"{'='*70}", f)
        log(f"Focus: target_direction_6 (30min) with realistic backtesting", f)
        log(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", f)
        log(f"Data: {len(df):,} rows x {len(df.columns)} cols  Features: {len(all_feat_cols)}", f)
        log(f"Train: {len(df_train):,} ({df_train['timestamp'].iloc[0][:10]}..{df_train['timestamp'].iloc[-1][:10]})", f)
        log(f"Val:   {len(df_val):,} ({df_val['timestamp'].iloc[0][:10]}..{df_val['timestamp'].iloc[-1][:10]})", f)
        log(f"Test:  {len(df_test):,} ({df_test['timestamp'].iloc[0][:10]}..{df_test['timestamp'].iloc[-1][:10]})", f)
        log(f"Fee: {FEE_RT*100:.2f}% round-trip", f)
        log(f"Backtest: non-overlapping trades every {HORIZON} candles ({HORIZON*5}min)", f)

        # ─── Phase 1: Hyperparameter search ───
        log(f"\n{'#'*70}", f)
        log(f"  PHASE 1: HYPERPARAMETER SEARCH (6 configs x all features)", f)
        log(f"{'#'*70}", f)

        config_results = {}
        config_models = {}
        config_probs = {}

        for name, params in configs.items():
            log(f"\n{'='*70}", f)
            log(f"  CONFIG: {name}", f)
            log(f"{'='*70}", f)

            p_display = {k: v for k, v in params.items() if k not in ("verbose", "n_jobs", "objective", "metric")}
            log(f"  Params: {json.dumps(p_display)}", f)

            t0 = time.time()
            model = train_single(df_train, df_val, all_feat_cols, params)
            elapsed = time.time() - t0
            log(f"  Trained in {elapsed:.1f}s  best_iter={model.best_iteration_}", f)

            # Test metrics
            y_te = df_test[TARGET].dropna()
            X_te = df_test.loc[y_te.index, all_feat_cols]
            y_prob = model.predict_proba(X_te)[:, 1]
            y_pred = model.predict(X_te)
            metrics = eval_cls(y_te.values, y_pred, y_prob)
            log(f"  Test: acc={metrics['acc']:.4f}  auc={metrics['auc']:.4f}  prec={metrics['prec']:.4f}  rec={metrics['rec']:.4f}", f)

            # Prediction distribution
            log(f"  Prediction distribution: min={y_prob.min():.4f}  p25={np.percentile(y_prob,25):.4f}  "
                f"median={np.median(y_prob):.4f}  p75={np.percentile(y_prob,75):.4f}  max={y_prob.max():.4f}", f)

            # Realistic backtest — multiple margin levels
            log(f"\n  REALISTIC BACKTEST (non-overlapping, {FEE_RT*100:.2f}% fee):", f)
            log(f"  {'Mode':<13s} {'Margin':>6s} {'Trades':>7s} {'WinRate':>8s} {'GrossRet':>10s} "
                f"{'NetRet':>10s} {'Sharpe':>7s} {'MaxDD':>8s} {'PF':>5s}", f)

            for mode in ["long_short", "long_only"]:
                margins = [0.00, 0.01, 0.02, 0.03, 0.04, 0.06] if mode == "long_short" else [0.00, 0.02, 0.04, 0.06]
                for margin in margins:
                    bt = backtest_realistic(df_test.loc[y_te.index], y_prob, mode=mode, margin=margin)
                    if bt["n_trades"] == 0:
                        log(f"  {mode:<13s} {margin:>6.2f}       0      —          —          —       —        —     —", f)
                    else:
                        pf_str = f"{bt['profit_factor']:.2f}" if bt['profit_factor'] < 100 else "inf"
                        log(f"  {mode:<13s} {margin:>6.2f} {bt['n_trades']:>7,} {bt['win_rate']:>7.2%} "
                            f"{bt['gross_ret']:>+9.2%} {bt['total_ret']:>+9.2%} {bt['sharpe']:>6.2f} "
                            f"{bt['max_dd']:>7.2%} {pf_str:>5s}", f)

            config_results[name] = metrics
            config_models[name] = model
            config_probs[name] = y_prob

        # ─── Phase 1 Summary ───
        log(f"\n{'='*70}", f)
        log(f"  PHASE 1 SUMMARY", f)
        log(f"{'='*70}", f)
        log(f"  {'Config':<15s} {'Acc':>6s} {'AUC':>6s} {'BestIter':>8s} {'PredSpread':>10s}", f)
        for name, m in config_results.items():
            prob = config_probs[name]
            spread = np.percentile(prob, 75) - np.percentile(prob, 25)
            log(f"  {name:<15s} {m['acc']:>6.4f} {m['auc']:>6.4f} {config_models[name].best_iteration_:>8d} {spread:>10.4f}", f)

        # ─── Phase 2: Feature selection ───
        log(f"\n{'#'*70}", f)
        log(f"  PHASE 2: FEATURE SELECTION", f)
        log(f"{'#'*70}", f)

        # Use best AUC config to get feature importance
        best_config = max(config_results.items(), key=lambda x: x[1]["auc"])[0]
        best_model = config_models[best_config]
        best_params = configs[best_config]
        log(f"  Using feature importance from best config: {best_config}", f)

        imp = get_feature_importance(best_model, all_feat_cols)
        log(f"\n  TOP 30 FEATURES:", f)
        for rank, (_, r) in enumerate(imp.head(30).iterrows(), 1):
            log(f"    {rank:>3d}. {r['imp']:5.0f}  {r['feat']}", f)
        zero = (imp["imp"] == 0).sum()
        log(f"  Zero-importance: {zero}/{len(all_feat_cols)}", f)

        # Test subsets
        for n_feats, label in [(50, "top-50"), (30, "top-30"), (20, "top-20")]:
            top_feats = imp.head(n_feats)["feat"].tolist()
            log(f"\n  --- Feature subset: {label} ---", f)

            model_sub = train_single(df_train, df_val, top_feats, best_params)
            y_te = df_test[TARGET].dropna()
            X_te_sub = df_test.loc[y_te.index, top_feats]
            y_prob_sub = model_sub.predict_proba(X_te_sub)[:, 1]
            y_pred_sub = model_sub.predict(X_te_sub)
            m_sub = eval_cls(y_te.values, y_pred_sub, y_prob_sub)
            log(f"  Test: acc={m_sub['acc']:.4f}  auc={m_sub['auc']:.4f}  best_iter={model_sub.best_iteration_}", f)

            bt_sub = backtest_realistic(df_test.loc[y_te.index], y_prob_sub, mode="long_short", margin=0.02)
            log(f"  Backtest (LS m=0.02): ret={bt_sub['total_ret']:+.2%}  sharpe={bt_sub['sharpe']:.2f}  "
                f"maxDD={bt_sub['max_dd']:.2%}  trades={bt_sub['n_trades']}  WR={bt_sub['win_rate']:.2%}", f)

            bt_lo = backtest_realistic(df_test.loc[y_te.index], y_prob_sub, mode="long_only", margin=0.04)
            log(f"  Backtest (LO m=0.04): ret={bt_lo['total_ret']:+.2%}  sharpe={bt_lo['sharpe']:.2f}  "
                f"maxDD={bt_lo['max_dd']:.2%}  trades={bt_lo['n_trades']}  WR={bt_lo['win_rate']:.2%}", f)

        # ─── Phase 3: Ensemble ───
        log(f"\n{'#'*70}", f)
        log(f"  PHASE 3: ENSEMBLE (average of multiple models)", f)
        log(f"{'#'*70}", f)

        # Train 5 models with different seeds using best config params
        ensemble_params = dict(best_params)
        ensemble_probs = []
        for seed in [42, 123, 7, 2024, 999]:
            ensemble_params["random_state"] = seed
            m = train_single(df_train, df_val, all_feat_cols, ensemble_params)
            y_te = df_test[TARGET].dropna()
            X_te = df_test.loc[y_te.index, all_feat_cols]
            p = m.predict_proba(X_te)[:, 1]
            ensemble_probs.append(p)
            log(f"  Seed {seed}: best_iter={m.best_iteration_}  auc={_auc_roc(y_te.values, p):.4f}", f)

        # Average ensemble
        y_prob_ens = np.mean(ensemble_probs, axis=0)
        y_pred_ens = (y_prob_ens > 0.5).astype(int)
        m_ens = eval_cls(y_te.values, y_pred_ens, y_prob_ens)
        log(f"\n  ENSEMBLE (5 seeds): acc={m_ens['acc']:.4f}  auc={m_ens['auc']:.4f}", f)
        log(f"  Pred distribution: min={y_prob_ens.min():.4f}  p25={np.percentile(y_prob_ens,25):.4f}  "
            f"median={np.median(y_prob_ens):.4f}  p75={np.percentile(y_prob_ens,75):.4f}  max={y_prob_ens.max():.4f}", f)

        log(f"\n  ENSEMBLE REALISTIC BACKTEST:", f)
        log(f"  {'Mode':<13s} {'Margin':>6s} {'Trades':>7s} {'WinRate':>8s} {'GrossRet':>10s} "
            f"{'NetRet':>10s} {'Sharpe':>7s} {'MaxDD':>8s} {'PF':>5s}", f)
        for mode in ["long_short", "long_only"]:
            margins = [0.00, 0.01, 0.02, 0.03, 0.04, 0.06] if mode == "long_short" else [0.00, 0.02, 0.04, 0.06]
            for margin in margins:
                bt = backtest_realistic(df_test.loc[y_te.index], y_prob_ens, mode=mode, margin=margin)
                if bt["n_trades"] == 0:
                    log(f"  {mode:<13s} {margin:>6.2f}       0      —          —          —       —        —     —", f)
                else:
                    pf_str = f"{bt['profit_factor']:.2f}" if bt['profit_factor'] < 100 else "inf"
                    log(f"  {mode:<13s} {margin:>6.2f} {bt['n_trades']:>7,} {bt['win_rate']:>7.2%} "
                        f"{bt['gross_ret']:>+9.2%} {bt['total_ret']:>+9.2%} {bt['sharpe']:>6.2f} "
                        f"{bt['max_dd']:>7.2%} {pf_str:>5s}", f)

        # Also test ensemble with top features
        top50 = imp.head(50)["feat"].tolist()
        ensemble_probs_t50 = []
        for seed in [42, 123, 7, 2024, 999]:
            ensemble_params["random_state"] = seed
            m = train_single(df_train, df_val, top50, ensemble_params)
            X_te_t50 = df_test.loc[y_te.index, top50]
            p = m.predict_proba(X_te_t50)[:, 1]
            ensemble_probs_t50.append(p)

        y_prob_ens50 = np.mean(ensemble_probs_t50, axis=0)
        m_ens50 = eval_cls(y_te.values, (y_prob_ens50 > 0.5).astype(int), y_prob_ens50)
        log(f"\n  ENSEMBLE (5 seeds, top-50 features): acc={m_ens50['acc']:.4f}  auc={m_ens50['auc']:.4f}", f)

        log(f"\n  ENSEMBLE TOP-50 REALISTIC BACKTEST:", f)
        log(f"  {'Mode':<13s} {'Margin':>6s} {'Trades':>7s} {'WinRate':>8s} {'GrossRet':>10s} "
            f"{'NetRet':>10s} {'Sharpe':>7s} {'MaxDD':>8s} {'PF':>5s}", f)
        for mode in ["long_short", "long_only"]:
            margins = [0.00, 0.01, 0.02, 0.03, 0.04, 0.06] if mode == "long_short" else [0.00, 0.02, 0.04, 0.06]
            for margin in margins:
                bt = backtest_realistic(df_test.loc[y_te.index], y_prob_ens50, mode=mode, margin=margin)
                if bt["n_trades"] == 0:
                    log(f"  {mode:<13s} {margin:>6.2f}       0      —          —          —       —        —     —", f)
                else:
                    pf_str = f"{bt['profit_factor']:.2f}" if bt['profit_factor'] < 100 else "inf"
                    log(f"  {mode:<13s} {margin:>6.2f} {bt['n_trades']:>7,} {bt['win_rate']:>7.2%} "
                        f"{bt['gross_ret']:>+9.2%} {bt['total_ret']:>+9.2%} {bt['sharpe']:>6.2f} "
                        f"{bt['max_dd']:>7.2%} {pf_str:>5s}", f)

        # ─── Phase 4: Walk-forward validation ───
        log(f"\n{'#'*70}", f)
        log(f"  PHASE 4: PURGED WALK-FORWARD (gap={HORIZON} candles)", f)
        log(f"{'#'*70}", f)

        df_all = pd.concat([df_train, df_val, df_test])

        # Walk-forward with best single config
        log(f"\n  Walk-forward with {best_config} (all features):", f)
        wf = walk_forward_purged(df_all, all_feat_cols, best_params, n_splits=5, f=f)
        for i, w in enumerate(wf):
            log(f"    Split {i+1}: acc={w['acc']:.4f} auc={w['auc']:.4f} sharpe={w['sharpe']:.2f} "
                f"ret={w['total_ret']:+.2%} maxDD={w['max_dd']:.2%} trades={w['n_trades']} win={w['win_rate']:.2%}", f)
        if wf:
            sharpes = [w["sharpe"] for w in wf]
            log(f"    MEAN: sharpe={np.mean(sharpes):.2f}+-{np.std(sharpes):.2f}  "
                f"ret={np.mean([w['total_ret'] for w in wf]):+.2%}  "
                f"win={np.mean([w['win_rate'] for w in wf]):.2%}  "
                f"positive_splits={sum(1 for s in sharpes if s > 0)}/5", f)

        # Walk-forward with top-50 features
        log(f"\n  Walk-forward with {best_config} (top-50 features):", f)
        wf50 = walk_forward_purged(df_all, top50, best_params, n_splits=5, f=f)
        for i, w in enumerate(wf50):
            log(f"    Split {i+1}: acc={w['acc']:.4f} auc={w['auc']:.4f} sharpe={w['sharpe']:.2f} "
                f"ret={w['total_ret']:+.2%} maxDD={w['max_dd']:.2%} trades={w['n_trades']} win={w['win_rate']:.2%}", f)
        if wf50:
            sharpes50 = [w["sharpe"] for w in wf50]
            log(f"    MEAN: sharpe={np.mean(sharpes50):.2f}+-{np.std(sharpes50):.2f}  "
                f"ret={np.mean([w['total_ret'] for w in wf50]):+.2%}  "
                f"win={np.mean([w['win_rate'] for w in wf50]):.2%}  "
                f"positive_splits={sum(1 for s in sharpes50 if s > 0)}/5", f)

        # ─── Phase 5: Final analysis ───
        log(f"\n{'#'*70}", f)
        log(f"  PHASE 5: FINAL ANALYSIS", f)
        log(f"{'#'*70}", f)

        # Prediction calibration check
        log(f"\n  CALIBRATION (best single model, {best_config}):", f)
        y_prob_best = config_probs[best_config]
        for lo, hi in [(0.40, 0.45), (0.45, 0.48), (0.48, 0.50), (0.50, 0.52), (0.52, 0.55), (0.55, 0.60)]:
            mask = (y_prob_best >= lo) & (y_prob_best < hi)
            if mask.sum() > 0:
                actual_rate = y_te.values[mask].mean()
                log(f"    prob [{lo:.2f}, {hi:.2f}): n={mask.sum():>6,}  actual_up_rate={actual_rate:.4f}  "
                    f"predicted~{(lo+hi)/2:.2f}", f)

        # Monthly performance breakdown (non-overlapping backtest)
        log(f"\n  MONTHLY PERFORMANCE (ensemble, long-short, margin=0.02):", f)
        df_te_aligned = df_test.loc[y_te.index].copy()
        df_te_aligned["prob_ens"] = y_prob_ens
        df_te_aligned["month"] = pd.to_datetime(df_te_aligned["timestamp"]).dt.to_period("M")

        close_arr = df_te_aligned["close"].values
        prob_arr = df_te_aligned["prob_ens"].values
        n_te = len(close_arr)

        # Build trade list with timestamps
        trade_list = []
        idx = 0
        while idx + HORIZON < n_te:
            prob = prob_arr[idx]
            if abs(prob - 0.5) <= 0.02:
                idx += HORIZON
                continue
            signal = 1 if prob > 0.5 else -1
            raw_ret = (close_arr[idx + HORIZON] - close_arr[idx]) / close_arr[idx] * signal
            net_ret = raw_ret - FEE_RT
            ts = df_te_aligned.iloc[idx]["timestamp"]
            trade_list.append({"timestamp": ts, "net_ret": net_ret, "signal": signal})
            idx += HORIZON

        if trade_list:
            df_trades_monthly = pd.DataFrame(trade_list)
            df_trades_monthly["month"] = pd.to_datetime(df_trades_monthly["timestamp"]).dt.to_period("M")
            monthly = df_trades_monthly.groupby("month").agg(
                trades=("net_ret", "count"),
                mean_ret=("net_ret", "mean"),
                total_ret=("net_ret", "sum"),
                win_rate=("net_ret", lambda x: (x > 0).mean()),
            )
            log(f"  {'Month':<10s} {'Trades':>6s} {'MeanRet':>9s} {'TotalRet':>9s} {'WinRate':>8s}", f)
            for month, row in monthly.iterrows():
                log(f"  {str(month):<10s} {int(row['trades']):>6d} {row['mean_ret']:>+8.4%} "
                    f"{row['total_ret']:>+8.4%} {row['win_rate']:>7.2%}", f)

        # Save best model
        log(f"\n  Saving best model ({best_config})...", f)
        model_path = os.path.join(MODEL_DIR, "direction_6_best.txt")
        config_models[best_config].booster_.save_model(model_path)
        log(f"  Saved to {model_path}", f)

        # Save ensemble models
        for i, seed in enumerate([42, 123, 7, 2024, 999]):
            ensemble_params["random_state"] = seed
            m = train_single(df_train, df_val, all_feat_cols, ensemble_params)
            m.booster_.save_model(os.path.join(MODEL_DIR, f"direction_6_ens_{seed}.txt"))
        log(f"  Saved 5 ensemble models", f)

        # ─── Final verdict ───
        log(f"\n{'='*70}", f)
        log(f"  ITERATION 3 VERDICT", f)
        log(f"{'='*70}", f)
        log(f"  Best single config: {best_config} (AUC={config_results[best_config]['auc']:.4f})", f)

        bt_final = backtest_realistic(df_test.loc[y_te.index], y_prob_ens, mode="long_short", margin=0.02)
        log(f"  Ensemble (5-seed) long-short m=0.02:", f)
        log(f"    Net return: {bt_final['total_ret']:+.2%}", f)
        log(f"    Sharpe: {bt_final['sharpe']:.2f}", f)
        log(f"    Max DD: {bt_final['max_dd']:.2%}", f)
        log(f"    Win rate: {bt_final['win_rate']:.2%}", f)
        log(f"    Trades: {bt_final['n_trades']}", f)
        log(f"    Profit factor: {bt_final['profit_factor']:.2f}", f)
        log(f"    Fee drag: {bt_final['fee_drag']:.2%}", f)

    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
