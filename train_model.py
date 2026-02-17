"""
BTC LightGBM Training Pipeline — Iteration 5
==============================================
Focus: ROBUSTNESS — fix walk-forward failure from Iter 4.

Key changes from Iter 4:
  1. Expanding-window walk-forward that tests ACTUAL winning configs
     (long_only, maker fees, high confidence filtering)
  2. Intermediate horizons: 2h (24) and 3h (36) candles
  3. Probability calibration (isotonic regression on val set)
  4. Multi-horizon consensus: trade only when 2+ horizons agree
  5. Stronger regularization for regime robustness
  6. Proper purge embargo (2x horizon gap between train/test)

Usage:
  python train_model.py
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
FEATURES_CSV = os.path.join(OUTPUT_DIR, "btc_features_5m.csv")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "training_results_v5.txt")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

FEE_TAKER_RT = 0.0008  # 0.08% round-trip
FEE_MAKER_RT = 0.0004  # 0.04% round-trip


def log(msg, f=None):
    print(msg)
    if f:
        f.write(msg + "\n")
        f.flush()


def auc_roc(y_true, y_prob):
    """Manual AUC-ROC."""
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
    auc = auc_roc(y_true, y_prob)
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


# ═══════════════════════════════════════════════════════════════════════════
# BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════

def backtest(df_test, y_prob, horizon, fee_rt=FEE_TAKER_RT,
             mode="long_only", margin=0.0, top_pct=None):
    """Non-overlapping backtest with transaction costs and confidence filtering."""
    close = df_test["close"].values
    n = len(close)

    # Collect all potential trades first (before filtering)
    all_trades = []
    i = 0
    while i + horizon < n:
        prob = y_prob[i]
        entry = close[i]
        exit_price = close[i + horizon]

        if mode == "long_only":
            if prob > 0.5 + margin:
                raw_ret = (exit_price - entry) / entry
                net_ret = raw_ret - fee_rt
                all_trades.append({"idx": i, "prob": prob, "net_ret": net_ret,
                                   "raw_ret": raw_ret, "signal": 1})
        elif mode == "long_short":
            if abs(prob - 0.5) > margin:
                signal = 1 if prob > 0.5 else -1
                raw_ret = (exit_price - entry) / entry * signal
                net_ret = raw_ret - fee_rt
                all_trades.append({"idx": i, "prob": prob, "net_ret": net_ret,
                                   "raw_ret": raw_ret, "signal": signal})
        i += horizon

    if not all_trades:
        return _empty_bt()

    df_t = pd.DataFrame(all_trades)

    # Confidence filtering: keep top X% most confident
    if top_pct is not None and len(df_t) > 5:
        abs_conf = abs(df_t["prob"] - 0.5)
        threshold = abs_conf.quantile(1.0 - top_pct)
        df_t = df_t[abs_conf >= threshold]

    if len(df_t) == 0:
        return _empty_bt()

    return _compute_bt_stats(df_t, close, horizon, fee_rt)


def _empty_bt():
    return {"total_ret": 0.0, "sharpe": 0.0, "max_dd": 0.0, "win_rate": 0.0,
            "n_trades": 0, "avg_ret": 0.0, "profit_factor": 0.0,
            "gross_ret": 0.0, "kelly": 0.0, "calmar": 0.0}


def _compute_bt_stats(df_t, close, horizon, fee_rt):
    cum = (1 + df_t["net_ret"]).cumprod()
    total_ret = cum.iloc[-1] - 1

    candles_per_year = 365.25 * 24 * 12
    trades_per_year = candles_per_year / horizon
    mu = df_t["net_ret"].mean()
    sigma = df_t["net_ret"].std()
    sharpe = (mu / sigma) * np.sqrt(trades_per_year) if sigma > 0 else 0

    cummax = cum.cummax()
    max_dd = ((cum - cummax) / cummax).min()

    win_rate = (df_t["net_ret"] > 0).mean()

    gross_wins = df_t.loc[df_t["net_ret"] > 0, "net_ret"].sum()
    gross_losses = abs(df_t.loc[df_t["net_ret"] < 0, "net_ret"].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    gross_ret = (1 + df_t["raw_ret"]).cumprod().iloc[-1] - 1

    avg_win = df_t.loc[df_t["net_ret"] > 0, "net_ret"].mean() if (df_t["net_ret"] > 0).any() else 0
    avg_loss = abs(df_t.loc[df_t["net_ret"] < 0, "net_ret"].mean()) if (df_t["net_ret"] < 0).any() else 1
    b = avg_win / avg_loss if avg_loss > 0 else 0
    p = win_rate
    kelly = (p * b - (1 - p)) / b if b > 0 else 0
    calmar = total_ret / abs(max_dd) if abs(max_dd) > 1e-8 else 0

    return {
        "total_ret": total_ret, "sharpe": sharpe, "max_dd": max_dd,
        "win_rate": win_rate, "n_trades": len(df_t), "avg_ret": mu,
        "profit_factor": profit_factor, "gross_ret": gross_ret,
        "kelly": kelly, "calmar": calmar,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PROBABILITY CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def calibrate_isotonic(y_val, prob_val, prob_test):
    """Isotonic regression calibration: fit on val, apply to test."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(prob_val, y_val)
    return ir.transform(prob_test)


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(X_tr, y_tr, X_va, y_va, params):
    """Train a single LightGBM model."""
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


# ═══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD (EXPANDING WINDOW)
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_expanding(df, feat_cols, target, params, horizon, n_splits=6,
                           test_frac=0.08, f=None, calibrate=True):
    """
    Expanding-window walk-forward with:
    - ALL data up to split point as training
    - Purge gap of 2*horizon between train and test
    - Isotonic calibration on last 15% of train as val
    - Tests multiple backtest configs (the ACTUAL winning ones)
    """
    n = len(df)
    test_size = int(n * test_frac)
    purge_gap = horizon * 2
    results = []

    for i in range(n_splits):
        # Test window: from back to front
        te_end = n - i * test_size
        te_start = te_end - test_size

        if te_start < 0:
            break

        # Train: everything up to purge gap before test
        tr_end = te_start - purge_gap

        if tr_end < test_size:  # Need minimum training data
            break

        df_te = df.iloc[te_start:te_end]
        df_tr_full = df.iloc[:tr_end]

        y_tr_full = df_tr_full[target].dropna()
        y_te = df_te[target].dropna()

        if len(y_tr_full) < 5000 or len(y_te) < 500:
            continue

        X_tr_full = df_tr_full.loc[y_tr_full.index, feat_cols]
        X_te = df_te.loc[y_te.index, feat_cols]

        # Split training into train/val (last 15% as val for early stopping + calibration)
        val_cut = int(len(X_tr_full) * 0.85)
        X_tr = X_tr_full.iloc[:val_cut]
        y_tr = y_tr_full.iloc[:val_cut]
        X_va = X_tr_full.iloc[val_cut:]
        y_va = y_tr_full.iloc[val_cut:]

        t0 = time.time()
        model = train_model(X_tr, y_tr, X_va, y_va, params)
        elapsed = time.time() - t0

        # Raw predictions
        y_prob_test = model.predict_proba(X_te)[:, 1]
        y_prob_val = model.predict_proba(X_va)[:, 1]

        # Calibrate
        if calibrate:
            y_prob_cal = calibrate_isotonic(y_va.values, y_prob_val, y_prob_test)
        else:
            y_prob_cal = y_prob_test

        raw_auc = auc_roc(y_te.values, y_prob_test)
        cal_auc = auc_roc(y_te.values, y_prob_cal) if calibrate else raw_auc

        # Backtest with MULTIPLE configs (test the winning ones!)
        bt_results = {}
        configs = [
            # The configs that worked in Iter 4 test set
            ("maker_LO_m04_top10", FEE_MAKER_RT, "long_only", 0.04, 0.10),
            ("maker_LO_m06_top10", FEE_MAKER_RT, "long_only", 0.06, 0.10),
            ("maker_LO_m04_top20", FEE_MAKER_RT, "long_only", 0.04, 0.20),
            ("maker_LO_m06_top20", FEE_MAKER_RT, "long_only", 0.06, 0.20),
            ("maker_LO_m02_top30", FEE_MAKER_RT, "long_only", 0.02, 0.30),
            ("maker_LO_m04_all", FEE_MAKER_RT, "long_only", 0.04, None),
            ("maker_LS_m02_all", FEE_MAKER_RT, "long_short", 0.02, None),
            ("taker_LO_m06_top10", FEE_TAKER_RT, "long_only", 0.06, 0.10),
            # Calibrated versions
            ("cal_maker_LO_m04_top10", FEE_MAKER_RT, "long_only", 0.04, 0.10),
            ("cal_maker_LO_m06_top10", FEE_MAKER_RT, "long_only", 0.06, 0.10),
            ("cal_maker_LO_m04_top20", FEE_MAKER_RT, "long_only", 0.04, 0.20),
            ("cal_maker_LO_m02_top30", FEE_MAKER_RT, "long_only", 0.02, 0.30),
        ]

        for name, fee, mode, margin, top_pct in configs:
            use_cal = name.startswith("cal_") and calibrate
            prob = y_prob_cal if use_cal else y_prob_test
            bt = backtest(df_te.loc[y_te.index], prob, horizon,
                          fee_rt=fee, mode=mode, margin=margin, top_pct=top_pct)
            bt_results[name] = bt

        result = {
            "split": i + 1,
            "train_size": len(X_tr),
            "val_size": len(X_va),
            "test_size": len(y_te),
            "raw_auc": raw_auc,
            "cal_auc": cal_auc,
            "best_iter": model.best_iteration_,
            "elapsed": elapsed,
            "bt": bt_results,
            "model": model,
            "te_start": te_start,
            "te_end": te_end,
        }
        results.append(result)

        if f:
            log(f"  Split {i+1}: train={len(X_tr):,} val={len(X_va):,} test={len(y_te):,}  "
                f"raw_auc={raw_auc:.4f}  cal_auc={cal_auc:.4f}  iter={model.best_iteration_}  "
                f"time={elapsed:.0f}s", f)
            # Show best BT configs
            for name, bt in bt_results.items():
                if bt["n_trades"] >= 3:
                    log(f"    {name}: trades={bt['n_trades']}  net={bt['total_ret']:+.2%}  "
                        f"sharpe={bt['sharpe']:.2f}  WR={bt['win_rate']:.1%}  "
                        f"PF={bt['profit_factor']:.2f}  kelly={bt['kelly']:.3f}", f)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-HORIZON CONSENSUS
# ═══════════════════════════════════════════════════════════════════════════

def multi_horizon_consensus_backtest(df_test, models, horizons, feat_cols,
                                     fee_rt=FEE_MAKER_RT, min_agree=2,
                                     margin=0.04, execution_horizon=None):
    """
    Trade only when multiple horizon models agree on direction.

    execution_horizon: which horizon to use for holding period (default: shortest)
    """
    if execution_horizon is None:
        execution_horizon = min(horizons)

    close = df_test["close"].values
    n = len(close)

    # Get predictions from all models
    probs = {}
    for h, model in zip(horizons, models):
        target = f"target_direction_{h}"
        y_te = df_test[target].dropna()
        X_te = df_test.loc[y_te.index, feat_cols]
        prob = model.predict_proba(X_te)[:, 1]
        # Create full-length array with NaN
        full_prob = np.full(n, np.nan)
        for j, idx in enumerate(y_te.index):
            pos = df_test.index.get_loc(idx)
            if isinstance(pos, int):
                full_prob[pos] = prob[j]
        probs[h] = full_prob

    trades = []
    i = 0
    while i + execution_horizon < n:
        # Count how many horizons are bullish
        bullish = 0
        bearish = 0
        valid = 0
        for h in horizons:
            p = probs[h][i]
            if np.isnan(p):
                continue
            valid += 1
            if p > 0.5 + margin:
                bullish += 1
            elif p < 0.5 - margin:
                bearish += 1

        if valid >= min_agree and bullish >= min_agree:
            # Long signal: multiple horizons agree bullish
            entry = close[i]
            exit_price = close[i + execution_horizon]
            raw_ret = (exit_price - entry) / entry
            net_ret = raw_ret - fee_rt
            trades.append({"idx": i, "net_ret": net_ret, "raw_ret": raw_ret,
                           "signal": 1, "prob": bullish / valid,
                           "bullish": bullish, "bearish": bearish})

        i += execution_horizon

    if not trades:
        return _empty_bt()

    df_t = pd.DataFrame(trades)
    return _compute_bt_stats(df_t, close, execution_horizon, fee_rt)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    if os.path.exists(FEATURES_FILE):
        print(f"Loading {FEATURES_FILE}...")
        df = pd.read_parquet(FEATURES_FILE)
    elif os.path.exists(FEATURES_CSV):
        print(f"Loading {FEATURES_CSV}...")
        df = pd.read_csv(FEATURES_CSV)
    else:
        print("ERROR: No features file found. Run btc_indicators.py first.")
        return

    print(f"  {len(df):,} rows x {len(df.columns)} cols")

    all_feat_cols = get_feature_cols(df)
    df_train, df_val, df_test = temporal_split(df)
    df_all = pd.concat([df_train, df_val, df_test])

    # Stronger regularization for Iter 5 (prevent regime overfitting)
    base_params = {
        "objective": "binary", "metric": "binary_logloss",
        "boosting_type": "gbdt", "n_estimators": 5000,
        "learning_rate": 0.005, "num_leaves": 24, "max_depth": 5,
        "min_child_samples": 500, "subsample": 0.5, "subsample_freq": 1,
        "colsample_bytree": 0.3, "colsample_bynode": 0.5,
        "reg_alpha": 5.0, "reg_lambda": 20.0,
        "feature_fraction_bynode": 0.5,
        "path_smooth": 10.0,
        "random_state": 42, "verbose": -1, "n_jobs": -1,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        log(f"BTC LightGBM Training — Iteration 5", f)
        log(f"{'='*80}", f)
        log(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", f)
        log(f"Data: {len(df):,} rows x {len(df.columns)} cols  Features: {len(all_feat_cols)}", f)
        log(f"Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}", f)
        log(f"Fees: taker={FEE_TAKER_RT*100:.2f}% RT, maker={FEE_MAKER_RT*100:.2f}% RT", f)
        log(f"\nKey changes from Iter 4:", f)
        log(f"  - Expanding-window walk-forward (all data up to split point)", f)
        log(f"  - WF tests ACTUAL winning configs (maker, long_only, high conf)", f)
        log(f"  - Isotonic calibration on validation set", f)
        log(f"  - Added 2h and 3h horizons", f)
        log(f"  - Stronger regularization (min_child=500, leaves=24, depth=5)", f)
        log(f"  - Multi-horizon consensus trading", f)
        log(f"  - Purge gap = 2*horizon between train and test", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: MULTI-HORIZON DIRECTION MODELS
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 1: MULTI-HORIZON DIRECTION MODELS", f)
        log(f"{'#'*80}", f)

        horizons = [
            (12, "1h"),
            (24, "2h"),
            (36, "3h"),
            (48, "4h"),
            (96, "8h"),
        ]

        horizon_results = {}
        for horizon, label in horizons:
            target = f"target_direction_{horizon}"
            if target not in df.columns:
                log(f"\n  SKIP {label} ({target}) — not in dataset", f)
                continue

            log(f"\n{'='*80}", f)
            log(f"  DIRECTION {label} (horizon={horizon}, target={target})", f)
            log(f"{'='*80}", f)

            y = df[target].dropna()
            pos_rate = y.mean()
            log(f"  Class balance: {pos_rate:.4f} positive ({len(y):,} samples)", f)

            # --- Train on fixed split ---
            t0 = time.time()
            y_tr = df_train[target].dropna()
            y_va = df_val[target].dropna()
            X_tr = df_train.loc[y_tr.index, all_feat_cols]
            X_va = df_val.loc[y_va.index, all_feat_cols]
            model = train_model(X_tr, y_tr, X_va, y_va, base_params)
            elapsed = time.time() - t0
            log(f"  Trained in {elapsed:.1f}s  best_iter={model.best_iteration_}", f)

            # Test set evaluation
            y_te = df_test[target].dropna()
            X_te = df_test.loc[y_te.index, all_feat_cols]
            y_prob = model.predict_proba(X_te)[:, 1]
            y_pred = model.predict(X_te)
            metrics = eval_cls(y_te.values, y_pred, y_prob)

            # Calibrate on val
            y_prob_val = model.predict_proba(X_va)[:, 1]
            y_prob_cal = calibrate_isotonic(y_va.values, y_prob_val, y_prob)

            cal_metrics = eval_cls(y_te.values, (y_prob_cal > 0.5).astype(int), y_prob_cal)

            log(f"  Test: acc={metrics['acc']:.4f}  auc={metrics['auc']:.4f}  "
                f"cal_auc={cal_metrics['auc']:.4f}", f)
            log(f"  Raw pred spread: p5={np.percentile(y_prob,5):.4f}  p50={np.median(y_prob):.4f}  "
                f"p95={np.percentile(y_prob,95):.4f}", f)
            log(f"  Cal pred spread: p5={np.percentile(y_prob_cal,5):.4f}  p50={np.median(y_prob_cal):.4f}  "
                f"p95={np.percentile(y_prob_cal,95):.4f}", f)

            # Calibration check
            log(f"\n  Calibration table:", f)
            for lo, hi in [(0.40, 0.45), (0.45, 0.48), (0.48, 0.50),
                           (0.50, 0.52), (0.52, 0.55), (0.55, 0.60), (0.60, 0.70)]:
                for prob_name, prob_arr in [("raw", y_prob), ("cal", y_prob_cal)]:
                    mask = (prob_arr >= lo) & (prob_arr < hi)
                    if mask.sum() > 20:
                        actual = y_te.values[mask].mean()
                        log(f"    {prob_name} [{lo:.2f},{hi:.2f}): n={mask.sum():>6,}  "
                            f"actual={actual:.4f}  expected~{(lo+hi)/2:.2f}  "
                            f"edge={actual-(lo+hi)/2:+.4f}", f)

            # Backtest matrix (condensed — focus on winning configs)
            log(f"\n  Backtest matrix:", f)
            log(f"  {'Config':<35s} {'Trades':>7s} {'WR':>6s} {'Net':>9s} "
                f"{'Sharpe':>7s} {'MaxDD':>8s} {'PF':>6s} {'Kelly':>6s} {'Calmar':>7s}", f)

            best_scenario = None
            best_sharpe = -999

            for prob_name, prob_arr in [("raw", y_prob), ("cal", y_prob_cal)]:
                for fee_label, fee_rt in [("maker", FEE_MAKER_RT), ("taker", FEE_TAKER_RT)]:
                    for mode in ["long_only", "long_short"]:
                        for margin in [0.00, 0.02, 0.04, 0.06]:
                            for top_pct in [None, 0.3, 0.2, 0.1]:
                                bt = backtest(
                                    df_test.loc[y_te.index], prob_arr, horizon,
                                    fee_rt=fee_rt, mode=mode, margin=margin, top_pct=top_pct,
                                )
                                if bt["n_trades"] < 5:
                                    continue

                                top_str = f"top{top_pct:.0%}" if top_pct else "all"
                                config_name = f"{prob_name}_{fee_label}_{mode[:2]}_m{margin:.2f}_{top_str}"
                                pf_str = f"{bt['profit_factor']:.2f}" if bt['profit_factor'] < 100 else "inf"

                                # Only print promising configs
                                if bt["total_ret"] > 0 or (bt["n_trades"] >= 50 and margin == 0):
                                    log(f"  {config_name:<35s} {bt['n_trades']:>7,} {bt['win_rate']:>5.1%} "
                                        f"{bt['total_ret']:>+8.2%} {bt['sharpe']:>6.2f} "
                                        f"{bt['max_dd']:>7.2%} {pf_str:>6s} {bt['kelly']:>5.3f} "
                                        f"{bt['calmar']:>6.2f}", f)

                                if bt["sharpe"] > best_sharpe and bt["total_ret"] > 0 and bt["n_trades"] >= 5:
                                    best_sharpe = bt["sharpe"]
                                    best_scenario = {"config": config_name, **bt}

            if best_scenario:
                log(f"\n  BEST {label}: {best_scenario['config']}  net={best_scenario['total_ret']:+.2%}  "
                    f"sharpe={best_scenario['sharpe']:.2f}  trades={best_scenario['n_trades']}", f)
            else:
                log(f"\n  BEST {label}: NO PROFITABLE SCENARIO", f)

            # Feature importance
            imp = get_feature_importance(model, all_feat_cols)
            log(f"\n  Top 15 features:", f)
            for rank, (_, r) in enumerate(imp.head(15).iterrows(), 1):
                log(f"    {rank:>3d}. {r['imp']:5.0f}  {r['feat']}", f)

            horizon_results[label] = {
                "metrics": metrics, "best_scenario": best_scenario,
                "model": model, "y_prob": y_prob, "y_prob_cal": y_prob_cal,
                "y_te": y_te, "horizon": horizon, "importance": imp,
            }

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: WALK-FORWARD VALIDATION (THE TRUTH TEST)
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 2: EXPANDING-WINDOW WALK-FORWARD VALIDATION", f)
        log(f"{'#'*80}", f)

        wf_results = {}
        for label, data in horizon_results.items():
            horizon = data["horizon"]
            target = f"target_direction_{horizon}"
            imp = data["importance"]

            # Use top-80 features for WF (slight regularization via feature selection)
            n_feats = min(80, len(all_feat_cols))
            wf_feats = imp.head(n_feats)["feat"].tolist()

            log(f"\n{'='*80}", f)
            log(f"  WALK-FORWARD: {label} (horizon={horizon}, top-{n_feats} features)", f)
            log(f"{'='*80}", f)

            wf = walk_forward_expanding(
                df_all, wf_feats, target, base_params, horizon,
                n_splits=6, test_frac=0.08, f=f, calibrate=True,
            )

            if not wf:
                log(f"  No valid walk-forward splits", f)
                continue

            # Aggregate WF results
            log(f"\n  WALK-FORWARD SUMMARY for {label}:", f)
            log(f"  {'Config':<35s} {'Pos/Total':>10s} {'Avg Net':>9s} {'Avg Sharpe':>11s} "
                f"{'Med Trades':>11s} {'Avg WR':>7s}", f)

            # Collect results for each config
            config_names = set()
            for w in wf:
                config_names.update(w["bt"].keys())

            config_scores = {}
            for config_name in sorted(config_names):
                nets = []
                sharpes = []
                trades_list = []
                wrs = []
                for w in wf:
                    bt = w["bt"].get(config_name, _empty_bt())
                    if bt["n_trades"] > 0:
                        nets.append(bt["total_ret"])
                        sharpes.append(bt["sharpe"])
                        trades_list.append(bt["n_trades"])
                        wrs.append(bt["win_rate"])

                if len(nets) == 0:
                    continue

                n_positive = sum(1 for n in nets if n > 0)
                avg_net = np.mean(nets)
                avg_sharpe = np.mean(sharpes)
                med_trades = np.median(trades_list)
                avg_wr = np.mean(wrs)

                ratio_str = f"{n_positive}/{len(nets)}"
                log(f"  {config_name:<35s} {ratio_str:>10s} {avg_net:>+8.2%} "
                    f"{avg_sharpe:>10.2f} {med_trades:>10.0f} {avg_wr:>6.1%}", f)

                config_scores[config_name] = {
                    "n_positive": n_positive, "n_splits": len(nets),
                    "avg_net": avg_net, "avg_sharpe": avg_sharpe,
                    "med_trades": med_trades, "avg_wr": avg_wr,
                }

            # Find best WF config (prioritize positive splits, then avg_net)
            best_wf_config = None
            best_wf_score = -999
            for name, scores in config_scores.items():
                # Score: positive_ratio * 10 + avg_net * 100
                score = (scores["n_positive"] / scores["n_splits"]) * 10 + scores["avg_net"] * 100
                if score > best_wf_score:
                    best_wf_score = score
                    best_wf_config = name

            if best_wf_config:
                s = config_scores[best_wf_config]
                log(f"\n  BEST WF CONFIG for {label}: {best_wf_config}", f)
                log(f"    Positive splits: {s['n_positive']}/{s['n_splits']}", f)
                log(f"    Avg net return: {s['avg_net']:+.2%}", f)
                log(f"    Avg Sharpe: {s['avg_sharpe']:.2f}", f)
                log(f"    Avg win rate: {s['avg_wr']:.1%}", f)

            # Also report AUC across splits
            raw_aucs = [w["raw_auc"] for w in wf]
            cal_aucs = [w["cal_auc"] for w in wf]
            log(f"\n  AUC across splits:", f)
            log(f"    Raw: {' '.join(f'{a:.4f}' for a in raw_aucs)}  "
                f"mean={np.mean(raw_aucs):.4f}  std={np.std(raw_aucs):.4f}", f)
            log(f"    Cal: {' '.join(f'{a:.4f}' for a in cal_aucs)}  "
                f"mean={np.mean(cal_aucs):.4f}  std={np.std(cal_aucs):.4f}", f)

            wf_results[label] = {
                "splits": wf,
                "config_scores": config_scores,
                "best_config": best_wf_config,
                "best_score": config_scores.get(best_wf_config, {}),
            }

        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: MULTI-HORIZON CONSENSUS
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 3: MULTI-HORIZON CONSENSUS TRADING", f)
        log(f"{'#'*80}", f)

        # Find horizons with models available
        available_horizons = [(data["horizon"], data["model"])
                              for label, data in horizon_results.items()
                              if data["model"] is not None]

        if len(available_horizons) >= 2:
            log(f"\n  Available horizons: {[h for h, _ in available_horizons]}", f)

            # Test various combinations
            from itertools import combinations

            y_te_min = df_test[f"target_direction_{min(h for h, _ in available_horizons)}"].dropna()

            for n_models in [2, 3]:
                for combo in combinations(available_horizons, n_models):
                    combo_horizons = [h for h, _ in combo]
                    combo_models = [m for _, m in combo]
                    combo_label = "+".join([f"{h}c" for h in combo_horizons])

                    for exec_h in combo_horizons[:2]:  # Test with different execution horizons
                        for margin in [0.02, 0.04, 0.06]:
                            for fee_label, fee_rt in [("maker", FEE_MAKER_RT)]:
                                bt = multi_horizon_consensus_backtest(
                                    df_test, combo_models, combo_horizons, all_feat_cols,
                                    fee_rt=fee_rt, min_agree=n_models,
                                    margin=margin, execution_horizon=exec_h,
                                )
                                if bt["n_trades"] >= 5 and bt["total_ret"] > 0:
                                    log(f"  {combo_label} exec={exec_h} agree={n_models} m={margin:.2f} {fee_label}: "
                                        f"trades={bt['n_trades']}  net={bt['total_ret']:+.2%}  "
                                        f"sharpe={bt['sharpe']:.2f}  WR={bt['win_rate']:.1%}  "
                                        f"PF={bt['profit_factor']:.2f}", f)

        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: ENSEMBLE (7-seed) FOR BEST HORIZONS
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 4: ENSEMBLE (7-seed) FOR PROMISING HORIZONS", f)
        log(f"{'#'*80}", f)

        # Pick horizons that showed ANY positive WF config
        promising = []
        for label, wf_data in wf_results.items():
            best_score = wf_data.get("best_score", {})
            if best_score and best_score.get("n_positive", 0) >= 2:
                promising.append(label)

        if not promising:
            # Fallback: pick top 2 by test AUC
            by_auc = sorted(horizon_results.items(), key=lambda x: x[1]["metrics"]["auc"], reverse=True)
            promising = [label for label, _ in by_auc[:2]]
            log(f"\n  No horizon passed WF >= 2/6 positive. Using top AUC: {promising}", f)
        else:
            log(f"\n  Promising horizons (>=2 positive WF splits): {promising}", f)

        ensemble_results = {}
        seeds = [42, 123, 7, 2024, 999, 314, 577]

        for label in promising:
            data = horizon_results[label]
            horizon = data["horizon"]
            target = f"target_direction_{horizon}"
            imp = data["importance"]
            n_feats = min(80, len(all_feat_cols))
            ens_feats = imp.head(n_feats)["feat"].tolist()

            log(f"\n  --- Ensemble for {label} (top-{n_feats} features, {len(seeds)} seeds) ---", f)

            y_tr = df_train[target].dropna()
            y_va = df_val[target].dropna()
            y_te = df_test[target].dropna()
            X_tr = df_train.loc[y_tr.index, ens_feats]
            X_va = df_val.loc[y_va.index, ens_feats]
            X_te = df_test.loc[y_te.index, ens_feats]

            ensemble_probs = []
            for seed in seeds:
                p = dict(base_params)
                p["random_state"] = seed
                m = train_model(X_tr, y_tr, X_va, y_va, p)
                prob = m.predict_proba(X_te)[:, 1]
                auc = auc_roc(y_te.values, prob)
                ensemble_probs.append(prob)
                log(f"    Seed {seed}: iter={m.best_iteration_}  auc={auc:.4f}", f)

            y_prob_ens = np.mean(ensemble_probs, axis=0)
            ens_auc = auc_roc(y_te.values, y_prob_ens)
            log(f"    ENSEMBLE AUC={ens_auc:.4f}", f)

            # Calibrate ensemble
            # First get val predictions from all seeds
            ens_val_probs = []
            for seed in seeds:
                p = dict(base_params)
                p["random_state"] = seed
                m = train_model(X_tr, y_tr, X_va, y_va, p)
                ens_val_probs.append(m.predict_proba(X_va)[:, 1])
            y_prob_val_ens = np.mean(ens_val_probs, axis=0)
            y_prob_ens_cal = calibrate_isotonic(y_va.values, y_prob_val_ens, y_prob_ens)
            cal_ens_auc = auc_roc(y_te.values, y_prob_ens_cal)
            log(f"    CALIBRATED ENSEMBLE AUC={cal_ens_auc:.4f}", f)

            # Backtest ensemble
            log(f"\n    Ensemble backtest:", f)
            best_ens = None
            best_ens_sharpe = -999

            for prob_name, prob_arr in [("raw", y_prob_ens), ("cal", y_prob_ens_cal)]:
                for fee_label, fee_rt in [("maker", FEE_MAKER_RT), ("taker", FEE_TAKER_RT)]:
                    for mode in ["long_only"]:
                        for margin in [0.02, 0.04, 0.06]:
                            for top_pct in [None, 0.3, 0.2, 0.1]:
                                bt = backtest(
                                    df_test.loc[y_te.index], prob_arr, horizon,
                                    fee_rt=fee_rt, mode=mode, margin=margin, top_pct=top_pct,
                                )
                                if bt["n_trades"] < 5:
                                    continue

                                top_str = f"top{top_pct:.0%}" if top_pct else "all"
                                config = f"{prob_name}_{fee_label}_LO_m{margin:.2f}_{top_str}"

                                if bt["total_ret"] > 0:
                                    pf_str = f"{bt['profit_factor']:.2f}" if bt['profit_factor'] < 100 else "inf"
                                    log(f"    {config}: trades={bt['n_trades']}  "
                                        f"net={bt['total_ret']:+.2%}  sharpe={bt['sharpe']:.2f}  "
                                        f"WR={bt['win_rate']:.1%}  PF={pf_str}  "
                                        f"kelly={bt['kelly']:.3f}  maxDD={bt['max_dd']:.2%}", f)

                                if bt["sharpe"] > best_ens_sharpe and bt["total_ret"] > 0:
                                    best_ens_sharpe = bt["sharpe"]
                                    best_ens = {"config": config, **bt}

            if best_ens:
                log(f"\n    BEST ENSEMBLE {label}: {best_ens['config']}  "
                    f"net={best_ens['total_ret']:+.2%}  sharpe={best_ens['sharpe']:.2f}", f)

            ensemble_results[label] = {
                "auc": ens_auc, "cal_auc": cal_ens_auc,
                "best": best_ens, "y_prob": y_prob_ens, "y_prob_cal": y_prob_ens_cal,
            }

        # ═══════════════════════════════════════════════════════════════
        # PHASE 5: ENSEMBLE WALK-FORWARD (ULTIMATE ROBUSTNESS TEST)
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  PHASE 5: ENSEMBLE WALK-FORWARD (ROBUSTNESS TEST)", f)
        log(f"{'#'*80}", f)

        for label in promising[:2]:  # Top 2 promising
            data = horizon_results[label]
            horizon = data["horizon"]
            target = f"target_direction_{horizon}"
            imp = data["importance"]
            n_feats = min(80, len(all_feat_cols))
            ens_feats = imp.head(n_feats)["feat"].tolist()

            log(f"\n  --- Ensemble Walk-Forward: {label} ---", f)

            n = len(df_all)
            test_size = int(n * 0.08)
            purge_gap = horizon * 2
            n_splits = 6

            for split_i in range(n_splits):
                te_end = n - split_i * test_size
                te_start = te_end - test_size
                if te_start < 0:
                    break
                tr_end = te_start - purge_gap
                if tr_end < test_size:
                    break

                df_te_wf = df_all.iloc[te_start:te_end]
                df_tr_wf = df_all.iloc[:tr_end]

                y_tr_wf = df_tr_wf[target].dropna()
                y_te_wf = df_te_wf[target].dropna()
                if len(y_tr_wf) < 5000 or len(y_te_wf) < 500:
                    continue

                X_tr_wf = df_tr_wf.loc[y_tr_wf.index, ens_feats]
                X_te_wf = df_te_wf.loc[y_te_wf.index, ens_feats]

                val_cut = int(len(X_tr_wf) * 0.85)
                X_tr_s = X_tr_wf.iloc[:val_cut]
                y_tr_s = y_tr_wf.iloc[:val_cut]
                X_va_s = X_tr_wf.iloc[val_cut:]
                y_va_s = y_tr_wf.iloc[val_cut:]

                # Train 5-seed mini-ensemble per split (faster than 7)
                split_probs = []
                split_val_probs = []
                for seed in seeds[:5]:
                    p = dict(base_params)
                    p["random_state"] = seed
                    m = train_model(X_tr_s, y_tr_s, X_va_s, y_va_s, p)
                    split_probs.append(m.predict_proba(X_te_wf)[:, 1])
                    split_val_probs.append(m.predict_proba(X_va_s)[:, 1])

                prob_ens = np.mean(split_probs, axis=0)
                prob_val_ens = np.mean(split_val_probs, axis=0)

                # Calibrate
                prob_ens_cal = calibrate_isotonic(y_va_s.values, prob_val_ens, prob_ens)

                raw_auc = auc_roc(y_te_wf.values, prob_ens)
                cal_auc = auc_roc(y_te_wf.values, prob_ens_cal)

                log(f"\n  Split {split_i+1}: train={len(X_tr_s):,} val={len(X_va_s):,} "
                    f"test={len(y_te_wf):,}  raw_auc={raw_auc:.4f}  cal_auc={cal_auc:.4f}", f)

                # Test key configs
                for prob_name, prob_arr in [("raw", prob_ens), ("cal", prob_ens_cal)]:
                    for margin in [0.02, 0.04, 0.06]:
                        for top_pct in [None, 0.2, 0.1]:
                            bt = backtest(
                                df_te_wf.loc[y_te_wf.index], prob_arr, horizon,
                                fee_rt=FEE_MAKER_RT, mode="long_only",
                                margin=margin, top_pct=top_pct,
                            )
                            if bt["n_trades"] >= 3:
                                top_str = f"top{top_pct:.0%}" if top_pct else "all"
                                log(f"    {prob_name}_maker_LO_m{margin:.2f}_{top_str}: "
                                    f"trades={bt['n_trades']}  net={bt['total_ret']:+.2%}  "
                                    f"sharpe={bt['sharpe']:.2f}  WR={bt['win_rate']:.1%}", f)

        # ═══════════════════════════════════════════════════════════════
        # FINAL VERDICT
        # ═══════════════════════════════════════════════════════════════
        log(f"\n{'#'*80}", f)
        log(f"  ITERATION 5 — FINAL VERDICT", f)
        log(f"{'#'*80}", f)

        log(f"\n  === Test Set Performance ===", f)
        for label, data in horizon_results.items():
            s = data["best_scenario"]
            auc = data["metrics"]["auc"]
            if s:
                log(f"    {label}: AUC={auc:.4f}  best: {s['config']}  "
                    f"net={s['total_ret']:+.2%}  sharpe={s['sharpe']:.2f}", f)
            else:
                log(f"    {label}: AUC={auc:.4f}  NO PROFITABLE SCENARIO", f)

        log(f"\n  === Walk-Forward Results ===", f)
        for label, wf_data in wf_results.items():
            best = wf_data.get("best_score", {})
            best_name = wf_data.get("best_config", "N/A")
            if best:
                log(f"    {label}: best WF config: {best_name}", f)
                log(f"      Positive: {best.get('n_positive', 0)}/{best.get('n_splits', 0)}  "
                    f"avg_net={best.get('avg_net', 0):+.2%}  avg_sharpe={best.get('avg_sharpe', 0):.2f}", f)
            else:
                log(f"    {label}: No WF data", f)

        log(f"\n  === Ensemble Performance ===", f)
        for label, ens_data in ensemble_results.items():
            best = ens_data.get("best")
            if best:
                log(f"    {label}: AUC={ens_data['auc']:.4f}  cal_AUC={ens_data['cal_auc']:.4f}  "
                    f"best: {best['config']}  net={best['total_ret']:+.2%}  "
                    f"sharpe={best['sharpe']:.2f}", f)

        # Production readiness check
        log(f"\n  === PRODUCTION READINESS CHECK ===", f)
        ready = True
        for label, wf_data in wf_results.items():
            best = wf_data.get("best_score", {})
            n_pos = best.get("n_positive", 0)
            n_splits = best.get("n_splits", 0)
            if n_splits > 0 and n_pos >= 4:
                log(f"    [PASS] {label}: {n_pos}/{n_splits} WF positive", f)
            elif n_splits > 0 and n_pos >= 3:
                log(f"    [MARGINAL] {label}: {n_pos}/{n_splits} WF positive", f)
            else:
                log(f"    [FAIL] {label}: {n_pos}/{n_splits} WF positive", f)
                ready = False

        if ready:
            log(f"\n  STATUS: PRODUCTION CANDIDATE — needs live paper trading validation", f)
        else:
            log(f"\n  STATUS: NOT PRODUCTION READY — WF validation insufficient", f)
            log(f"  Next steps: see FINDINGS.md", f)

        # Save best models
        for label, data in horizon_results.items():
            if data["model"]:
                model_path = os.path.join(MODEL_DIR, f"v5_{label}_{data['horizon']}.txt")
                data["model"].booster_.save_model(model_path)
                log(f"\n  Saved {model_path}", f)

    print(f"\nResults written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
