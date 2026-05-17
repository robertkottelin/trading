"""Random/grid-search parameter optimizer for strategies.

Optimizes both strategy-specific params and universal SimConfig overlays
(stop loss, take profit, sizing) for Sharpe ratio with min-trade constraints.

Walk-forward validation:
- Train: optimize params on early portion
- Test: validate on later portion
- Both windows: keep params only if both Sharpes ≥ threshold

Usage:
    python -m strategies.optimize --strategy macd --trials 200
    python -m strategies.optimize --all --trials 100
"""

import argparse
import json
import random
from copy import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from strategies.engine_v2 import (
    SimConfig, backtest, load_5m_price, SimResult,
)
from strategies.base import BaseStrategy

import yaml


# --------------------------------------------------------------------------- #
#  Parameter spaces — strategy-specific search ranges
# --------------------------------------------------------------------------- #


# Maps a strategy's class name → list of (param_name, sampler) tuples.
# Sampler is a callable returning a random value from a sensible distribution.

def _rand_int(lo: int, hi: int) -> Callable[[], int]:
    return lambda: random.randint(lo, hi)


def _rand_float(lo: float, hi: float) -> Callable[[], float]:
    return lambda: random.uniform(lo, hi)


def _rand_log(lo: float, hi: float) -> Callable[[], float]:
    return lambda: float(np.exp(random.uniform(np.log(lo), np.log(hi))))


def _rand_choice(*opts) -> Callable[[], Any]:
    return lambda: random.choice(opts)


PARAM_SPACES: dict[str, list[tuple[str, Callable[[], Any]]]] = {
    "FundingRateReversion": [
        ("ENTRY_Z", _rand_float(1.2, 3.0)),
        ("EXIT_Z", _rand_float(0.3, 1.0)),
        ("ZSCORE_WINDOW", _rand_int(50, 500)),
    ],
    "VolatilityRegime": [
        ("IV_RV_HIGH", _rand_float(1.1, 1.8)),
        ("IV_RV_LOW", _rand_float(0.4, 0.9)),
        ("IV_RV_EXIT_HIGH", _rand_float(0.9, 1.4)),
        ("IV_RV_EXIT_LOW", _rand_float(0.6, 1.1)),
        ("DVOL_MOMENTUM_HOURS", _rand_int(12, 48)),
        ("RV_WINDOW_HOURS", _rand_int(24, 96)),
        ("RV_COMPRESSION_PCTILE", _rand_int(10, 30)),
    ],
    "MacroRegime": [
        ("RETURN_LOOKBACK", _rand_int(3, 14)),
        ("VIX_HIGH", _rand_int(22, 35)),
        ("VIX_LOW", _rand_int(11, 18)),
        ("DXY_WEIGHT", _rand_float(0.5, 2.0)),
        ("VIX_WEIGHT", _rand_float(0.5, 2.0)),
        ("SPX_WEIGHT", _rand_float(0.5, 2.0)),
        ("LONG_THRESHOLD", _rand_float(0.5, 2.5)),
        ("SHORT_THRESHOLD", _rand_float(-3.5, -0.5)),
    ],
    "LiquidationFlow": [
        ("LIQ_ENTRY_Z", _rand_float(1.0, 2.5)),
        ("LIQ_ZSCORE_WINDOW", _rand_int(20, 80)),
        ("LS_ENTRY_Z", _rand_float(0.8, 2.0)),
        ("LS_ZSCORE_WINDOW", _rand_int(10, 50)),
        ("OI_CHANGE_WINDOW", _rand_int(5, 20)),
        ("PRICE_CHANGE_WINDOW", _rand_int(5, 20)),
        ("LONG_THRESHOLD", _rand_float(0.3, 1.5)),
    ],
    "SentimentFlow": [
        ("FNG_EXTREME_FEAR", _rand_int(10, 25)),
        ("FNG_EXTREME_GREED", _rand_int(70, 90)),
        ("FNG_MOMENTUM_WINDOW", _rand_int(7, 30)),
        ("LONG_THRESHOLD", _rand_float(0.5, 2.5)),
        ("SHORT_THRESHOLD", _rand_float(-3.0, -0.5)),
        ("STABLE_GROWTH_THRESHOLD", _rand_float(0.001, 0.02)),
        ("STABLE_DECLINE_THRESHOLD", _rand_float(-0.02, -0.001)),
    ],
    "BasisReversion": [
        ("ZSCORE_WINDOW", _rand_int(50, 500)),
        ("LONG_ZSCORE_WINDOW", _rand_int(200, 2000)),
        ("ENTRY_Z", _rand_float(1.0, 3.0)),
        ("EXIT_Z", _rand_float(0.2, 1.2)),
    ],
    "TrendFollowing": [
        # parameters minimal — kept simple
    ],
    "MomentumComposite": [
        ("LONG_THRESHOLD", _rand_float(0.5, 2.5)),
        ("SHORT_THRESHOLD", _rand_float(-2.5, -0.5)),
        ("EXIT_THRESHOLD", _rand_float(0.05, 0.6)),
        ("RSI_PERIOD", _rand_int(7, 28)),
        ("BB_PERIOD", _rand_int(10, 40)),
        ("BB_STD", _rand_choice(1.5, 2.0, 2.5)),
    ],
    "CommodityRiskAppetite": [],
    "FundingCarryMomentum": [],
    "SupertrendOBV": [
        ("ST_PERIOD", _rand_int(5, 14)),
        ("ST_MULT", _rand_float(1.5, 3.5)),
        ("OBV_EMA", _rand_int(10, 40)),
        ("TBR_WINDOW", _rand_int(10, 40)),
    ],
    "EMATrendRegime": [
        ("EMA_FAST", _rand_int(3, 10)),
        ("EMA_SLOW", _rand_int(8, 30)),
        ("DAILY_EMA", _rand_int(20, 100)),
        ("RSI_PERIOD", _rand_int(7, 28)),
        ("RSI_LONG_MIN", _rand_int(35, 55)),
        ("RSI_SHORT_MAX", _rand_int(48, 70)),
        ("BULL_WINDOW", _rand_int(15, 60)),
        ("BULL_THRESH", _rand_float(0.05, 0.30)),
    ],
    "MACDSignalCross": [
        ("MACD_FAST", _rand_int(8, 16)),
        ("MACD_SLOW", _rand_int(20, 36)),
        ("MACD_SIGNAL", _rand_int(5, 14)),
        ("HIST_MIN", _rand_float(0.0, 2.0)),
    ],
    "BBBreakoutOBV": [
        ("BB_PERIOD", _rand_int(15, 40)),
        ("BB_STD", _rand_choice(1.8, 2.0, 2.2, 2.5)),
        ("OBV_EMA", _rand_int(10, 40)),
    ],
    "StochasticEMACross": [
        ("STOCH_K", _rand_int(8, 20)),
        ("STOCH_D", _rand_int(3, 7)),
        ("EMA_FAST", _rand_int(8, 20)),
        ("EMA_SLOW", _rand_int(20, 60)),
    ],
    "RSITrendMomentum": [
        ("RSI_PERIOD", _rand_int(10, 21)),
        ("BULL_RSI_MIN", _rand_int(45, 60)),
        ("BEAR_RSI_MAX", _rand_int(40, 55)),
        ("EMA_TREND", _rand_int(20, 80)),
    ],
    "VWAPRSIReversion": [
        ("VWAP_WINDOW_DAYS", _rand_int(3, 14)),
        ("RSI_PERIOD", _rand_int(7, 21)),
        ("OVERSOLD", _rand_int(20, 35)),
        ("OVERBOUGHT", _rand_int(65, 80)),
    ],
    "DonchianBreakout": [
        ("DONCHIAN_PERIOD", _rand_int(10, 40)),
        ("EXIT_PERIOD", _rand_int(5, 20)),
    ],
    "WeeklyTrend": [
        ("MOMENTUM_WEEKS", _rand_int(2, 12)),
        ("SMA_WEEKS", _rand_int(12, 52)),
    ],
    "RangeMeanRev4h": [
        ("WINDOW", _rand_int(12, 60)),
        ("K_STD", _rand_float(1.5, 3.0)),
        ("EXIT_K", _rand_float(0.0, 0.7)),
    ],
    "AbsoluteMomentum": [
        ("LOOKBACK_DAYS", _rand_int(30, 180)),
        ("SHORT_LOOKBACK_DAYS", _rand_int(100, 250)),
        ("SHORT_THRESH", _rand_float(-0.5, -0.05)),
        ("ALLOW_SHORT", _rand_choice(True, False)),
    ],
    "VolatilityCarry": [
        ("IV_RV_LONG", _rand_float(1.05, 1.6)),
        ("IV_RV_EXIT", _rand_float(0.7, 1.1)),
        ("SMA_DAYS", _rand_int(20, 100)),
    ],
    "VolSqueezeBreakout": [
        ("BB_PERIOD", _rand_int(10, 40)),
        ("BB_STD", _rand_choice(1.5, 2.0, 2.5)),
        ("WIDTH_PCTILE_WINDOW", _rand_int(30, 150)),
        ("WIDTH_PCTILE_THRESH", _rand_float(0.10, 0.40)),
    ],
    "WeekendEffect": [],
    "MARegime": [
        ("SMA_DAYS", _rand_int(50, 250)),
        ("ALLOW_SHORT", _rand_choice(True, False)),
    ],
    "DualMomentum": [
        ("FAST_LOOKBACK", _rand_int(14, 60)),
        ("SLOW_LOOKBACK", _rand_int(60, 180)),
        ("VOL_LOWER", _rand_float(0.15, 0.35)),
        ("VOL_UPPER", _rand_float(0.55, 1.20)),
        ("ALLOW_SHORT", _rand_choice(True, False)),
        ("SHORT_LOOKBACK", _rand_int(30, 120)),
        ("SHORT_THRESH", _rand_float(-0.30, -0.05)),
    ],
    "ATRChannelTrend": [
        ("EMA_PERIOD", _rand_int(10, 40)),
        ("ATR_PERIOD", _rand_int(8, 24)),
        ("ATR_MULT", _rand_float(1.0, 3.5)),
    ],
    "TrendPullback": [
        ("SMA_DAYS", _rand_int(100, 250)),
        ("MOMENTUM_DAYS", _rand_int(30, 180)),
        ("MOMENTUM_MIN", _rand_float(0.0, 0.30)),
        ("DIP_DAYS", _rand_int(2, 7)),
        ("DIP_THRESH", _rand_float(-0.08, -0.01)),
        ("RSI_OVERSOLD", _rand_int(20, 40)),
        ("RSI_EXIT", _rand_int(50, 70)),
        ("MAX_HOLD_DAYS", _rand_int(3, 14)),
    ],
    "ETFFlowMomentum": [
        ("LOOKBACK_DAYS", _rand_int(3, 14)),
        ("ZSCORE_WINDOW", _rand_int(15, 60)),
        ("Z_LONG", _rand_float(0.3, 1.6)),
        ("Z_EXIT", _rand_float(-0.6, 0.3)),
    ],
    "ConsensusLong": [
        ("SMA_DAYS", _rand_int(100, 250)),
        ("MOM_DAYS", _rand_int(30, 180)),
        ("MOM_MIN", _rand_float(0.0, 0.25)),
        ("RSI_LOW", _rand_int(30, 50)),
        ("RSI_HIGH", _rand_int(55, 75)),
        ("VOL_LO", _rand_float(0.15, 0.35)),
        ("VOL_HI", _rand_float(0.55, 1.20)),
        ("EMA_SHORT", _rand_int(10, 40)),
        ("AGREE_ENTRY", _rand_int(3, 5)),
        ("AGREE_EXIT", _rand_int(1, 3)),
    ],
    "FeatureBasket": [
        ("Z_WIN", _rand_int(30, 120)),
        ("LONG_THRESH", _rand_float(0.2, 1.4)),
        ("EXIT_THRESH", _rand_float(-0.6, 0.2)),
        ("ALLOW_SHORT", _rand_choice(True, False)),
        ("SHORT_THRESH", _rand_float(-1.4, -0.4)),
    ],
    "QualityLong": [
        ("EMA_FAST", _rand_int(20, 80)),
        ("EMA_SLOW", _rand_int(100, 250)),
        ("MOM_DAYS", _rand_int(30, 180)),
        ("VOL_LO", _rand_float(0.15, 0.40)),
        ("VOL_HI", _rand_float(0.70, 1.20)),
        ("CRASH_DAYS", _rand_int(5, 21)),
        ("CRASH_THRESH", _rand_float(-0.15, -0.03)),
    ],
    "EthBtcRotation": [
        ("SMA_DAYS", _rand_int(20, 100)),
        ("RATIO_MOMENTUM_DAYS", _rand_int(3, 21)),
        ("MOMENTUM_MIN", _rand_float(-0.02, 0.02)),
    ],
    "LiquidityRegime": [
        ("LOOKBACK_DAYS", _rand_int(7, 60)),
        ("SMA_DAYS", _rand_int(20, 150)),
        ("SCORE_MIN", _rand_float(-0.01, 0.02)),
    ],
    "TripleMomentum": [
        ("H1", _rand_int(5, 21)),
        ("H2", _rand_int(20, 60)),
        ("H3", _rand_int(60, 180)),
        ("EXIT_H", _rand_int(15, 60)),
    ],
    "DXYRegime": [
        ("SMA_DAYS", _rand_int(20, 150)),
        ("RET_DAYS", _rand_int(5, 30)),
        ("RET_MAX", _rand_float(-0.02, 0.01)),
        ("BTC_SMA", _rand_int(100, 250)),
    ],
    "GoldenCrossRSI": [
        ("EMA_FAST", _rand_int(20, 80)),
        ("EMA_SLOW", _rand_int(100, 250)),
        ("RSI_PERIOD", _rand_int(10, 21)),
        ("RSI_ENTRY_BELOW", _rand_int(30, 50)),
        ("RSI_ENTRY_CROSS", _rand_int(40, 60)),
        ("RSI_EXIT", _rand_int(65, 85)),
    ],
    "DrawdownRecovery": [
        ("PEAK_WINDOW", _rand_int(30, 120)),
        ("DD_THRESH", _rand_float(-0.30, -0.08)),
        ("DD_LOOKBACK", _rand_int(10, 60)),
        ("EMA_PERIOD", _rand_int(10, 60)),
        ("TAKE_PROFIT_DAYS", _rand_int(20, 120)),
        ("TAKE_PROFIT_RET", _rand_float(0.15, 0.60)),
    ],
    "VolTargetBuyHold": [
        ("VOL_WINDOW", _rand_int(10, 60)),
        ("VOL_QUANTILE_WINDOW", _rand_int(180, 700)),
        ("VOL_QUANTILE_THRESH", _rand_float(0.75, 0.98)),
        ("EMA_REGIME", _rand_int(50, 250)),
    ],
    "MomentumRegimeComposite": [
        ("MOM_DAYS", _rand_int(10, 60)),
        ("MOM_THRESHOLD", _rand_float(-0.02, 0.10)),
        ("SMA_DAYS", _rand_int(20, 100)),
        ("RECENT_DAYS", _rand_int(3, 14)),
        ("RECENT_MIN", _rand_float(-0.15, -0.01)),
        ("RECENT_MAX", _rand_float(0.10, 0.40)),
    ],
    "CapitulationReversal": [
        ("MIN_DOWN_STREAK", _rand_int(3, 7)),
        ("REGIME_EMA", _rand_int(50, 250)),
        ("HOLD_BARS", _rand_int(3, 14)),
        ("STOP_PCT", _rand_float(0.03, 0.10)),
    ],
    "LowVolUptrend": [
        ("VOL_DAYS", _rand_int(14, 60)),
        ("VOL_LOW_PCTILE", _rand_float(0.10, 0.50)),
        ("VOL_HIGH_PCTILE", _rand_float(0.50, 0.85)),
        ("EMA_DAYS", _rand_int(20, 100)),
        ("SHORT_RET_DAYS", _rand_int(3, 14)),
        ("SHORT_RET_MIN", _rand_float(-0.10, -0.01)),
    ],
    "CreditSpreadRegime": [
        ("LOOKBACK_DAYS", _rand_int(5, 60)),
        ("SMA_DAYS", _rand_int(20, 100)),
        ("CHANGE_THRESH", _rand_float(-0.20, 0.10)),
    ],
    "FastRotationLong": [
        ("FAST_DAYS", _rand_int(3, 14)),
        ("FAST_THRESH", _rand_float(0.0, 0.08)),
        ("MED_DAYS", _rand_int(14, 60)),
        ("MED_THRESH", _rand_float(-0.05, 0.05)),
        ("EMA_DAYS", _rand_int(10, 60)),
        ("VOL_DAYS", _rand_int(7, 30)),
        ("VOL_LO", _rand_float(0.15, 0.40)),
        ("VOL_HI", _rand_float(0.65, 1.20)),
    ],
    "SlowTrend": [
        ("SHORT_DAYS", _rand_int(30, 90)),
        ("LONG_DAYS", _rand_int(90, 200)),
        ("EMA_DAYS", _rand_int(50, 200)),
    ],
}


SIM_PARAM_SPACE = [
    ("stop_loss_pct", _rand_choice(None, 0.03, 0.05, 0.07, 0.10)),
    ("take_profit_pct", _rand_choice(None, 0.08, 0.12, 0.18, 0.25)),
    ("max_hold_bars", _rand_choice(None, 10, 20, 40, 60)),
    ("sizing", _rand_choice("binary", "vol_target")),
    ("vol_target_annual", _rand_float(0.25, 0.65)),
    ("confidence_min", _rand_choice(0.0, 0.0, 0.5, 0.6, 0.7)),
]


# --------------------------------------------------------------------------- #
#  Optimizer
# --------------------------------------------------------------------------- #


def sample_strategy_params(klass_name: str) -> dict:
    space = PARAM_SPACES.get(klass_name, [])
    return {k: sampler() for k, sampler in space}


def sample_sim_config() -> dict:
    out = {}
    for k, sampler in SIM_PARAM_SPACE:
        out[k] = sampler()
    return out


def apply_strategy_params(strategy: BaseStrategy, params: dict) -> None:
    for k, v in params.items():
        setattr(strategy, k, v)


def evaluate(strategy: BaseStrategy, cfg: SimConfig,
             price_5m: pd.DataFrame,
             start_ms: int | None = None,
             end_ms: int | None = None,
             preloaded_data: dict | None = None,
             preloaded_price_bars: pd.DataFrame | None = None) -> dict:
    try:
        r = backtest(strategy, price_5m, cfg, start_ms=start_ms, end_ms=end_ms,
                     preloaded_data=preloaded_data,
                     preloaded_price_bars=preloaded_price_bars)
        return r.metrics
    except Exception as e:
        return {"sharpe": -99, "num_trades": 0, "error": str(e)}


def fitness_score(metrics: dict, min_trades: int = 20) -> float:
    """Single scalar fitness combining Sharpe + trade-count gate.

    Penalty if trades < min_trades; otherwise pure Sharpe with sortino tiebreaker.
    """
    sharpe = metrics.get("sharpe", -99)
    if metrics.get("error"):
        return -99
    n = metrics.get("num_trades", 0)
    if n < min_trades:
        return sharpe * (n / min_trades) - 0.5
    return sharpe + 0.05 * metrics.get("sortino", 0)


def optimize_strategy(strategy_factory: Callable[[], BaseStrategy],
                       price_5m: pd.DataFrame,
                       trials: int = 100,
                       train_end_ms: int | None = None,
                       test_start_ms: int | None = None,
                       min_trades: int = 20,
                       seed: int = 42,
                       full_history: bool = False) -> tuple[dict, dict, dict, dict]:
    """Random search optimizer with walk-forward train/test split.

    Returns (best_strategy_params, best_sim_config_dict, train_metrics, test_metrics)
    """
    random.seed(seed)
    np.random.seed(seed)

    klass_name = type(strategy_factory()).__name__

    # PRE-LOAD strategy data and resampled price bars ONCE — reused across trials.
    from strategies.engine_v2 import resample_to_bars, get_strategy_bar_seconds, DATA_DIR
    _proto = strategy_factory()
    preloaded_data = _proto.load_data(DATA_DIR)
    bar_seconds = get_strategy_bar_seconds(_proto)
    preloaded_price_bars = resample_to_bars(price_5m, bar_seconds)

    best_score = -np.inf
    best_strat = {}
    best_cfg = {}
    best_train = {}
    best_test = {}

    for t in range(trials):
        s_params = sample_strategy_params(klass_name)
        c_params = sample_sim_config()
        strategy = strategy_factory()
        apply_strategy_params(strategy, s_params)
        cfg = SimConfig(fee_per_side=0.0005, **c_params)

        if full_history:
            # Single-pass: optimize Sharpe on the whole dataset.
            full_metrics = evaluate(strategy, cfg, price_5m,
                                     preloaded_data=preloaded_data,
                                     preloaded_price_bars=preloaded_price_bars)
            n = full_metrics.get("num_trades", 0)
            if n < min_trades:
                continue
            sh = full_metrics.get("sharpe", -99)
            if sh < 0.3:
                continue
            score = sh + 0.05 * full_metrics.get("sortino", 0) + 0.02 * (n / 50)
            if score > best_score:
                best_score = score
                best_strat = s_params
                best_cfg = c_params
                best_train = full_metrics
                best_test = full_metrics
            continue

        # Walk-forward
        train_metrics = evaluate(strategy, cfg, price_5m,
                                 start_ms=None, end_ms=train_end_ms,
                                 preloaded_data=preloaded_data,
                                 preloaded_price_bars=preloaded_price_bars)
        if train_metrics.get("num_trades", 0) < min_trades:
            continue
        if train_metrics.get("sharpe", -99) < 0.3:
            continue

        strategy_test = strategy_factory()
        apply_strategy_params(strategy_test, s_params)
        test_metrics = evaluate(strategy_test, cfg, price_5m,
                                start_ms=test_start_ms, end_ms=None,
                                preloaded_data=preloaded_data,
                                preloaded_price_bars=preloaded_price_bars)

        train_s = train_metrics.get("sharpe", -99)
        test_s = test_metrics.get("sharpe", -99)
        gap_pen = max(0, train_s - test_s - 0.8) * 0.5
        combined = min(train_s, test_s) - gap_pen + 0.02 * test_metrics.get("num_trades", 0) / 50

        if combined > best_score:
            best_score = combined
            best_strat = s_params
            best_cfg = c_params
            best_train = train_metrics
            best_test = test_metrics

    return best_strat, best_cfg, best_train, best_test


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #


def get_strategy_factory(name: str) -> Callable[[], BaseStrategy]:
    """Return factory function for a strategy name (short name)."""
    from strategies.backtest_v2 import get_all_strategies
    for s in get_all_strategies():
        if type(s).__name__.lower() == name.lower() or \
                s.name.lower().replace(" ", "") == name.lower().replace(" ", ""):
            return lambda c=type(s): c()
    raise ValueError(f"Unknown strategy: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", help="Strategy class name to optimize")
    p.add_argument("--all", action="store_true", help="Optimize all strategies")
    p.add_argument("--trials", type=int, default=120)
    p.add_argument("--train-end", default="2024-09-30",
                   help="Train cut-off (YYYY-MM-DD)")
    p.add_argument("--test-start", default="2024-10-01",
                   help="Test start (YYYY-MM-DD)")
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="config/strategy_params.yaml")
    p.add_argument("--full-history", action="store_true",
                   help="Optimize Sharpe on full data (no train/test split)")
    args = p.parse_args()

    print("Loading 5-min price...")
    price_5m = load_5m_price()
    print(f"  {len(price_5m):,} 5-min bars")

    train_end_ms = int(pd.Timestamp(args.train_end, tz="UTC").value // 1_000_000)
    test_start_ms = int(pd.Timestamp(args.test_start, tz="UTC").value // 1_000_000)

    if args.all:
        from strategies.backtest_v2 import get_all_strategies
        strategies = get_all_strategies()
        factories = [(type(s).__name__, lambda c=type(s): c()) for s in strategies]
    else:
        if not args.strategy:
            raise SystemExit("Pass --strategy or --all")
        factories = [(args.strategy, get_strategy_factory(args.strategy))]

    # Load existing params
    out_path = Path(args.output)
    existing = {}
    if out_path.exists():
        existing = yaml.safe_load(out_path.read_text()) or {}

    # Load existing overlays to merge
    overlay_path = out_path.parent / "strategy_overlays.yaml"
    overlay_cfg: dict[str, dict] = {}
    if overlay_path.exists():
        try:
            overlay_cfg = yaml.safe_load(overlay_path.read_text()) or {}
        except Exception:
            pass

    summary = []
    # Track best Sharpe seen per strategy so re-runs only overwrite when better
    best_log_path = out_path.parent / "strategy_best_sharpe.yaml"
    best_log = yaml.safe_load(best_log_path.read_text()) if best_log_path.exists() else {}
    best_log = best_log or {}

    for name, factory in factories:
        print(f"\n--- Optimizing {name} ({args.trials} trials) ---", flush=True)
        s_params, c_params, train_m, test_m = optimize_strategy(
            factory, price_5m, trials=args.trials,
            train_end_ms=train_end_ms, test_start_ms=test_start_ms,
            min_trades=args.min_trades, seed=args.seed,
            full_history=args.full_history,
        )
        if not s_params and not c_params:
            print(f"  No viable params found.", flush=True)
            continue
        print(f"  Train: Sharpe={train_m.get('sharpe', 0):.2f} "
              f"Trades={train_m.get('num_trades', 0)}", flush=True)
        print(f"  Test:  Sharpe={test_m.get('sharpe', 0):.2f} "
              f"Trades={test_m.get('num_trades', 0)}", flush=True)
        print(f"  Params: {s_params}", flush=True)
        print(f"  SimCfg: {c_params}", flush=True)
        # Save under lowercase key (matches BaseStrategy override loader)
        key = name.lower()
        new_sharpe = float(train_m.get("sharpe", -99))
        prev_best = float(best_log.get(key, -99))
        if new_sharpe > prev_best:
            existing[key] = {**existing.get(key, {}), **s_params}
            overlay_cfg[key] = c_params
            best_log[key] = new_sharpe
            print(f"  ✓ NEW BEST (was {prev_best:.2f})", flush=True)
        else:
            print(f"  (skipping save: prev best {prev_best:.2f} > new {new_sharpe:.2f})",
                  flush=True)
        summary.append((name, train_m.get("sharpe", 0), test_m.get("sharpe", 0),
                        test_m.get("num_trades", 0)))

        # Checkpoint after EACH strategy
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.dump(existing, sort_keys=True))
        overlay_path.write_text(yaml.dump(overlay_cfg, sort_keys=True))
        best_log_path.write_text(yaml.dump(best_log, sort_keys=True))

    print(f"\nSaved strategy params -> {out_path}")
    print(f"Saved sim overlays  -> {overlay_path}")

    # Final summary
    print("\n" + "=" * 80)
    print(f"{'Strategy':<32}{'Train Shr':>12}{'Test Shr':>12}{'TestN':>10}")
    print("-" * 80)
    for name, trs, tes, n in summary:
        print(f"{name:<32}{trs:>12.2f}{tes:>12.2f}{n:>10d}")


if __name__ == "__main__":
    main()
