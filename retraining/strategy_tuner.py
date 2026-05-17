"""Strategy parameter tuner — Optuna-based optimization for all 13 strategies.

Runs Optuna trials for each strategy, backtesting across 4 historical periods.
Anti-overfitting constraints ensure parameters are robust, not period-specific.
Best params written to config/strategy_params_staging.yaml.

Usage:
    python retraining/strategy_tuner.py              # tune all 13 strategies
    python retraining/strategy_tuner.py --strategy funding_rate --trials 5
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import optuna
import yaml

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [strategy_tuner] %(levelname)s: %(message)s")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.backtest import backtest_strategy, load_price_reference
from strategies.funding_rate import FundingRateReversion
from strategies.volatility_regime import VolatilityRegime
from strategies.liquidation_flow import LiquidationFlow
from strategies.sentiment_flow import SentimentFlow
from strategies.trend_following import TrendFollowing
from strategies.momentum_composite import MomentumComposite
from strategies.macro_regime import MacroRegime
from strategies.basis_reversion import BasisReversion
from strategies.taker_flow import TakerFlowImbalance
from strategies.commodity_risk import CommodityRiskAppetite
from strategies.funding_carry import FundingCarryMomentum
from strategies.supertrend_obv import SupertrendOBV
from strategies.ema_trend_regime import EMATrendRegime

OUTPUT_PATH = Path("config/strategy_params_staging.yaml")

# Backtest periods for robustness validation
PERIODS = [
    ("2020-2022", date(2020, 1, 1), date(2021, 12, 31)),
    ("2022-2024", date(2022, 1, 1), date(2023, 12, 31)),
    ("2024-2026", date(2024, 1, 1), date(2026, 3, 1)),
    ("Full",      None,              None),
]

# Anti-overfitting constraints (applied in objective — violations return -999)
MIN_SHARPE_PER_PERIOD = -0.2      # no single period worse than this
MIN_TRADES_PER_PERIOD = 8         # at least 8 trades per period
MAX_SHARPE_SPREAD = 4.0           # max(Sharpes) - min(Sharpes) < 4.0
MIN_TRADES_PER_YEAR_FULL = 20     # at least 20 trades/year overall

DEFAULT_TRIALS = 30


# =============================================================================
# PARAMETER GRIDS  (suggest_int / suggest_float ranges per strategy class name)
# =============================================================================

PARAM_GRIDS: dict[str, dict] = {
    "FundingRateReversion": {
        "SHORT_ZSCORE":      ("int",   40, 120),
        "LONG_ZSCORE":       ("int",   90, 360),
        "ENTRY_Z_SHORT":     ("float", 1.2, 3.0),
        "ENTRY_Z_LONG":      ("float", 1.0, 2.5),
        "EXIT_Z":            ("float", 0.2, 1.0),
        "OI_BOOST_Z":        ("float", 0.5, 2.0),
        "PRICE_TREND_WINDOW": ("int",  5, 14),
    },
    "VolatilityRegime": {
        "RV_WINDOW_HOURS":          ("int",   12, 48),
        "RV_COMPRESSION_PCTILE":    ("int",   5, 25),
        "IV_RV_HIGH":               ("float", 1.2, 2.0),
        "IV_RV_LOW":                ("float", 0.5, 0.9),
        "IV_RV_EXIT_HIGH":          ("float", 1.0, 1.4),
        "IV_RV_EXIT_LOW":           ("float", 0.7, 1.0),
        "DVOL_MOMENTUM_HOURS":      ("int",   12, 48),
    },
    "LiquidationFlow": {
        "LIQ_ZSCORE_WINDOW": ("int",   15, 60),
        "LIQ_ENTRY_Z":       ("float", 0.8, 2.0),
        "OI_CHANGE_WINDOW":  ("int",   3, 10),
        "PRICE_CHANGE_WINDOW": ("int", 3, 10),
        "LS_ZSCORE_WINDOW":  ("int",   20, 80),
        "LS_ENTRY_Z":        ("float", 0.8, 2.0),
        "LONG_THRESHOLD":    ("float", 0.5, 2.0),
    },
    "SentimentFlow": {
        "FNG_EXTREME_FEAR":       ("int",   10, 25),
        "FNG_EXTREME_GREED":      ("int",   75, 90),
        "FNG_MOMENTUM_WINDOW":    ("int",   7, 21),
        "STABLE_GROWTH_THRESHOLD": ("float", 0.002, 0.01),
        "STABLE_DECLINE_THRESHOLD": ("float", -0.008, -0.001),
        "LONG_THRESHOLD":         ("float", 1.0, 2.5),
        "SHORT_THRESHOLD":        ("float", -2.5, -1.0),
    },
    "TrendFollowing": {
        "SHORT_FAST":  ("int",  72, 288),
        "SHORT_SLOW":  ("int", 144, 576),
        "ADX_TREND":   ("int",  20,  35),
        "ADX_RANGE":   ("int",  12,  22),
    },
    "MomentumComposite": {
        "RSI_PERIOD":       ("int",   7, 21),
        "BB_PERIOD":        ("int",  14, 30),
        "BB_STD":           ("int",   1,  3),
        "LONG_THRESHOLD":   ("float", 0.8, 2.0),
        "SHORT_THRESHOLD":  ("float", -2.0, -0.8),
        "EXIT_THRESHOLD":   ("float", 0.1, 0.6),
    },
    "MacroRegime": {
        "LONG_THRESHOLD":   ("float", 1.0, 3.5),
        "SHORT_THRESHOLD":  ("float", -3.5, -1.0),
        "VIX_LOW":          ("int",   14, 22),
        "VIX_HIGH":         ("int",   24, 40),
        "RETURN_LOOKBACK":  ("int",    3, 10),
        "SPX_WEIGHT":       ("float", 0.5, 1.5),
        "DXY_WEIGHT":       ("float", 0.5, 1.5),
        "VIX_WEIGHT":       ("float", 0.5, 1.5),
    },
    "BasisReversion": {
        "ZSCORE_WINDOW":      ("int",   168, 504),
        "ENTRY_Z":            ("float", 1.5,  3.5),
        "EXIT_Z":             ("float", 0.3,  1.5),
        "LONG_ZSCORE_WINDOW": ("int",   504, 1512),
    },
    "TakerFlowImbalance": {
        "ZSCORE_WINDOW": ("int",   24, 96),
        "MIN_PERIODS":   ("int",    8, 24),
        "ENTRY_Z":       ("float", 1.0, 2.5),
        "EXIT_Z":        ("float", 0.3, 1.0),
    },
    "CommodityRiskAppetite": {
        "RETURN_WINDOW":          ("int",   1, 7),
        "ZSCORE_WINDOW":          ("int",  10, 40),
        "ENTRY_Z":                ("float", 0.5, 1.5),
        "EXIT_Z":                 ("float", 0.05, 0.4),
        "OIL_CRASH_THRESHOLD":    ("float", -0.15, -0.04),
        "BULL_MARKET_WINDOW":     ("int",   30, 90),
        "BULL_MARKET_THRESHOLD":  ("float", 0.10, 0.50),
    },
    "FundingCarryMomentum": {
        "CARRY_WINDOW":       ("int",    3, 12),
        "TREND_WINDOW":       ("int",    7, 28),
        "ENTRY_CUM":          ("float",  0.00010, 0.00060),
        "ENTRY_CUM_SHORT":    ("float", -0.00060, -0.00005),
        "EXIT_CUM_LONG":      ("float",  0.000005, 0.00005),
        "EXTREME_Z":          ("float",  1.0, 2.5),
        "ZSCORE_WINDOW":      ("int",   90, 270),
    },
    "SupertrendOBV": {
        "ST_PERIOD":           ("int",    4, 14),
        "ST_MULT":             ("float",  1.0, 3.5),
        "OBV_EMA":             ("int",   10, 40),
        "BULL_RETURN_WINDOW":  ("int",   30, 120),
        "BULL_RETURN_THRESH":  ("float", 0.10, 0.40),
    },
    "EMATrendRegime": {
        "EMA_FAST":       ("int",   3, 10),
        "EMA_SLOW":       ("int",   8, 21),
        "RSI_PERIOD":     ("int",   5, 14),
        "RSI_LONG_MIN":   ("int",  35, 52),
        "RSI_SHORT_MAX":  ("int",  48, 65),
        "DAILY_EMA":      ("int",  20, 80),
        "BULL_WINDOW":    ("int",  10, 40),
        "BULL_THRESH":    ("float", 0.05, 0.35),
    },
}

# Strategy registry: class_name → (class, display_key)
STRATEGY_REGISTRY = {
    "FundingRateReversion":  (FundingRateReversion,  "FundingRateReversion"),
    "VolatilityRegime":      (VolatilityRegime,       "VolatilityRegime"),
    "LiquidationFlow":       (LiquidationFlow,        "LiquidationFlow"),
    "SentimentFlow":         (SentimentFlow,          "SentimentFlow"),
    "TrendFollowing":        (TrendFollowing,         "TrendFollowing"),
    "MomentumComposite":     (MomentumComposite,      "MomentumComposite"),
    "MacroRegime":           (MacroRegime,            "MacroRegime"),
    "BasisReversion":        (BasisReversion,         "BasisReversion"),
    "TakerFlowImbalance":    (TakerFlowImbalance,     "TakerFlowImbalance"),
    "CommodityRiskAppetite": (CommodityRiskAppetite,  "CommodityRiskAppetite"),
    "FundingCarryMomentum":  (FundingCarryMomentum,   "FundingCarryMomentum"),
    "SupertrendOBV":         (SupertrendOBV,          "SupertrendOBV"),
    "EMATrendRegime":        (EMATrendRegime,         "EMATrendRegime"),
}


def _sample_params(trial: optuna.Trial, grid: dict) -> dict:
    params = {}
    for name, spec in grid.items():
        ptype, lo, hi = spec
        if ptype == "int":
            params[name] = trial.suggest_int(name, int(lo), int(hi))
        else:
            params[name] = trial.suggest_float(name, float(lo), float(hi))
    return params


def _apply_params(instance, params: dict):
    """Monkey-patch params onto a strategy instance (shadows class constants)."""
    for k, v in params.items():
        if hasattr(instance, k):
            setattr(instance, k, v)


def _run_period_backtest(strategy_class, preloaded_data: dict,
                          price_df, params: dict) -> dict:
    """Instantiate strategy with params, run 4-period backtest. Returns metrics dict."""
    results = {}
    for period_name, start, end in PERIODS:
        instance = strategy_class.__new__(strategy_class)  # skip __init__ (no YAML load)
        # Copy class defaults first
        for attr in dir(strategy_class):
            if attr.isupper() and not attr.startswith("_"):
                try:
                    setattr(instance, attr, getattr(strategy_class, attr))
                except AttributeError:
                    pass
        _apply_params(instance, params)
        # Patch load_data to return preloaded data
        instance.load_data = lambda d, _data=preloaded_data: _data

        try:
            result = backtest_strategy(instance, price_df, start_date=start, end_date=end)
            if "error" in result:
                results[period_name] = {"sharpe": 0.0, "trades_per_year": 0.0,
                                         "num_trades": 0, "error": result["error"]}
            else:
                m = result["metrics"]
                results[period_name] = {
                    "sharpe": float(m.get("sharpe", 0.0)),
                    "trades_per_year": float(m.get("trades_per_year", 0.0)),
                    "num_trades": int(m.get("num_trades", 0)),
                }
        except Exception as e:
            results[period_name] = {"sharpe": 0.0, "trades_per_year": 0.0,
                                     "num_trades": 0, "error": str(e)}
    return results


def _composite_score(period_results: dict) -> float:
    """Compute composite optimization score from 4-period results.

    Returns -999 if any anti-overfitting constraint is violated.
    Otherwise: 40% avg Sharpe + 30% min Sharpe + 15% trade frequency + 15% robustness bonus.
    """
    sharpes = []
    trade_counts = []
    tpy_values = []

    for period_name, m in period_results.items():
        sharpe = m.get("sharpe", 0.0)
        n_trades = m.get("num_trades", 0)
        tpy = m.get("trades_per_year", 0.0)

        # Anti-overfitting: any period Sharpe too negative → fail
        if sharpe < MIN_SHARPE_PER_PERIOD:
            return -999.0

        # Anti-overfitting: too few trades per period → fail
        if period_name != "Full" and n_trades < MIN_TRADES_PER_PERIOD:
            return -999.0

        sharpes.append(sharpe)
        trade_counts.append(n_trades)
        if period_name == "Full":
            tpy_values.append(tpy)

    if not sharpes:
        return -999.0

    # Anti-overfitting: Sharpe variance too high (fit to one period)
    if max(sharpes) - min(sharpes) > MAX_SHARPE_SPREAD:
        return -999.0

    # Anti-overfitting: too few trades in full period
    full_tpy = tpy_values[0] if tpy_values else 0.0
    if full_tpy < MIN_TRADES_PER_YEAR_FULL:
        return -999.0

    avg_sharpe = float(np.mean(sharpes))
    min_sharpe = float(min(sharpes))
    freq_score = min(full_tpy / 100.0, 1.0)

    score = 0.40 * max(avg_sharpe, 0) + 0.30 * max(min_sharpe, 0) + 0.15 * freq_score
    return score


def tune_strategy(strategy_class, n_trials: int = DEFAULT_TRIALS) -> dict | None:
    """Run Optuna optimization for one strategy. Returns best params dict or None."""
    class_name = strategy_class.__name__
    grid = PARAM_GRIDS.get(class_name)
    if not grid:
        log.warning("No param grid for %s — skipping", class_name)
        return None

    log.info("Tuning %s (%d trials)...", class_name, n_trials)

    # Preload data once (shared across all trials)
    base_instance = strategy_class.__new__(strategy_class)
    for attr in dir(strategy_class):
        if attr.isupper() and not attr.startswith("_"):
            try:
                setattr(base_instance, attr, getattr(strategy_class, attr))
            except AttributeError:
                pass

    try:
        preloaded_data = base_instance.load_data("raw_data")
    except Exception as e:
        log.warning("Could not preload data for %s: %s — skipping", class_name, e)
        return None

    price_df = load_price_reference()

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, grid)
        period_results = _run_period_backtest(strategy_class, preloaded_data,
                                               price_df, params)
        return _composite_score(period_results)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_value = study.best_value
    best_params = study.best_params

    if best_value <= 0:
        log.warning("%s: best score=%.3f (no improvement over defaults — keeping defaults)",
                    class_name, best_value)
        return None

    # Validate best params vs baseline (default params → no YAML overrides)
    default_instance = strategy_class.__new__(strategy_class)
    for attr in dir(strategy_class):
        if attr.isupper() and not attr.startswith("_"):
            try:
                setattr(default_instance, attr, getattr(strategy_class, attr))
            except AttributeError:
                pass
    default_instance.load_data = lambda d, _data=preloaded_data: _data

    default_results = _run_period_backtest(strategy_class, preloaded_data,
                                            price_df, {})
    default_score = _composite_score(default_results)

    if best_value <= default_score:
        log.info("%s: tuned score=%.3f not better than defaults=%.3f — keeping defaults",
                 class_name, best_value, default_score)
        return None

    log.info("%s: best score=%.3f (vs defaults=%.3f) — %d params improved",
             class_name, best_value, default_score, len(best_params))
    log.info("%s: best params: %s", class_name, best_params)
    return best_params


def run_all(strategy_filter: str | None = None, n_trials: int = DEFAULT_TRIALS):
    """Tune all (or one) strategy and write config/strategy_params_staging.yaml."""
    # Load existing staging params if partial run
    staging_params: dict = {}
    if OUTPUT_PATH.exists():
        try:
            existing = yaml.safe_load(OUTPUT_PATH.read_text()) or {}
            staging_params.update(existing)
        except Exception:
            pass

    strategies_to_tune = list(STRATEGY_REGISTRY.items())
    if strategy_filter:
        strategies_to_tune = [(k, v) for k, v in strategies_to_tune
                               if k.lower() == strategy_filter.lower()
                               or strategy_filter.lower() in k.lower()]
        if not strategies_to_tune:
            log.error("No strategy matched '%s'. Available: %s",
                      strategy_filter, list(STRATEGY_REGISTRY.keys()))
            sys.exit(1)

    for class_name, (strategy_class, yaml_key) in strategies_to_tune:
        try:
            best_params = tune_strategy(strategy_class, n_trials=n_trials)
        except Exception as e:
            log.error("Tuning %s failed: %s — skipping", class_name, e)
            continue

        if best_params:
            staging_params[yaml_key.lower()] = best_params
        # If None, leave previous entry (or absent = use class defaults)

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(staging_params, f, default_flow_style=False, sort_keys=True)

    log.info("Strategy params written to %s (%d strategies tuned)",
             OUTPUT_PATH, len(staging_params))


def main():
    parser = argparse.ArgumentParser(description="Tune strategy parameters with Optuna")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Tune only this strategy (class name or substring)")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"Optuna trials per strategy (default: {DEFAULT_TRIALS})")
    args = parser.parse_args()
    run_all(strategy_filter=args.strategy, n_trials=args.trials)


if __name__ == "__main__":
    main()
