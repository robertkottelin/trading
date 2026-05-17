"""Backtest runner with the v2 engine — realistic PnL, configurable risk overlays.

Usage:
    python -m strategies.backtest_v2
    python -m strategies.backtest_v2 --strategy 1
    python -m strategies.backtest_v2 --sl 0.05 --tp 0.10 --sizing confidence
"""

import argparse
import sys
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from strategies.engine_v2 import (
    SimConfig, backtest, load_5m_price, print_results,
    compute_signal_correlations, _tf_label,
)
from strategies.base import BaseStrategy
from strategies.funding_rate import FundingRateReversion
from strategies.volatility_regime import VolatilityRegime
from strategies.macro_regime import MacroRegime
from strategies.liquidation_flow import LiquidationFlow
from strategies.sentiment_flow import SentimentFlow
from strategies.basis_reversion import BasisReversion
from strategies.trend_following import TrendFollowing
from strategies.momentum_composite import MomentumComposite
from strategies.commodity_risk import CommodityRiskAppetite
from strategies.funding_carry import FundingCarryMomentum
from strategies.supertrend_obv import SupertrendOBV
from strategies.ema_trend_regime import EMATrendRegime
from strategies.macd_signal_cross import MACDSignalCross
from strategies.bb_breakout_obv import BBBreakoutOBV
from strategies.stochastic_ema_cross import StochasticEMACross
from strategies.rsi_trend_momentum import RSITrendMomentum
from strategies.vwap_rsi_reversion import VWAPRSIReversion
from strategies.donchian_breakout import DonchianBreakout
from strategies.weekly_trend import WeeklyTrend
from strategies.range_meanrev_4h import RangeMeanRev4h
from strategies.absolute_momentum import AbsoluteMomentum
from strategies.volatility_carry import VolatilityCarry
from strategies.vol_squeeze_breakout import VolSqueezeBreakout
from strategies.weekend_effect import WeekendEffect
from strategies.ma_regime import MARegime
from strategies.dual_momentum import DualMomentum
from strategies.atr_channel_trend import ATRChannelTrend
from strategies.trend_pullback import TrendPullback
from strategies.etf_flow_momentum import ETFFlowMomentum
from strategies.consensus_long import ConsensusLong
from strategies.feature_basket import FeatureBasket
from strategies.quality_long import QualityLong
from strategies.eth_btc_rotation import EthBtcRotation
from strategies.liquidity_regime import LiquidityRegime
from strategies.triple_momentum import TripleMomentum
from strategies.dxy_regime import DXYRegime
from strategies.golden_cross_rsi import GoldenCrossRSI
from strategies.drawdown_recovery import DrawdownRecovery
from strategies.vol_target_buyhold import VolTargetBuyHold
from strategies.momentum_regime_composite import MomentumRegimeComposite
from strategies.capitulation_reversal import CapitulationReversal
from strategies.low_vol_uptrend import LowVolUptrend
from strategies.credit_spread_regime import CreditSpreadRegime
from strategies.fast_rotation_long import FastRotationLong
from strategies.meta_consensus import MetaConsensus
from strategies.slow_trend import SlowTrend


def get_all_strategies() -> list[BaseStrategy]:
    return [
        FundingRateReversion(),
        VolatilityRegime(),
        MacroRegime(),
        LiquidationFlow(),
        SentimentFlow(),
        BasisReversion(),
        TrendFollowing(),
        MomentumComposite(),
        CommodityRiskAppetite(),
        FundingCarryMomentum(),
        SupertrendOBV(),
        EMATrendRegime(),
        MACDSignalCross(),
        BBBreakoutOBV(),
        StochasticEMACross(),
        RSITrendMomentum(),
        VWAPRSIReversion(),
        # New strategies (multi-timeframe, diverse data)
        DonchianBreakout(),
        WeeklyTrend(),
        RangeMeanRev4h(),
        AbsoluteMomentum(),
        VolatilityCarry(),
        VolSqueezeBreakout(),
        WeekendEffect(),
        MARegime(),
        DualMomentum(),
        ATRChannelTrend(),
        TrendPullback(),
        ETFFlowMomentum(),
        ConsensusLong(),
        FeatureBasket(),
        QualityLong(),
        EthBtcRotation(),
        LiquidityRegime(),
        TripleMomentum(),
        DXYRegime(),
        GoldenCrossRSI(),
        DrawdownRecovery(),
        VolTargetBuyHold(),
        MomentumRegimeComposite(),
        CapitulationReversal(),
        LowVolUptrend(),
        CreditSpreadRegime(),
        FastRotationLong(),
        MetaConsensus(),
        SlowTrend(),
    ]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", type=int, default=None,
                   help="Index of single strategy to test (1-based)")
    p.add_argument("--fee", type=float, default=0.0005, help="Per-side fee (default 5 bps)")
    p.add_argument("--sl", type=float, default=None, help="Stop loss pct (e.g. 0.05)")
    p.add_argument("--tp", type=float, default=None, help="Take profit pct")
    p.add_argument("--max-hold", type=int, default=None, help="Max hold bars")
    p.add_argument("--conf-min", type=float, default=0.0, help="Min confidence to take signal")
    p.add_argument("--sizing", default="binary", choices=["binary", "confidence", "vol_target"])
    p.add_argument("--vol-target", type=float, default=0.40, help="Annualized vol target")
    p.add_argument("--leverage-max", type=float, default=1.0)
    p.add_argument("--no-corr", action="store_true", help="Skip correlation matrix")
    p.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    p.add_argument("--no-overlays", action="store_true",
                   help="Ignore config/strategy_overlays.yaml")
    p.add_argument("--min-sharpe", type=float, default=2.0,
                   help="Sharpe threshold for the passers list")
    return p.parse_args()


def load_overlays(path: str = "config/strategy_overlays.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def cfg_for_strategy(base: SimConfig, strategy: BaseStrategy, overlays: dict) -> SimConfig:
    """Apply per-strategy overlay if present."""
    key = type(strategy).__name__.lower()
    ovr = overlays.get(key) or {}
    if not ovr:
        return base
    cfg = copy(base)
    for k, v in ovr.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main():
    args = parse_args()

    cfg = SimConfig(
        fee_per_side=args.fee,
        stop_loss_pct=args.sl,
        take_profit_pct=args.tp,
        max_hold_bars=args.max_hold,
        confidence_min=args.conf_min,
        sizing=args.sizing,
        vol_target_annual=args.vol_target,
        leverage_max=args.leverage_max,
    )

    print("Loading 5-min price reference...")
    price_5m = load_5m_price()
    print(f"Price data: {len(price_5m):,} 5-min bars, "
          f"{pd.to_datetime(price_5m['ts_ms'].iloc[0], unit='ms')} → "
          f"{pd.to_datetime(price_5m['ts_ms'].iloc[-1], unit='ms')}")

    start_ms = end_ms = None
    if args.start:
        start_ms = int(pd.Timestamp(args.start, tz="UTC").value // 1_000_000)
    if args.end:
        end_ms = int(pd.Timestamp(args.end, tz="UTC").value // 1_000_000)

    strategies = get_all_strategies()
    if args.strategy:
        strategies = [strategies[args.strategy - 1]]

    overlays = {} if args.no_overlays else load_overlays()

    print(f"\nRunning v2 backtest on {len(strategies)} strategies...")
    print(f"Base config: fee={cfg.fee_per_side*10000:.0f}bps/side, "
          f"sl={cfg.stop_loss_pct}, tp={cfg.take_profit_pct}, "
          f"sizing={cfg.sizing}, conf_min={cfg.confidence_min}")
    if overlays:
        print(f"Per-strategy overlays loaded: {len(overlays)} entries")

    results = []
    for s in strategies:
        scfg = cfg_for_strategy(cfg, s, overlays)
        r = backtest(s, price_5m, scfg, start_ms=start_ms, end_ms=end_ms)
        results.append(r)

    print_results(results)

    # Highlight passers
    print()
    thr = args.min_sharpe
    print(f"CANDIDATES PASSING SHARPE >= {thr:.1f}:")
    passers = [r for r in results if r.metrics.get("sharpe", 0) >= thr
               and r.metrics.get("num_trades", 0) >= 20]
    for r in passers:
        m = r.metrics
        print(f"  ✓ {m['strategy']:<32} Sharpe={m['sharpe']:.2f} "
              f"Trades={m['num_trades']} Return={m['total_return']:.1%}")
    if not passers:
        print("  (none yet)")

    # Correlation
    if not args.no_corr and len(results) > 1:
        print("\nCROSS-STRATEGY DAILY-RETURN CORRELATION:")
        corr = compute_signal_correlations(results)
        if not corr.empty:
            names = list(corr.columns)
            header = f"{'':>32}" + "".join(f"{n[:11]:>12}" for n in names)
            print(header)
            for i, n in enumerate(names):
                row = f"{n[:31]:>32}"
                for j in range(len(names)):
                    row += f"{corr.iloc[i, j]:>12.2f}"
                print(row)


if __name__ == "__main__":
    main()
