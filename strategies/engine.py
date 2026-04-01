"""Strategy Engine — runs all 6 selected strategies and formats output for LLM.

Usage:
    # In pipeline (uses market_context_data/):
    from strategies.engine import StrategyEngine
    engine = StrategyEngine()
    result = engine.generate_signals()
    text = result["text_summary"]

    # For backtesting (uses raw_data/):
    engine = StrategyEngine(data_dir="raw_data")
    result = engine.generate_signals()
"""

import logging
from pathlib import Path

import pandas as pd

from strategies.base import BaseStrategy, StrategySignal
from strategies.funding_rate import FundingRateReversion
from strategies.volatility_regime import VolatilityRegime
from strategies.liquidation_flow import LiquidationFlow
from strategies.sentiment_flow import SentimentFlow
from strategies.trend_following import TrendFollowing
from strategies.momentum_composite import MomentumComposite

log = logging.getLogger(__name__)


def get_selected_strategies() -> list[BaseStrategy]:
    """Return the 6 selected production strategies."""
    return [
        FundingRateReversion(),
        VolatilityRegime(),
        LiquidationFlow(),
        SentimentFlow(),
        TrendFollowing(),
        MomentumComposite(),
    ]


class StrategyEngine:
    """Orchestrates all conventional strategies for the LLM pipeline."""

    def __init__(self, data_dir: str = "raw_data"):
        self.data_dir = data_dir
        self.strategies = get_selected_strategies()

    def generate_signals(self) -> dict:
        """Run all strategies and return structured signals.

        Returns:
            {
                "signals": {name: StrategySignal, ...},
                "consensus": {long_count, short_count, inactive_count},
                "text_summary": str  # formatted for LLM prompt
            }
        """
        signals = {}
        long_count = 0
        short_count = 0
        inactive_count = 0

        for strat in self.strategies:
            try:
                data = strat.load_data(self.data_dir)
                signal = strat.compute_signal(data)
                signals[strat.name] = signal

                if signal.direction == "LONG":
                    long_count += 1
                elif signal.direction == "SHORT":
                    short_count += 1
                else:
                    inactive_count += 1

                log.info("Strategy '%s': %s (conf=%.2f)",
                         strat.name, signal.direction, signal.confidence)
            except Exception as e:
                log.warning("Strategy '%s' failed: %s", strat.name, e)
                signals[strat.name] = StrategySignal(
                    "INACTIVE", 0.0, f"Error: {e}", {"error": str(e)}
                )
                inactive_count += 1

        consensus = {
            "long_count": long_count,
            "short_count": short_count,
            "inactive_count": inactive_count,
            "total": len(self.strategies),
        }

        text = self._format_text(signals, consensus)

        return {
            "signals": signals,
            "consensus": consensus,
            "text_summary": text,
        }

    def _format_text(self, signals: dict[str, StrategySignal],
                      consensus: dict) -> str:
        """Format strategy signals for the LLM prompt."""
        lines = [f"CONVENTIONAL STRATEGY SIGNALS ({len(signals)} strategies):"]
        lines.append("")

        for i, (name, sig) in enumerate(signals.items(), 1):
            # Direction with confidence
            if sig.direction == "INACTIVE":
                lines.append(f"  {i}. {name.upper()}: INACTIVE")
            else:
                lines.append(
                    f"  {i}. {name.upper()}: {sig.direction} "
                    f"(confidence: {sig.confidence:.2f})"
                )

            # Explanation
            lines.append(f"     {sig.explanation}")

            # Key metrics
            if sig.details:
                details_str = " | ".join(f"{k}={v}" for k, v in sig.details.items()
                                          if k != "error")
                if details_str:
                    lines.append(f"     Key: {details_str}")

            lines.append("")

        # Consensus
        lines.append(
            f"  STRATEGY CONSENSUS: {consensus['long_count']} LONG, "
            f"{consensus['short_count']} SHORT, "
            f"{consensus['inactive_count']} INACTIVE"
        )

        return "\n".join(lines)
