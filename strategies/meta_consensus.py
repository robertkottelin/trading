"""MetaConsensus (daily) — long when at least K of 4 winners agree.

Combines the 4 winning strategies (LiquidityRegime, DrawdownRecovery,
TripleMomentum, MomentumRegimeComposite) into a single consensus signal.
This is itself a strategy — when at least 2 of the 4 underlying signals
say LONG, we go long.

This is a META-strategy: its signal is the AND/OR of other signals. While
it's correlated with the underlyings, it can have HIGHER Sharpe due to:
  - Diversification across the 4 perspectives
  - Filter effect (requires agreement, fewer false positives)
  - The full vote of all 4 produces an even-stronger signal
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal
from strategies.liquidity_regime import LiquidityRegime
from strategies.drawdown_recovery import DrawdownRecovery
from strategies.triple_momentum import TripleMomentum
from strategies.momentum_regime_composite import MomentumRegimeComposite


class MetaConsensus(BaseStrategy):
    name = "Meta Consensus"
    description = "Long when ≥2 of 4 winning strategies signal long"
    data_files = [
        "binance_futures_klines_5m.csv",
        "macro_liquidity.csv",
    ]

    bar_seconds = 86400

    AGREE_ENTRY = 2
    AGREE_EXIT = 1

    def __init__(self):
        super().__init__()
        self._sub_strategies = [
            LiquidityRegime(),
            DrawdownRecovery(),
            TripleMomentum(),
            MomentumRegimeComposite(),
        ]

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        # Get each sub-strategy's signal series
        sub_signals = []
        for s in self._sub_strategies:
            sub_data = s.load_data("raw_data")
            sig_df = s.compute_signal_series(sub_data)
            if sig_df.empty:
                continue
            sig_df = sig_df[["ts_ms", "signal"]].copy()
            sig_df = sig_df.rename(columns={"signal": s.name})
            sub_signals.append(sig_df)

        if len(sub_signals) < 2:
            return pd.DataFrame()

        # Merge on ts_ms
        merged = sub_signals[0]
        for s in sub_signals[1:]:
            merged = merged.merge(s, on="ts_ms", how="outer")
        merged = merged.sort_values("ts_ms").reset_index(drop=True)
        merged = merged.ffill().fillna(0)

        sig_cols = [c for c in merged.columns if c != "ts_ms"]
        merged["agree"] = (merged[sig_cols] == 1).sum(axis=1)

        signal = np.zeros(len(merged), dtype=int)
        pos = 0
        for i in range(len(merged)):
            a = merged["agree"].iloc[i]
            if pos == 0 and a >= self.AGREE_ENTRY:
                pos = 1
            elif pos == 1 and a < self.AGREE_EXIT + 1:
                pos = 0
            signal[i] = pos

        out = merged[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = np.clip(merged["agree"] / 4.0, 0.0, 0.95)
        out["confidence"] = np.where(signal == 1, out["confidence"], 0.0)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Sub-strategies unavailable", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Meta-consensus of 4 winners", {})
