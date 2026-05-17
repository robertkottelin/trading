"""DonchianBreakout (4h bars) — classic Turtle/Faber breakout.

Long when price closes above N-bar high; short when below N-bar low. The
4-hour timeframe gives a balance between trade frequency and noise. Stops
are handled by the universal SimConfig overlay; the strategy emits raw
breakout signals (which persist until the opposite breakout occurs).

Thesis: trend-following on a 4h cadence catches multi-day moves while
re-positioning quickly. Uncorrelated with daily-bar EMA/MACD strategies
because it reacts to a different scale of price extreme.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class DonchianBreakout(BaseStrategy):
    name = "Donchian Breakout 4h"
    description = "4h-bar 20-period Donchian channel breakout — classic trend following"
    data_files = ["binance_futures_klines_5m.csv"]

    # Strategy params
    BAR_HOURS = 4
    DONCHIAN_PERIOD = 20      # Look back 20 4h bars (~3.3 days) for breakout level
    EXIT_PERIOD = 10          # Exit if 10-bar opposite breakout (Turtle Method 1)
    MIN_HISTORY = 50

    bar_seconds = BAR_HOURS * 3600

    def _resample_4h(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        if df_5m.empty:
            return pd.DataFrame()
        df = df_5m.copy()
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        bar_ms = self.bar_seconds * 1000
        df["bucket"] = (df["ts_ms"] // bar_ms) * bar_ms
        agg = df.groupby("bucket").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).reset_index().rename(columns={"bucket": "ts_ms"})
        return agg.sort_values("ts_ms").reset_index(drop=True)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._resample_4h(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()

        n = self.DONCHIAN_PERIOD
        exit_n = self.EXIT_PERIOD

        # Channel built from PRIOR closes (exclude current bar — vital for no look-ahead)
        upper = bars["close"].shift(1).rolling(n, min_periods=n).max()
        lower = bars["close"].shift(1).rolling(n, min_periods=n).min()
        exit_upper = bars["close"].shift(1).rolling(exit_n, min_periods=exit_n).max()
        exit_lower = bars["close"].shift(1).rolling(exit_n, min_periods=exit_n).min()

        close = bars["close"]
        signal = np.zeros(len(bars), dtype=int)
        pos = 0
        for i in range(self.MIN_HISTORY, len(bars)):
            if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
                signal[i] = pos
                continue
            c = close.iloc[i]
            if pos == 0:
                if c >= upper.iloc[i]:
                    pos = 1
                elif c <= lower.iloc[i]:
                    pos = -1
            elif pos == 1:
                # Exit long if drops below 10-bar low
                if c <= exit_lower.iloc[i]:
                    pos = 0
                # Reverse to short if breakout down
                if c <= lower.iloc[i]:
                    pos = -1
            elif pos == -1:
                if c >= exit_upper.iloc[i]:
                    pos = 0
                if c >= upper.iloc[i]:
                    pos = 1
            signal[i] = pos

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = np.where(signal != 0, 0.65, 0.0)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        return StrategySignal(d, float(last["confidence"]),
                              f"Donchian {self.DONCHIAN_PERIOD}-bar 4h breakout: {d}", {})
