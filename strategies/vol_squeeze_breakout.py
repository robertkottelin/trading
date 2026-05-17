"""VolSqueezeBreakout (daily) — only trade after volatility compression.

Compression regimes precede big moves. Strategy waits until BBwidth is in
its 20th percentile of the past 90 days, then takes the side of the next
breakout. Stops handled by SimConfig overlay.

Differentiated from BBBreakoutOBV: that one requires OBV confirmation and
fires on any band touch; this one requires a SQUEEZE first, which filters
out trend-continuation breakouts that lack vol expansion.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class VolSqueezeBreakout(BaseStrategy):
    name = "Vol Squeeze Breakout"
    description = "Wait for BB width compression, then ride breakout direction"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    BB_PERIOD = 20
    BB_STD = 2.0
    WIDTH_PCTILE_WINDOW = 90
    WIDTH_PCTILE_THRESH = 0.25
    MIN_HISTORY = 120

    def _daily_ohlcv(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).reset_index().rename(columns={"bucket": "ts_ms"})
        return agg.sort_values("ts_ms").reset_index(drop=True)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily_ohlcv(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()

        close = bars["close"]
        c_prev = close.shift(1)
        sma = c_prev.rolling(self.BB_PERIOD, min_periods=self.BB_PERIOD).mean()
        std = c_prev.rolling(self.BB_PERIOD, min_periods=self.BB_PERIOD).std()
        upper = sma + self.BB_STD * std
        lower = sma - self.BB_STD * std
        width = (upper - lower) / sma
        width_rank = width.rolling(self.WIDTH_PCTILE_WINDOW, min_periods=30).rank(pct=True)

        signal = np.zeros(len(bars), dtype=int)
        pos = 0
        squeezed = False
        for i in range(self.MIN_HISTORY, len(bars)):
            wr = width_rank.iloc[i]
            cp = close.iloc[i]
            u = upper.iloc[i]
            l = lower.iloc[i]
            s_i = sma.iloc[i]
            if pd.isna(wr) or pd.isna(u) or pd.isna(l):
                signal[i] = pos
                continue
            # Track squeeze state: enter when width rank < threshold
            if wr < self.WIDTH_PCTILE_THRESH:
                squeezed = True

            if pos == 0:
                if squeezed and cp >= u:
                    pos = 1
                    squeezed = False
                elif squeezed and cp <= l:
                    pos = -1
                    squeezed = False
            elif pos == 1:
                # Exit long when price drops back below mid SMA
                if cp < s_i:
                    pos = 0
            elif pos == -1:
                if cp > s_i:
                    pos = 0
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
                              "BB squeeze breakout", {})
