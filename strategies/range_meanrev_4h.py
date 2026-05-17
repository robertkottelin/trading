"""RangeMeanReversion (4h bars) — counter-trend mean reversion at extremes.

Thesis: BTC chops around within structural trends. On 4-hour bars, a 6%
move away from the 24-bar (4-day) mean is typically followed by reversion
toward the mean. This complements trend strategies which take the OTHER
side of these reversions.

Entry:
  - LONG  when close < lower_band  (price -2σ below 24-bar SMA)
  - SHORT when close > upper_band  (price +2σ above 24-bar SMA)
Exit:
  - Price reverts back through the mean
  - Or universal stop-loss triggers
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class RangeMeanRev4h(BaseStrategy):
    name = "Range Mean Reversion 4h"
    description = "4h Bollinger-style mean reversion — buy 2σ below, sell 2σ above 24-bar SMA"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 4 * 3600
    WINDOW = 24            # 4 days
    K_STD = 2.0            # entry bands at +/- K*std
    EXIT_K = 0.2           # exit when within +/-0.2*std of mean
    MIN_HISTORY = 50

    def _resample(self, df_5m: pd.DataFrame) -> pd.DataFrame:
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
        ).reset_index().rename(columns={"bucket": "ts_ms"})
        return agg.sort_values("ts_ms").reset_index(drop=True)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._resample(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()
        c = bars["close"].shift(1)  # all stats computed from prior bars to avoid look-ahead
        sma = c.rolling(self.WINDOW, min_periods=self.WINDOW).mean()
        std = c.rolling(self.WINDOW, min_periods=self.WINDOW).std()
        upper = sma + self.K_STD * std
        lower = sma - self.K_STD * std
        z = (bars["close"] - sma) / std

        signal = np.zeros(len(bars), dtype=int)
        pos = 0
        for i in range(self.MIN_HISTORY, len(bars)):
            zi = z.iloc[i]
            if pd.isna(zi):
                signal[i] = pos
                continue
            if pos == 0:
                if zi <= -self.K_STD:
                    pos = 1
                elif zi >= self.K_STD:
                    pos = -1
            elif pos == 1:
                if zi >= -self.EXIT_K:    # close longs once we're near mean
                    pos = 0
            elif pos == -1:
                if zi <= self.EXIT_K:
                    pos = 0
            signal[i] = pos

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        # confidence = extremity of z-score capped
        out["confidence"] = np.clip(np.abs(z.fillna(0)) / 3.0, 0.0, 0.9)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient 4h bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        return StrategySignal(d, float(last["confidence"]),
                              "4h mean reversion (±2σ to mean)", {})
