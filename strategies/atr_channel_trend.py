"""ATRChannelTrend (daily) — Keltner-channel breakout trend follower.

LONG when daily close breaks above 20-day EMA + 2*ATR(14).
SHORT when daily close breaks below 20-day EMA - 2*ATR(14).
Position held until close crosses back through the 20-EMA (mid-line).

ATR-based bands adapt to volatility unlike Bollinger Bands which use stdev
of close (sensitive to outliers). This produces fewer but cleaner signals.
Typically pairs well with vol-targeting overlay.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class ATRChannelTrend(BaseStrategy):
    name = "ATR Channel Trend"
    description = "Keltner-style breakout: EMA20 ± 2*ATR14, exit at EMA mid"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    EMA_PERIOD = 20
    ATR_PERIOD = 14
    ATR_MULT = 2.0
    MIN_HISTORY = 60

    def _daily_ohlc(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        for c in ["open", "high", "low", "close"]:
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

    @staticmethod
    def _wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int) -> pd.Series:
        prev = close.shift(1)
        tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()],
                       axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily_ohlc(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()
        # Use prior bars only — shift everything by 1
        high = bars["high"].shift(1)
        low = bars["low"].shift(1)
        close_prev = bars["close"].shift(1)
        ema = close_prev.ewm(span=self.EMA_PERIOD, adjust=False,
                              min_periods=self.EMA_PERIOD).mean()
        atr = self._wilder_atr(high, low, close_prev, self.ATR_PERIOD)
        upper = ema + self.ATR_MULT * atr
        lower = ema - self.ATR_MULT * atr

        c = bars["close"]
        signal = np.zeros(len(bars), dtype=int)
        pos = 0
        for i in range(self.MIN_HISTORY, len(bars)):
            if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
                signal[i] = pos
                continue
            ci = c.iloc[i]
            ei = ema.iloc[i]
            if pos == 0:
                if ci > upper.iloc[i]:
                    pos = 1
                elif ci < lower.iloc[i]:
                    pos = -1
            elif pos == 1:
                if ci < ei:
                    pos = 0
            elif pos == -1:
                if ci > ei:
                    pos = 0
            signal[i] = pos

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = np.where(signal != 0, 0.6, 0.0)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        return StrategySignal(d, float(last["confidence"]),
                              "ATR channel breakout", {})
