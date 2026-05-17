"""CapitulationReversal (daily) — buy first up-day after N consecutive down days.

Empirical pattern: 4+ consecutive down days in BTC ofen mark short-term capitulation.
The first GREEN day after that often kicks off a 5-15% rally. Combine with a
regime filter (price > 200d SMA) to avoid catching falling knives in bears.

Logic:
  - Count consecutive down days (close[i] < close[i-1])
  - LONG when streak >= 4 AND today is up AND price > 200d EMA
  - EXIT when 5 bars elapsed OR price < entry * 0.95

Long-only. Trades are infrequent but each one captures a sharp bounce.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class CapitulationReversal(BaseStrategy):
    name = "Capitulation Reversal"
    description = "Long on 1st up-day after 4+ consecutive down days, in bull regime"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    MIN_DOWN_STREAK = 4
    REGIME_EMA = 200
    HOLD_BARS = 5
    STOP_PCT = 0.05
    MIN_HISTORY = 220

    def _daily(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(close=("close", "last")).reset_index()
        return agg.rename(columns={"bucket": "ts_ms"}).sort_values("ts_ms").reset_index(drop=True)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()
        c = bars["close"].shift(1)        # prior close
        ema = c.ewm(span=self.REGIME_EMA, adjust=False,
                     min_periods=self.REGIME_EMA).mean()

        # Compute "consecutive down days ending at i-1"
        down = (c.diff() < 0).astype(int)
        streak = down.copy()
        for i in range(1, len(streak)):
            if down.iloc[i] == 1:
                streak.iloc[i] = streak.iloc[i - 1] + 1

        # Entry condition: yesterday had streak >= 4; today is closing UP (close > yesterday's close)
        today_close = bars["close"]
        prev_close = c
        today_up = today_close > prev_close

        regime_ok = c > ema
        entry = (streak.shift(1) >= self.MIN_DOWN_STREAK) & today_up & regime_ok

        signal = np.zeros(len(bars), dtype=int)
        pos = 0
        entry_price = 0.0
        bars_held = 0
        for i in range(self.MIN_HISTORY, len(bars)):
            cp_now = bars["close"].iloc[i]
            if pd.isna(cp_now):
                signal[i] = pos
                continue
            if pos == 0:
                if bool(entry.iloc[i]):
                    pos = 1
                    entry_price = cp_now
                    bars_held = 0
            else:
                bars_held += 1
                if bars_held >= self.HOLD_BARS:
                    pos = 0
                elif cp_now < entry_price * (1 - self.STOP_PCT):
                    pos = 0
            signal[i] = pos

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = np.where(signal == 1, 0.7, 0.0)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Capitulation reversal bounce", {})
