"""WeekendEffect (daily) — exploits documented BTC Fri-Mon drift bias.

Multiple academic studies (Kaiser 2019, Aharon & Qadan 2020) document that
BTC tends to drift UP from Friday close to Monday open. Mechanism: lower
exchange volume on weekends, lighter sell-pressure from miners/exchanges,
retail buying continues. We capture this with a long position held only
from Friday close to Monday close.

Signal:
  LONG on Friday's signal bar (executed at Saturday open)
  FLAT on Monday's signal bar (executed at Tuesday open) — i.e. 3-day hold

Average hold ~3 days, ~52 trades/year. Very low correlation with momentum/
trend strategies since timing is calendar-based, not price-based.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class WeekendEffect(BaseStrategy):
    name = "Weekend Effect"
    description = "Long-only Friday close → Monday close to capture weekend drift"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400
    MIN_HISTORY = 10

    def _daily(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(
            close=("close", "last"),
            open=("open", "first"),
        ).reset_index().rename(columns={"bucket": "ts_ms"})
        agg = agg.sort_values("ts_ms").reset_index(drop=True)
        agg["dt"] = pd.to_datetime(agg["ts_ms"], unit="ms", utc=True)
        agg["dow"] = agg["dt"].dt.dayofweek  # Mon=0 .. Sun=6
        return agg

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()

        # Position desired AT END of bar i is consumed at bar i+1 open (engine shifts).
        # We want to be LONG during Sat, Sun, Mon (dow 5, 6, 0).
        # That means signal at end of Fri (dow=4) → LONG for Sat open.
        # Signal at end of Mon (dow=0) → FLAT for Tue open.
        signal = np.where(bars["dow"].isin([4, 5, 6]), 1, 0)
        out = bars[["ts_ms"]].copy()
        out["signal"] = signal.astype(int)
        out["confidence"] = np.where(signal == 1, 0.55, 0.0)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Weekend Fri-Mon drift", {})
