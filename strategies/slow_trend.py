"""SlowTrend (daily) — long-horizon trend filter with strict regime.

Long-only when:
  60d return > 0 AND
  120d return > 0 AND
  close > 100d EMA

This is a SLOW-MOVING strategy that catches multi-month trends. Long
holding periods mean fewer trades but each trade has high Sharpe because
we sit out chop. Vol-target sizing in overlay handles position sizing
during regime entries.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class SlowTrend(BaseStrategy):
    name = "Slow Trend"
    description = "60d AND 120d momentum positive AND close > 100d EMA"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    SHORT_DAYS = 60
    LONG_DAYS = 120
    EMA_DAYS = 100
    MIN_HISTORY = 140

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
        c = bars["close"].shift(1)
        r1 = c / c.shift(self.SHORT_DAYS) - 1
        r2 = c / c.shift(self.LONG_DAYS) - 1
        ema = c.ewm(span=self.EMA_DAYS, adjust=False, min_periods=self.EMA_DAYS).mean()
        long_ok = (r1 > 0) & (r2 > 0) & (c > ema)
        signal = np.where(long_ok.fillna(False), 1, 0)
        conf = np.where(long_ok.fillna(False),
                        np.clip(0.5 + r2.fillna(0) * 0.6, 0.5, 0.9), 0.0)
        out = bars[["ts_ms"]].copy()
        out["signal"] = signal.astype(int)
        out["confidence"] = conf
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Slow trend (60d AND 120d momentum + EMA)", {})
