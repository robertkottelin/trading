"""WeeklyTrend (7d bars) — long-only momentum on weekly resolution.

Long-only weekly trend follower: position long if 4-week return positive
AND price above 26-week SMA; flat otherwise. Long-only avoids the typical
pain of shorting in a structurally appreciating asset (BTC). Trades roughly
every 1-3 months — extremely low turnover.

Sharpe-target rationale:
- BTC's drift has historically been ~+50%/yr, but with high vol
- Filtering out drawdown periods (when below 26w SMA) trims left tail
- 4-week momentum confirmation reduces noise
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class WeeklyTrend(BaseStrategy):
    name = "Weekly Trend"
    description = "Long-only weekly trend: 4w momentum AND price > 26w SMA"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 7 * 86400  # 1 week bars
    MOMENTUM_WEEKS = 4
    SMA_WEEKS = 26
    MIN_HISTORY = 30

    def _resample_weekly(self, df_5m: pd.DataFrame) -> pd.DataFrame:
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
        bars = self._resample_weekly(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()

        close = bars["close"]
        mom = close / close.shift(self.MOMENTUM_WEEKS) - 1
        sma = close.shift(1).rolling(self.SMA_WEEKS, min_periods=self.SMA_WEEKS).mean()

        signal = np.where((mom > 0) & (close > sma), 1, 0)
        # Confidence scales with momentum strength
        conf = np.clip(mom.fillna(0) * 2.0, 0.0, 0.9)
        conf = np.where(signal == 1, conf, 0.0)

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = conf
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient weekly bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Weekly trend (4w mom + above 26w SMA)", {})
