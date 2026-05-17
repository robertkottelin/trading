"""MomentumRegimeComposite (daily) — composite using LiquidityRegime template.

Inspired by LiquidityRegime's 1.54 Sharpe approach: TIGHT short-term filter
(15-day signal) + tight SMA (~36 days) + vol-target sizing.

Applies the same compact-filter approach but to a price-momentum signal
instead of macro:
  - 21-day return > X
  - close > 36d SMA
  - 7-day return is in healthy range (avoid late entries)
  - Pure long-only
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class MomentumRegimeComposite(BaseStrategy):
    name = "Momentum Regime Composite"
    description = "Tight 21d-momentum + 36d-SMA regime + 7d health filter, long-only"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    MOM_DAYS = 21
    MOM_THRESHOLD = 0.0
    SMA_DAYS = 36
    RECENT_DAYS = 7
    RECENT_MIN = -0.05
    RECENT_MAX = 0.20
    MIN_HISTORY = 70

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
        mom21 = c / c.shift(self.MOM_DAYS) - 1
        mom7 = c / c.shift(self.RECENT_DAYS) - 1
        sma = c.rolling(self.SMA_DAYS, min_periods=self.SMA_DAYS).mean()

        long_ok = (
            (mom21 > self.MOM_THRESHOLD)
            & (c > sma)
            & (mom7 > self.RECENT_MIN)
            & (mom7 < self.RECENT_MAX)
        )
        signal = np.where(long_ok.fillna(False), 1, 0)
        conf = np.where(long_ok.fillna(False),
                        np.clip(0.5 + mom21.fillna(0) * 2.0, 0.5, 0.9), 0.0)

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
                              "21d momentum + tight regime", {})
