"""FastRotationLong (daily) — short-horizon long-only in tight regime.

Built using the LiquidityRegime / MomentumRegimeComposite template that
delivered Sharpe 1.54 / 1.60:
  - Tight short-term signal (7-day strength)
  - Tight regime filter (21-day EMA)
  - Health check (medium momentum positive)
  - Vol-target sizing in overlay

Logic:
  LONG when:
    7d_return > 3%           (recent buying)
    21d_return > 0           (not just a bounce)
    close > 21d EMA          (above short MA)
    realized vol (14d) > 25% AND < 95%   (tradable regime)
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class FastRotationLong(BaseStrategy):
    name = "Fast Rotation Long"
    description = "7d-strength + 21d-trend + EMA regime + vol band, long-only"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    FAST_DAYS = 7
    FAST_THRESH = 0.03
    MED_DAYS = 21
    MED_THRESH = 0.0
    EMA_DAYS = 21
    VOL_DAYS = 14
    VOL_LO = 0.25
    VOL_HI = 0.95
    MIN_HISTORY = 50

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
        fast = c / c.shift(self.FAST_DAYS) - 1
        med = c / c.shift(self.MED_DAYS) - 1
        ema = c.ewm(span=self.EMA_DAYS, adjust=False, min_periods=self.EMA_DAYS).mean()
        log_ret = np.log(c / c.shift(1))
        rv = log_ret.rolling(self.VOL_DAYS, min_periods=10).std() * np.sqrt(365)

        long_ok = (
            (fast > self.FAST_THRESH)
            & (med > self.MED_THRESH)
            & (c > ema)
            & (rv > self.VOL_LO)
            & (rv < self.VOL_HI)
        )
        signal = np.where(long_ok.fillna(False), 1, 0)
        conf = np.where(long_ok.fillna(False),
                        np.clip(0.5 + fast.fillna(0) * 3.0, 0.5, 0.9), 0.0)

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
                              "7d strength + 21d trend + vol band", {})
