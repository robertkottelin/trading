"""LowVolUptrend (daily) — long when vol contracts inside an uptrend.

Phenomenon: in BTC bull markets, periods of LOW realized vol (the 20th
percentile of trailing year) often precede the next leg up. Vol compression
inside an uptrend signals consolidation before continuation.

Logic:
  - Compute 30d realized vol
  - LONG when:
      * vol < 30th percentile of trailing 1-year vol  (compressed regime)
      * close > 50d EMA  (uptrend)
      * 7d return > -3% (not breaking down)
  - EXIT when:
      * vol > 60th percentile (regime broken — too volatile)
      * close < 50d EMA  (uptrend broken)

Pure regime-driven, complementary to momentum-driven strategies.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class LowVolUptrend(BaseStrategy):
    name = "Low Vol Uptrend"
    description = "Long during vol-compression regimes inside uptrends"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    VOL_DAYS = 30
    VOL_QUANT_WIN = 365
    VOL_LOW_PCTILE = 0.30
    VOL_HIGH_PCTILE = 0.60
    EMA_DAYS = 50
    SHORT_RET_DAYS = 7
    SHORT_RET_MIN = -0.03
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
        c = bars["close"].shift(1)
        log_ret = np.log(c / c.shift(1))
        rv = log_ret.rolling(self.VOL_DAYS, min_periods=10).std() * np.sqrt(365)
        low_thresh = rv.rolling(self.VOL_QUANT_WIN, min_periods=60).quantile(self.VOL_LOW_PCTILE)
        hi_thresh = rv.rolling(self.VOL_QUANT_WIN, min_periods=60).quantile(self.VOL_HIGH_PCTILE)
        ema = c.ewm(span=self.EMA_DAYS, adjust=False, min_periods=self.EMA_DAYS).mean()
        short_ret = c / c.shift(self.SHORT_RET_DAYS) - 1

        signal = np.zeros(len(bars), dtype=int)
        pos = 0
        for i in range(self.MIN_HISTORY, len(bars)):
            r = rv.iloc[i]
            lt = low_thresh.iloc[i]
            ht = hi_thresh.iloc[i]
            ci = c.iloc[i]
            ei = ema.iloc[i]
            sr = short_ret.iloc[i]
            if any(pd.isna(x) for x in (r, lt, ht, ci, ei, sr)):
                signal[i] = pos
                continue
            if pos == 0:
                if r < lt and ci > ei and sr > self.SHORT_RET_MIN:
                    pos = 1
            else:
                if r > ht or ci < ei:
                    pos = 0
            signal[i] = pos

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = np.where(signal == 1, 0.65, 0.0)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Low-vol regime + uptrend", {})
