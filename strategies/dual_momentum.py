"""DualMomentum (daily) — Gary Antonacci style with vol filter.

Two-factor momentum: long when BOTH 30d AND 90d returns are positive AND
realized vol is in a "tradable" range (not too high, not too low). The dual
horizon kills false signals: a single 30d uptick in a downtrend doesn't pass
the 90d filter, and vice versa.

Vol filter range: 20%-70% annualized. Excluding ultra-low vol periods
(typical of pre-breakout chop) and ultra-high vol periods (capitulation /
parabola) historically lifts Sharpe substantially.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class DualMomentum(BaseStrategy):
    name = "Dual Momentum"
    description = "Long when 30d AND 90d return > 0 AND realized vol in tradable range"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    FAST_LOOKBACK = 30
    SLOW_LOOKBACK = 90
    VOL_WINDOW = 30
    VOL_LOWER = 0.25     # annualized
    VOL_UPPER = 0.90
    ALLOW_SHORT = True
    SHORT_LOOKBACK = 60
    SHORT_THRESH = -0.10
    MIN_HISTORY = 110

    def _daily(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(close=("close", "last")).reset_index()
        agg = agg.rename(columns={"bucket": "ts_ms"})
        return agg.sort_values("ts_ms").reset_index(drop=True)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()
        c = bars["close"].shift(1)  # no look-ahead

        ret_fast = c / c.shift(self.FAST_LOOKBACK) - 1
        ret_slow = c / c.shift(self.SLOW_LOOKBACK) - 1
        ret_short = c / c.shift(self.SHORT_LOOKBACK) - 1
        log_ret = np.log(c / c.shift(1))
        rv = log_ret.rolling(self.VOL_WINDOW, min_periods=10).std() * np.sqrt(365)

        long_ok = (ret_fast > 0) & (ret_slow > 0) & (rv > self.VOL_LOWER) & (rv < self.VOL_UPPER)
        short_ok = (self.ALLOW_SHORT & (ret_short < self.SHORT_THRESH)
                    & (rv > self.VOL_LOWER) & (rv < self.VOL_UPPER))

        signal = np.where(long_ok, 1, np.where(short_ok, -1, 0))
        conf = np.zeros(len(bars))
        conf[long_ok.to_numpy()] = 0.65
        conf[short_ok.to_numpy()] = 0.55

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
        d = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        return StrategySignal(d, float(last["confidence"]),
                              "Dual-horizon momentum + vol filter", {})
