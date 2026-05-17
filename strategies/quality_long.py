"""QualityLong (daily) — multi-filter long-only quality factor.

Pattern: every winning strategy in the portfolio is long-only with strong
filters. This combines the four orthogonal filters that have the best
individual win rates:

  A. Regime:      50d EMA > 200d EMA  (golden cross trend)
  B. Momentum:    90d return > 0
  C. Stability:   30d realized vol in [25%, 90%] annualized
  D. Not falling: 10d return > -8%   (avoid catching knives)

Enter long when ALL four hold; exit when ANY breaks. Long-only — BTC
structural drift means SHORT trades have low expected value AND high
catastrophic-loss probability in bull markets.

Vol-target sizing in the overlay caps single-day exposure. Tight stops
won't help (this is a regime strategy, not a swing strategy), so the
overlay typically chooses no SL with vol_target ~0.35.

Reasonable trade count: ~30-50/yr (one cycle per major regime).
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class QualityLong(BaseStrategy):
    name = "Quality Long"
    description = "Long when all of: 50/200 EMA cross, 90d mom>0, vol in band, not in 10d crash"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    EMA_FAST = 50
    EMA_SLOW = 200
    MOM_DAYS = 90
    VOL_DAYS = 30
    VOL_LO = 0.25
    VOL_HI = 0.95
    CRASH_DAYS = 10
    CRASH_THRESH = -0.08
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
        c = bars["close"].shift(1)  # no look-ahead
        ema_f = c.ewm(span=self.EMA_FAST, adjust=False, min_periods=self.EMA_FAST).mean()
        ema_s = c.ewm(span=self.EMA_SLOW, adjust=False, min_periods=self.EMA_SLOW).mean()
        mom = c / c.shift(self.MOM_DAYS) - 1
        log_ret = np.log(c / c.shift(1))
        rv = log_ret.rolling(self.VOL_DAYS, min_periods=10).std() * np.sqrt(365)
        crash = c / c.shift(self.CRASH_DAYS) - 1

        regime = ema_f > ema_s
        positive_mom = mom > 0
        vol_ok = (rv > self.VOL_LO) & (rv < self.VOL_HI)
        not_crashing = crash > self.CRASH_THRESH

        long_ok = regime & positive_mom & vol_ok & not_crashing
        signal = np.where(long_ok.fillna(False), 1, 0)

        # Confidence scales with momentum strength
        conf = np.where(long_ok.fillna(False),
                        np.clip(0.5 + mom.fillna(0) * 1.0, 0.5, 0.9), 0.0)

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
                              "Quality long (4-filter consensus)", {})
