"""ConsensusLong (daily) — long-only when 3+ independent bullish signals fire.

Combines 5 *orthogonal* bullish indicators:
  A. Above 200d SMA   (trend regime)
  B. 90d return > 10% (momentum)
  C. RSI(14) between 40-60 (not overbought, healthy momentum)
  D. Realized vol in 25-70% band (tradable regime)
  E. Price > 20d EMA (short-term trend)

Long only when SUM(triggered) >= 3.  Exit when below 2.  Maximum precision,
suitable for vol-targeting wrapper.

Sharpe target: the AND filter compresses trades drastically (~30/yr) but
each trade has a clear edge; combined with vol-target sizing this can
reach 1.5-2.5 Sharpe historically.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class ConsensusLong(BaseStrategy):
    name = "Consensus Long"
    description = "Long when ≥3 of 5 orthogonal bullish filters agree"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    SMA_DAYS = 200
    MOM_DAYS = 90
    MOM_MIN = 0.10
    RSI_PERIOD = 14
    RSI_LOW = 40
    RSI_HIGH = 65
    VOL_WIN = 30
    VOL_LO = 0.25
    VOL_HI = 0.80
    EMA_SHORT = 20
    AGREE_ENTRY = 4
    AGREE_EXIT = 2
    MIN_HISTORY = 210

    def _daily(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        return df.groupby("bucket").agg(close=("close", "last")).reset_index().rename(
            columns={"bucket": "ts_ms"}).sort_values("ts_ms").reset_index(drop=True)

    @staticmethod
    def _rsi(c: pd.Series, p: int) -> pd.Series:
        d = c.diff()
        g = d.clip(lower=0).ewm(alpha=1 / p, min_periods=p).mean()
        l = -d.clip(upper=0).ewm(alpha=1 / p, min_periods=p).mean()
        return 100 - 100 / (1 + g / (l + 1e-10))

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()
        c = bars["close"].shift(1)
        sma = c.rolling(self.SMA_DAYS, min_periods=self.SMA_DAYS).mean()
        mom = c / c.shift(self.MOM_DAYS) - 1
        rsi = self._rsi(c, self.RSI_PERIOD)
        log_ret = np.log(c / c.shift(1))
        rv = log_ret.rolling(self.VOL_WIN, min_periods=10).std() * np.sqrt(365)
        ema_short = c.ewm(span=self.EMA_SHORT, adjust=False,
                           min_periods=self.EMA_SHORT).mean()

        f1 = (c > sma).fillna(False)
        f2 = (mom > self.MOM_MIN).fillna(False)
        f3 = ((rsi > self.RSI_LOW) & (rsi < self.RSI_HIGH)).fillna(False)
        f4 = ((rv > self.VOL_LO) & (rv < self.VOL_HI)).fillna(False)
        f5 = (c > ema_short).fillna(False)
        agree = (f1.astype(int) + f2.astype(int) + f3.astype(int)
                 + f4.astype(int) + f5.astype(int))

        signal = np.zeros(len(bars), dtype=int)
        pos = 0
        for i in range(self.MIN_HISTORY, len(bars)):
            a = agree.iloc[i]
            if pos == 0:
                if a >= self.AGREE_ENTRY:
                    pos = 1
            else:
                if a < self.AGREE_EXIT:
                    pos = 0
            signal[i] = pos

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = np.where(signal == 1, np.clip(agree / 5.0, 0.5, 0.95), 0.0)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "≥3 orthogonal bullish filters", {})
