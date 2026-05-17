"""TrendPullback (daily) — long-only dip-buy in strong uptrends.

Buy when:
  - In confirmed uptrend (close > 200d SMA AND 90d return > 10%)
  - AND price pulled back: 3-day return < -3% OR RSI(14) < 35

Exit when RSI > 55 or 5 bars elapsed (max hold).

This captures the classic "buy the dip" edge in a structural bull market.
Long-only, low correlation with breakout/momentum strategies because it
fires ON the OPPOSITE move within the same trend.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class TrendPullback(BaseStrategy):
    name = "Trend Pullback"
    description = "Buy 3-day dips inside confirmed uptrends; RSI-exit"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    SMA_DAYS = 200
    MOMENTUM_DAYS = 90
    MOMENTUM_MIN = 0.10
    DIP_DAYS = 3
    DIP_THRESH = -0.03
    RSI_PERIOD = 14
    RSI_OVERSOLD = 35
    RSI_EXIT = 55
    MAX_HOLD_DAYS = 6
    MIN_HISTORY = 210

    def _daily(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(close=("close", "last")).reset_index()
        return agg.rename(columns={"bucket": "ts_ms"}).sort_values("ts_ms").reset_index(drop=True)

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_g = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_l = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_g / (avg_l + 1e-10)
        return 100 - 100 / (1 + rs)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()
        c = bars["close"].shift(1)
        sma = c.rolling(self.SMA_DAYS, min_periods=self.SMA_DAYS).mean()
        ret_mom = c / c.shift(self.MOMENTUM_DAYS) - 1
        ret_dip = c / c.shift(self.DIP_DAYS) - 1
        rsi = self._rsi(c, self.RSI_PERIOD)

        bull_regime = (c > sma) & (ret_mom > self.MOMENTUM_MIN)
        dip = (ret_dip < self.DIP_THRESH) | (rsi < self.RSI_OVERSOLD)

        signal = np.zeros(len(bars), dtype=int)
        bars_held = 0
        pos = 0
        for i in range(self.MIN_HISTORY, len(bars)):
            r_i = rsi.iloc[i]
            if pos == 0:
                if bool(bull_regime.iloc[i]) and bool(dip.iloc[i]) and not pd.isna(r_i):
                    pos = 1
                    bars_held = 0
            else:
                bars_held += 1
                if (not pd.isna(r_i) and r_i > self.RSI_EXIT) or bars_held >= self.MAX_HOLD_DAYS:
                    pos = 0
                # Also exit if regime broken
                if not bool(bull_regime.iloc[i]):
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
                              "Trend pullback dip-buy", {})
