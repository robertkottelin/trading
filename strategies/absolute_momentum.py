"""AbsoluteMomentum (daily) — time-series momentum, the most robust effect in finance.

Long-only: long when 90-day return is positive, flat otherwise. Optional
short: when 90-day return is highly negative AND below 200d SMA.

Decades of evidence on stocks/futures/FX/crypto: simple absolute momentum
(also called "time-series momentum") earns positive risk-adjusted returns.
On BTC, the daily-bar version typically lands Sharpe 1.0-1.5; combined with
vol-targeting via the SimConfig overlay, it can push above 2.0.

Differentiation: pure return-based (no oscillators), so corr with
indicator-driven strategies is low.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class AbsoluteMomentum(BaseStrategy):
    name = "Absolute Momentum"
    description = "90-day time-series momentum, long-only with optional short tail"
    data_files = ["binance_futures_klines_5m.csv"]

    # Daily bars
    bar_seconds = 86400

    LOOKBACK_DAYS = 90
    SHORT_LOOKBACK_DAYS = 200   # 200d SMA filter for shorts
    SHORT_THRESH = -0.25        # short only if 90d return < -25%
    ALLOW_SHORT = True
    MIN_HISTORY = 220

    def _resample_daily(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        if df_5m.empty:
            return pd.DataFrame()
        df = df_5m.copy()
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(
            close=("close", "last"),
            high=("high", "max"),
            low=("low", "min"),
            open=("open", "first"),
        ).reset_index().rename(columns={"bucket": "ts_ms"})
        return agg.sort_values("ts_ms").reset_index(drop=True)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._resample_daily(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()

        close = bars["close"]
        # Use yesterday's close for both signal and filter — no look-ahead
        c_prev = close.shift(1)
        ret_90 = c_prev / c_prev.shift(self.LOOKBACK_DAYS) - 1
        sma_200 = c_prev.rolling(self.SHORT_LOOKBACK_DAYS, min_periods=self.SHORT_LOOKBACK_DAYS).mean()

        signal = np.zeros(len(bars), dtype=int)
        confidence = np.zeros(len(bars))
        for i in range(self.MIN_HISTORY, len(bars)):
            r = ret_90.iloc[i]
            sma = sma_200.iloc[i]
            cp = c_prev.iloc[i]
            if pd.isna(r) or pd.isna(sma):
                continue
            if r > 0:
                signal[i] = 1
                confidence[i] = float(min(0.4 + r * 1.5, 0.9))
            elif self.ALLOW_SHORT and r < self.SHORT_THRESH and cp < sma:
                signal[i] = -1
                confidence[i] = float(min(0.4 + abs(r) * 1.5, 0.9))
            else:
                signal[i] = 0
                confidence[i] = 0.0

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = confidence
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient daily bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        return StrategySignal(d, float(last["confidence"]),
                              "90d absolute momentum", {})
