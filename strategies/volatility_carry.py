"""VolatilityCarry (daily) — vol risk premium harvest.

When implied volatility (DVOL) > realized volatility, the asset offers a
positive vol risk premium for option SELLERS. We can't sell options directly
on dYdX, but the EQUIVALENT trade is a long-only directional position taken
when IV is RICHLY priced vs RV (suggests fear-driven option demand →
typically followed by mean-reverting calm + drift higher in spot).

Conversely, when IV < RV (calm option market vs realized chop), spot price
is more likely to break out hard in either direction — we stand aside.

Signal:
  LONG  when IV/RV > 1.20 AND price > 50d SMA
  FLAT  otherwise (no short — too risky given vol-event tail risk)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from strategies.base import BaseStrategy, StrategySignal


class VolatilityCarry(BaseStrategy):
    name = "Volatility Carry"
    description = "Long BTC when implied vol > realized vol (vol risk premium harvest)"
    data_files = [
        "binance_futures_klines_5m.csv",
        "deribit_dvol.csv",
    ]

    bar_seconds = 86400

    IV_RV_LONG = 1.20       # entry threshold
    IV_RV_EXIT = 0.95       # exit when premium evaporates
    SMA_DAYS = 50
    RV_WINDOW_HRS = 48      # realized vol over past 48h, annualized
    MIN_HISTORY = 60

    def _resample_daily(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(
            close=("close", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
        ).reset_index().rename(columns={"bucket": "ts_ms"})
        return agg.sort_values("ts_ms").reset_index(drop=True)

    def _load_iv_daily(self, data: dict, day_buckets: np.ndarray) -> pd.Series:
        dvol = data.get("deribit_dvol.csv", pd.DataFrame())
        if dvol.empty:
            return pd.Series(np.nan, index=range(len(day_buckets)))
        dvol = dvol.copy()
        dvol["ts_ms"] = pd.to_numeric(dvol["timestamp_ms"], errors="coerce").astype("int64")
        dvol["bucket"] = (dvol["ts_ms"] // 86_400_000) * 86_400_000
        daily_iv = dvol.groupby("bucket")["dvol_close"].last()
        # Align to provided day_buckets
        return pd.Series(day_buckets).map(daily_iv).ffill()

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._resample_daily(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()

        close = bars["close"].shift(1)  # prior-day close, no look-ahead
        # Realized vol (annualized %) from close-to-close log returns
        log_ret = np.log(close / close.shift(1))
        rv = log_ret.rolling(7, min_periods=5).std() * np.sqrt(365) * 100  # 7-day daily-bar
        rv = rv.fillna(method="bfill") if False else rv.bfill()

        iv = self._load_iv_daily(data, bars["ts_ms"].to_numpy())
        iv_shift = iv.shift(1)  # prior-day IV available before today's open

        sma = close.rolling(self.SMA_DAYS, min_periods=self.SMA_DAYS).mean()
        ratio = iv_shift / rv.replace(0, np.nan)

        signal = np.zeros(len(bars), dtype=int)
        confidence = np.zeros(len(bars))
        pos = 0
        for i in range(self.MIN_HISTORY, len(bars)):
            r = ratio.iloc[i]
            sma_i = sma.iloc[i]
            c_prev = close.iloc[i]
            if pd.isna(r) or pd.isna(sma_i) or pd.isna(c_prev):
                signal[i] = pos
                continue
            if pos == 0:
                if r > self.IV_RV_LONG and c_prev > sma_i:
                    pos = 1
            else:
                if r < self.IV_RV_EXIT or c_prev < sma_i:
                    pos = 0
            signal[i] = pos
            confidence[i] = float(min(0.4 + max(r - 1.0, 0) * 0.8, 0.9)) if pos != 0 else 0.0

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = confidence
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient data", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Vol risk premium harvest (IV > RV)", {})
