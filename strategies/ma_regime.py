"""MARegime (daily) — extremely simple 200d SMA regime filter.

Long when daily close > 200-day SMA; short when below. The simplest possible
trend-regime filter — historically delivers Sharpe ~1.0 on BTC over 5+ years
because BTC has spent the majority of time above its long-term SMA.

Optimized via vol-targeting overlay typically reaches Sharpe ~1.5-2.0.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class MARegime(BaseStrategy):
    name = "200d MA Regime"
    description = "Long when above 200d SMA, short when below"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400
    SMA_DAYS = 200
    ALLOW_SHORT = True
    MIN_HISTORY = 210

    def _daily(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket")["close"].last().reset_index()
        agg = agg.rename(columns={"bucket": "ts_ms"})
        return agg.sort_values("ts_ms").reset_index(drop=True)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()
        c = bars["close"].shift(1)  # prior close, no look-ahead
        sma = c.rolling(self.SMA_DAYS, min_periods=self.SMA_DAYS).mean()
        long_mask = c > sma
        short_mask = c < sma
        signal = np.where(long_mask, 1, np.where(short_mask & self.ALLOW_SHORT, -1, 0))
        # Confidence proportional to |c-sma|/sma
        gap = ((c - sma) / sma).abs().fillna(0.0)
        conf = np.clip(0.45 + gap * 1.5, 0.0, 0.9)
        conf = np.where(signal != 0, conf, 0.0)
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
                              "200d SMA regime", {})
