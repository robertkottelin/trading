"""DXYRegime (daily) — long BTC when USD weakens.

Negative correlation between DXY (US Dollar Index) and risk assets including
BTC. When DXY breaks below 50d SMA AND DXY 10d return is negative, dollar
weakness fuels BTC rallies. Conversely, dollar strength caps BTC upside.

Signal:
  LONG when DXY_close < DXY_SMA50 AND DXY_10d_return < 0
  FLAT otherwise (no short — BTC's positive drift dominates dollar effects)

Uses macro_fx.csv. Pure FX-driven signal → uncorrelated with TA strategies.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class DXYRegime(BaseStrategy):
    name = "DXY Regime"
    description = "Long BTC when DXY is weakening (price < 50d SMA AND 10d return < 0)"
    data_files = ["binance_futures_klines_5m.csv", "macro_fx.csv"]

    bar_seconds = 86400

    SMA_DAYS = 50
    RET_DAYS = 10
    RET_MAX = 0.0
    BTC_SMA = 200       # secondary safety: BTC also needs above 200d SMA
    MIN_HISTORY = 220

    def _daily_btc(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(close=("close", "last")).reset_index()
        return agg.rename(columns={"bucket": "ts_ms"}).sort_values("ts_ms").reset_index(drop=True)

    def _load_dxy(self, data: dict) -> pd.DataFrame:
        df = data.get("macro_fx.csv", pd.DataFrame())
        if df.empty or "DXYNYB_close" not in df.columns:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        out["dxy"] = pd.to_numeric(df["DXYNYB_close"], errors="coerce")
        out["ts_ms"] = (out["date"].astype("int64") // 1_000_000 // 86_400_000) * 86_400_000
        out = out.dropna(subset=["dxy"]).sort_values("ts_ms")
        return out.drop_duplicates("ts_ms", keep="last")[["ts_ms", "dxy"]].reset_index(drop=True)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily_btc(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()
        dxy = self._load_dxy(data)
        if dxy.empty:
            return pd.DataFrame()

        merged = bars[["ts_ms", "close"]].merge(dxy, on="ts_ms", how="left")
        merged["dxy"] = merged["dxy"].ffill()

        d = merged["dxy"].shift(1)
        d_sma = d.rolling(self.SMA_DAYS, min_periods=self.SMA_DAYS).mean()
        d_ret = d / d.shift(self.RET_DAYS) - 1

        c = merged["close"].shift(1)
        btc_sma = c.rolling(self.BTC_SMA, min_periods=self.BTC_SMA).mean()

        long_ok = (d < d_sma) & (d_ret < self.RET_MAX) & (c > btc_sma)
        signal = np.where(long_ok.fillna(False), 1, 0)
        conf = np.where(long_ok.fillna(False),
                        np.clip(0.5 + (-d_ret.fillna(0)) * 5.0, 0.5, 0.9), 0.0)

        out = merged[["ts_ms"]].copy()
        out["signal"] = signal.astype(int)
        out["confidence"] = conf
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient FX data", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "DXY weakness regime", {})
