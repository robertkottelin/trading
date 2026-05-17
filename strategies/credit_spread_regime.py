"""CreditSpreadRegime (daily) — long BTC when credit spreads narrowing.

BAMLH0A0HYM2 (US High Yield credit spread) is a risk-on/risk-off indicator.
NARROWING spreads = risk-on regime → BTC tends to rally.
WIDENING spreads = risk-off → BTC under pressure.

Designed in the LiquidityRegime template: TIGHT short-term signal + tight
SMA filter + vol-target sizing (delivered Sharpe 1.54 for liquidity).

Logic:
  - Compute 15-day change in HY credit spread (BAMLH0A0HYM2)
  - LONG when spread_change < 0 AND BTC > 36d SMA
  - FLAT otherwise

Uses macro_credit.csv.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class CreditSpreadRegime(BaseStrategy):
    name = "Credit Spread Regime"
    description = "Long BTC when HY credit spread narrowing AND BTC above SMA"
    data_files = ["binance_futures_klines_5m.csv", "macro_credit.csv"]

    bar_seconds = 86400

    LOOKBACK_DAYS = 15
    SMA_DAYS = 36
    CHANGE_THRESH = 0.0    # spread must be falling (negative change)
    MIN_HISTORY = 60

    def _daily_btc(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(close=("close", "last")).reset_index()
        return agg.rename(columns={"bucket": "ts_ms"}).sort_values("ts_ms").reset_index(drop=True)

    def _load_credit(self, data: dict) -> pd.DataFrame:
        df = data.get("macro_credit.csv", pd.DataFrame())
        if df.empty or "BAMLH0A0HYM2" not in df.columns:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        out["hy"] = pd.to_numeric(df["BAMLH0A0HYM2"], errors="coerce")
        out["ts_ms"] = (out["date"].astype("int64") // 1_000_000 // 86_400_000) * 86_400_000
        return out.dropna(subset=["hy"]).sort_values("ts_ms").drop_duplicates(
            "ts_ms", keep="last")[["ts_ms", "hy"]].reset_index(drop=True)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily_btc(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()
        credit = self._load_credit(data)
        if credit.empty:
            return pd.DataFrame()

        merged = bars[["ts_ms", "close"]].merge(credit, on="ts_ms", how="left")
        merged["hy"] = merged["hy"].ffill()

        hy = merged["hy"].shift(1)
        hy_chg = hy - hy.shift(self.LOOKBACK_DAYS)   # absolute change in pct points

        c = merged["close"].shift(1)
        sma = c.rolling(self.SMA_DAYS, min_periods=self.SMA_DAYS).mean()

        long_ok = (hy_chg < self.CHANGE_THRESH) & (c > sma)
        signal = np.where(long_ok.fillna(False), 1, 0)
        conf = np.where(long_ok.fillna(False),
                        np.clip(0.5 + (-hy_chg.fillna(0)) * 0.5, 0.5, 0.9), 0.0)

        out = merged[["ts_ms"]].copy()
        out["signal"] = signal.astype(int)
        out["confidence"] = conf
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient credit data", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Credit spread narrowing regime", {})
