"""LiquidityRegime (daily) — long when global liquidity is expanding.

Thesis: BTC is highly sensitive to global liquidity. M2 money supply (WM2NS)
and Fed balance sheet (WALCL) drive risk-asset valuations. When liquidity
EXPANDS (WALCL up + WM2 up), BTC rallies. When it contracts, BTC drifts
lower.

Signal:
  - Compute 4-week change in WALCL and WM2 (weekly data, ffill to daily)
  - Composite "liquidity score" = 0.5*WALCL_4w + 0.5*WM2_4w
  - LONG when score > 0 AND BTC close > 50d SMA (regime filter)
  - FLAT otherwise (no short — liquidity-driven bear markets are unpredictable)

Uses macro_liquidity.csv. Pure macro signal, ORTHOGONAL to TA on BTC.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class LiquidityRegime(BaseStrategy):
    name = "Liquidity Regime"
    description = "Long BTC when WALCL+M2 4-week change positive AND price > 50d SMA"
    data_files = ["binance_futures_klines_5m.csv", "macro_liquidity.csv"]

    bar_seconds = 86400

    LOOKBACK_DAYS = 28
    SMA_DAYS = 50
    SCORE_MIN = 0.0
    MIN_HISTORY = 80

    def _daily_btc(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(close=("close", "last")).reset_index()
        return agg.rename(columns={"bucket": "ts_ms"}).sort_values("ts_ms").reset_index(drop=True)

    def _load_liquidity(self, data: dict) -> pd.DataFrame:
        df = data.get("macro_liquidity.csv", pd.DataFrame())
        if df.empty:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        out["WALCL"] = pd.to_numeric(df.get("WALCL"), errors="coerce")
        out["WM2NS"] = pd.to_numeric(df.get("WM2NS"), errors="coerce")
        out["ts_ms"] = (out["date"].astype("int64") // 1_000_000 // 86_400_000) * 86_400_000
        out = out.sort_values("ts_ms").drop_duplicates("ts_ms", keep="last").reset_index(drop=True)
        return out[["ts_ms", "WALCL", "WM2NS"]]

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily_btc(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()

        liq = self._load_liquidity(data)
        if liq.empty:
            return pd.DataFrame()

        merged = bars[["ts_ms", "close"]].merge(liq, on="ts_ms", how="left")
        merged["WALCL"] = merged["WALCL"].ffill()
        merged["WM2NS"] = merged["WM2NS"].ffill()

        # Use prior day's values for both series
        walcl = merged["WALCL"].shift(1)
        wm2 = merged["WM2NS"].shift(1)
        walcl_chg = walcl / walcl.shift(self.LOOKBACK_DAYS) - 1
        wm2_chg = wm2 / wm2.shift(self.LOOKBACK_DAYS) - 1
        score = 0.5 * walcl_chg + 0.5 * wm2_chg

        c = merged["close"].shift(1)
        sma = c.rolling(self.SMA_DAYS, min_periods=self.SMA_DAYS).mean()

        long_ok = (score > self.SCORE_MIN) & (c > sma)
        signal = np.where(long_ok.fillna(False), 1, 0)
        conf = np.where(long_ok.fillna(False),
                        np.clip(0.5 + score.fillna(0) * 50, 0.5, 0.9), 0.0)

        out = merged[["ts_ms"]].copy()
        out["signal"] = signal.astype(int)
        out["confidence"] = conf
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient macro data", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Liquidity regime (WALCL+M2 expansion)", {})
