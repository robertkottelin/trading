"""EthBtcRotation (daily) — long BTC when ETH/BTC ratio momentum reverses.

ETH/BTC has been in long-term DOWNTREND for most of 2022-2026 (BTC outperforms).
The signal: when ETH/BTC starts rising (alt momentum returning) AND BTC own
trend is intact, both can rally together. When ETH/BTC weakens with BTC
strong, often a sign of BTC dominance phase — ALSO bullish for BTC.

Practical signal: long BTC when ETH/BTC momentum > -X% (not in free-fall)
AND BTC above its 50d SMA. This captures the regime where BTC is leading
without ETH crashing.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class EthBtcRotation(BaseStrategy):
    name = "ETH/BTC Rotation"
    description = "Long BTC when ETH/BTC ratio breaks up through 50d SMA (altseason proxy)"
    data_files = ["binance_futures_klines_5m.csv", "macro_crypto_adjacent.csv"]

    bar_seconds = 86400

    SMA_DAYS = 50
    RATIO_MOMENTUM_DAYS = 14
    MOMENTUM_MIN = -0.05    # tolerate up to -5% over 2 weeks (not free-falling)
    BTC_SMA_DAYS = 50
    MIN_HISTORY = 80

    def _daily_btc(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(close=("close", "last")).reset_index()
        return agg.rename(columns={"bucket": "ts_ms"}).sort_values("ts_ms").reset_index(drop=True)

    def _load_ratio(self, data: dict) -> pd.DataFrame:
        df = data.get("macro_crypto_adjacent.csv", pd.DataFrame())
        if df.empty or "ETHUSD_close" not in df.columns or "BTCUSD_close" not in df.columns:
            return pd.DataFrame()
        out = pd.DataFrame()
        out["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        eth = pd.to_numeric(df["ETHUSD_close"], errors="coerce")
        btc = pd.to_numeric(df["BTCUSD_close"], errors="coerce")
        out["ratio"] = (eth / btc.replace(0, np.nan)).astype(float)
        out["ts_ms"] = (out["date"].astype("int64") // 1_000_000 // 86_400_000) * 86_400_000
        out = out.dropna(subset=["ratio"]).sort_values("ts_ms")
        out = out.drop_duplicates("ts_ms", keep="last").reset_index(drop=True)
        return out[["ts_ms", "ratio"]]

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily_btc(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty:
            return pd.DataFrame()
        ratio_df = self._load_ratio(data)
        if ratio_df.empty:
            return pd.DataFrame()

        # Align ratio to BTC daily bars
        merged = bars[["ts_ms"]].merge(ratio_df, on="ts_ms", how="left")
        merged["ratio"] = merged["ratio"].ffill()
        if len(merged) < self.MIN_HISTORY:
            return pd.DataFrame()

        r = merged["ratio"].shift(1)
        r_mom = r / r.shift(self.RATIO_MOMENTUM_DAYS) - 1

        # BTC also needs to be in uptrend (regime filter)
        bars2 = self._daily_btc(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        bars2 = bars2.rename(columns={"close": "btc_close"})
        merged = merged.merge(bars2[["ts_ms", "btc_close"]], on="ts_ms", how="left")
        bc = merged["btc_close"].shift(1)
        btc_sma = bc.rolling(self.BTC_SMA_DAYS, min_periods=self.BTC_SMA_DAYS).mean()

        # Long when ratio is not free-falling AND BTC is in uptrend
        rising = r_mom > self.MOMENTUM_MIN
        btc_trend = bc > btc_sma
        long_ok = rising & btc_trend

        signal = np.where(long_ok.fillna(False), 1, 0)
        conf = np.where(long_ok.fillna(False),
                        np.clip(0.5 + r_mom.fillna(0) * 5.0, 0.5, 0.9), 0.0)

        out = merged[["ts_ms"]].copy()
        out["signal"] = signal.astype(int)
        out["confidence"] = conf
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "No ETH/BTC data", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "ETH/BTC rotation (altseason proxy)", {})
