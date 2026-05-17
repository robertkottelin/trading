"""ETFFlowMomentum (daily) — long when BTC ETFs see strong recent volume.

Uses IBIT (BlackRock) + GBTC (Grayscale) daily VOLUME as a proxy for net
institutional flow. When 5-day cumulative volume z-score is high (heavy
inflows ramping), price tends to follow during the subsequent days.

Long-only signal. Activates only since 2024-01 when spot ETFs launched.
Largely uncorrelated with on-chain / TA / macro strategies.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from strategies.base import BaseStrategy, StrategySignal


class ETFFlowMomentum(BaseStrategy):
    name = "ETF Flow Momentum"
    description = "Long BTC when IBIT+GBTC 5d volume z-score high"
    data_files = ["binance_futures_klines_5m.csv", "macro_crypto_adjacent.csv"]

    bar_seconds = 86400

    LOOKBACK_DAYS = 5
    ZSCORE_WINDOW = 30
    Z_LONG = 0.8
    Z_EXIT = -0.2
    MIN_HISTORY = 60

    def _daily_close(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(close=("close", "last")).reset_index()
        return agg.rename(columns={"bucket": "ts_ms"}).sort_values("ts_ms").reset_index(drop=True)

    def _load_etf_volume(self, data: dict) -> pd.DataFrame:
        """Load IBIT+GBTC+FBTC daily volume from macro_crypto_adjacent.csv."""
        df = data.get("macro_crypto_adjacent.csv", pd.DataFrame())
        if df.empty:
            return pd.DataFrame(columns=["bucket", "volume"])
        vol_cols = [c for c in ("IBIT_volume", "GBTC_volume", "FBTC_volume")
                    if c in df.columns]
        if not vol_cols:
            return pd.DataFrame(columns=["bucket", "volume"])
        d = pd.to_datetime(df["date"], utc=True, errors="coerce")
        bucket = (d.astype("int64") // 1_000_000 // 86_400_000) * 86_400_000
        for c in vol_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        vol = df[vol_cols].sum(axis=1)
        out = pd.DataFrame({"bucket": bucket.values, "volume": vol.values})
        return out.sort_values("bucket").reset_index(drop=True)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily_close(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()

        etf = self._load_etf_volume(data)
        if etf.empty:
            return pd.DataFrame()  # no data — strategy inactive

        # Align ETF volume onto the daily bar grid (forward-fill weekends)
        et_by_bucket = etf.set_index("bucket")["volume"]
        vol = bars["ts_ms"].map(et_by_bucket).ffill().fillna(0.0)
        # rolling cumulative + z-score
        cum = vol.rolling(self.LOOKBACK_DAYS, min_periods=1).sum()
        m = cum.rolling(self.ZSCORE_WINDOW, min_periods=5).mean()
        s = cum.rolling(self.ZSCORE_WINDOW, min_periods=5).std()
        z = (cum - m) / s.replace(0, np.nan)
        z_prev = z.shift(1)  # use yesterday's z as input

        signal = np.zeros(len(bars), dtype=int)
        pos = 0
        for i in range(self.MIN_HISTORY, len(bars)):
            zi = z_prev.iloc[i]
            if pd.isna(zi):
                signal[i] = pos
                continue
            if pos == 0 and zi > self.Z_LONG:
                pos = 1
            elif pos == 1 and zi < self.Z_EXIT:
                pos = 0
            signal[i] = pos

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = np.where(signal == 1, 0.6, 0.0)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "No ETF data or insufficient history", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "ETF volume momentum", {})
