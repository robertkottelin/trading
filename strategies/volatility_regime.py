"""Strategy 2: Volatility Regime & IV/RV Divergence.

Thesis: Low realized volatility compresses before explosive moves. DVOL
(implied vol) diverging from realized vol reveals market expectations.
Expensive protection (high IV/RV) is contrarian bullish; cheap options
(low IV/RV) with complacency is bearish.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class VolatilityRegime(BaseStrategy):

    name = "Volatility Regime"
    description = "Trades IV/RV divergence and vol compression breakouts"
    data_files = [
        "deribit_dvol.csv",
        "binance_futures_klines_5m.csv",
    ]

    # Tunable parameters
    RV_WINDOW_HOURS = 24       # Realized vol window
    DVOL_PERCENTILE_WINDOW = 90  # Days for percentile ranking
    RV_COMPRESSION_PCTILE = 15   # Below this = vol compressed
    IV_RV_HIGH = 1.4            # Above this = overpriced protection
    IV_RV_LOW = 0.7             # Below this = underpriced risk
    IV_RV_EXIT_HIGH = 1.15      # Exit when IV/RV normalizes
    IV_RV_EXIT_LOW = 0.85
    DVOL_MOMENTUM_HOURS = 24   # DVOL change lookback

    def _build_daily_vol_data(self, data: dict) -> pd.DataFrame:
        """Build daily DataFrame with DVOL and realized vol."""
        # Load DVOL (hourly)
        dvol_df = data.get("deribit_dvol.csv", pd.DataFrame())
        if dvol_df.empty or "dvol_close" not in dvol_df.columns:
            return pd.DataFrame()

        dvol_df["ts_ms"] = pd.to_numeric(dvol_df["timestamp_ms"], errors="coerce")
        dvol_df["dvol_close"] = pd.to_numeric(dvol_df["dvol_close"], errors="coerce")
        dvol_df["datetime"] = pd.to_datetime(dvol_df["ts_ms"], unit="ms", utc=True)
        dvol_df["date"] = dvol_df["datetime"].dt.date

        # Resample DVOL to daily (last value)
        dvol_daily = dvol_df.groupby("date").agg(
            dvol=("dvol_close", "last"),
            dvol_high=("dvol_close", "max") if "dvol_high" not in dvol_df.columns
            else ("dvol_high", "max"),
            dvol_low=("dvol_close", "min") if "dvol_low" not in dvol_df.columns
            else ("dvol_low", "min"),
        ).reset_index()

        # Compute DVOL momentum (daily change)
        dvol_daily["dvol_momentum"] = dvol_daily["dvol"].diff()

        # Load Binance futures for realized vol
        bnc = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if bnc.empty or "close" not in bnc.columns:
            return pd.DataFrame()

        bnc["close"] = pd.to_numeric(bnc["close"], errors="coerce")
        bnc["ts_ms"] = pd.to_numeric(bnc["open_time_ms"], errors="coerce")
        bnc["datetime"] = pd.to_datetime(bnc["ts_ms"], unit="ms", utc=True)
        bnc = bnc.sort_values("ts_ms")

        # 5m log returns
        bnc["log_ret"] = np.log(bnc["close"] / bnc["close"].shift(1))
        bnc["date"] = bnc["datetime"].dt.date

        # Daily realized vol: std of 5m returns * sqrt(288) * sqrt(365) * 100
        # 288 candles per day, annualize
        daily_rv = bnc.groupby("date")["log_ret"].std().reset_index()
        daily_rv.columns = ["date", "rv_daily_raw"]
        daily_rv["rv_annualized"] = daily_rv["rv_daily_raw"] * np.sqrt(288) * np.sqrt(365) * 100

        # Merge
        merged = dvol_daily.merge(daily_rv, on="date", how="inner")
        merged = merged.sort_values("date").reset_index(drop=True)
        return merged

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        df = self._build_daily_vol_data(data)
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        # RV percentile rank (rolling)
        window = self.DVOL_PERCENTILE_WINDOW
        df["rv_pctile"] = df["rv_annualized"].rolling(window, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # DVOL percentile rank
        df["dvol_pctile"] = df["dvol"].rolling(window, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # IV/RV ratio
        df["iv_rv_ratio"] = df["dvol"] / (df["rv_annualized"] + 1e-6)

        # Generate signals
        signal = np.zeros(len(df))
        confidence = np.zeros(len(df))
        in_position = 0

        for i in range(1, len(df)):
            rv_pct = df["rv_pctile"].iloc[i]
            iv_rv = df["iv_rv_ratio"].iloc[i]
            dvol_mom = df["dvol_momentum"].iloc[i]

            if pd.isna(rv_pct) or pd.isna(iv_rv):
                signal[i] = in_position
                confidence[i] = 0.0
                continue

            # Signal 1: Vol compression + DVOL direction
            if rv_pct * 100 < self.RV_COMPRESSION_PCTILE:
                if dvol_mom > 0:
                    signal[i] = 1  # LONG — market expects upside breakout
                    confidence[i] = min(0.55 + (self.RV_COMPRESSION_PCTILE - rv_pct * 100) * 0.01, 0.80)
                    in_position = 1
                    continue
                elif dvol_mom < 0:
                    signal[i] = -1  # SHORT — market expects downside
                    confidence[i] = min(0.55 + (self.RV_COMPRESSION_PCTILE - rv_pct * 100) * 0.01, 0.80)
                    in_position = -1
                    continue

            # Signal 2: IV/RV mean reversion
            if iv_rv > self.IV_RV_HIGH:
                signal[i] = 1  # LONG — expensive protection, fear overblown
                confidence[i] = min(0.55 + (iv_rv - self.IV_RV_HIGH) * 0.2, 0.85)
                in_position = 1
            elif iv_rv < self.IV_RV_LOW:
                signal[i] = -1  # SHORT — cheap options, complacency
                confidence[i] = min(0.55 + (self.IV_RV_LOW - iv_rv) * 0.3, 0.85)
                in_position = -1
            elif in_position != 0:
                # Exit when IV/RV normalizes
                if self.IV_RV_EXIT_LOW < iv_rv < self.IV_RV_EXIT_HIGH:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = in_position
                    confidence[i] = confidence[i - 1] * 0.95
            else:
                signal[i] = 0
                confidence[i] = 0.0

        df["signal"] = signal.astype(int)
        df["confidence"] = confidence

        return df[["date", "signal", "confidence", "dvol", "rv_annualized",
                    "iv_rv_ratio", "rv_pctile", "dvol_momentum"]].copy()

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient vol data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        details = {
            "dvol": f"{last['dvol']:.1f}",
            "rv_24h": f"{last['rv_annualized']:.1f}",
            "iv_rv_ratio": f"{last['iv_rv_ratio']:.2f}",
            "rv_pctile": f"{last['rv_pctile'] * 100:.0f}th",
            "dvol_momentum": f"{last['dvol_momentum']:+.1f}",
        }

        if direction == "LONG":
            if last["rv_pctile"] * 100 < self.RV_COMPRESSION_PCTILE:
                expl = f"Vol compression (RV {last['rv_pctile']*100:.0f}th pctile) + DVOL rising"
            else:
                expl = f"IV/RV={last['iv_rv_ratio']:.2f} (overpriced protection, fear overblown)"
        elif direction == "SHORT":
            if last["rv_pctile"] * 100 < self.RV_COMPRESSION_PCTILE:
                expl = f"Vol compression (RV {last['rv_pctile']*100:.0f}th pctile) + DVOL falling"
            else:
                expl = f"IV/RV={last['iv_rv_ratio']:.2f} (cheap options, complacency)"
        else:
            expl = f"Vol in normal range (IV/RV={last['iv_rv_ratio']:.2f}, RV pctile={last['rv_pctile']*100:.0f}th)"

        return StrategySignal(direction, conf, expl, details)
