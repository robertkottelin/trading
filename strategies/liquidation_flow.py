"""Strategy 4: Liquidation Cascade & Positioning Momentum.

Thesis: Liquidation cascades are self-reinforcing but exhaust. OI-price
divergence reveals crowded positioning. Extreme L/S ratios are contrarian.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class LiquidationFlow(BaseStrategy):

    name = "Liquidation & Positioning"
    description = "Trades liquidation cascades, OI divergence, and L/S extremes"
    data_files = [
        "coinalyze_liquidations_daily.csv",
        "coinalyze_long_short_ratio_daily.csv",
        "coinalyze_oi_daily.csv",
        "binance_futures_klines_5m.csv",
    ]

    # Tunable parameters
    LIQ_ZSCORE_WINDOW = 30
    LIQ_ENTRY_Z = 1.2
    OI_CHANGE_WINDOW = 5     # days for OI change
    PRICE_CHANGE_WINDOW = 5  # days for price change
    LS_ZSCORE_WINDOW = 45
    LS_ENTRY_Z = 1.2
    LONG_THRESHOLD = 1.0
    SHORT_THRESHOLD = -1.0

    def _build_daily_data(self, data: dict) -> pd.DataFrame:
        """Merge liquidation, L/S ratio, OI, and price data."""
        # Daily price from Binance futures
        bnc = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if bnc.empty:
            return pd.DataFrame()

        bnc["close"] = pd.to_numeric(bnc["close"], errors="coerce")
        bnc["ts_ms"] = pd.to_numeric(bnc["open_time_ms"], errors="coerce")
        bnc["date"] = pd.to_datetime(bnc["ts_ms"], unit="ms", utc=True).dt.date
        price_daily = bnc.groupby("date")["close"].last().reset_index()
        price_daily.columns = ["date", "price"]

        result = price_daily.copy()

        # Liquidations
        liq = data.get("coinalyze_liquidations_daily.csv", pd.DataFrame())
        if not liq.empty and "long_liquidations" in liq.columns:
            liq["ts_ms"] = pd.to_numeric(liq["timestamp_ms"], errors="coerce")
            liq["date"] = pd.to_datetime(liq["ts_ms"], unit="ms", utc=True).dt.date
            for col in ["long_liquidations", "short_liquidations"]:
                liq[col] = pd.to_numeric(liq[col], errors="coerce")
            liq = liq[["date", "long_liquidations", "short_liquidations"]].drop_duplicates("date", keep="last")
            liq["liq_imbalance"] = liq["long_liquidations"] - liq["short_liquidations"]
            result = result.merge(liq, on="date", how="left")

        # L/S ratio
        ls = data.get("coinalyze_long_short_ratio_daily.csv", pd.DataFrame())
        if not ls.empty and "ls_ratio" in ls.columns:
            ls["ts_ms"] = pd.to_numeric(ls["timestamp_ms"], errors="coerce")
            ls["date"] = pd.to_datetime(ls["ts_ms"], unit="ms", utc=True).dt.date
            ls["ls_ratio"] = pd.to_numeric(ls["ls_ratio"], errors="coerce")
            ls = ls[["date", "ls_ratio"]].drop_duplicates("date", keep="last")
            result = result.merge(ls, on="date", how="left")

        # OI
        oi = data.get("coinalyze_oi_daily.csv", pd.DataFrame())
        if not oi.empty and "open_interest_close" in oi.columns:
            oi["ts_ms"] = pd.to_numeric(oi["timestamp_ms"], errors="coerce")
            oi["date"] = pd.to_datetime(oi["ts_ms"], unit="ms", utc=True).dt.date
            oi["open_interest_close"] = pd.to_numeric(oi["open_interest_close"], errors="coerce")
            oi = oi[["date", "open_interest_close"]].drop_duplicates("date", keep="last")
            result = result.merge(oi, on="date", how="left")

        result = result.sort_values("date").reset_index(drop=True)
        result = result.ffill()
        return result

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        df = self._build_daily_data(data)
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        sub_signals = pd.DataFrame(index=df.index)

        # Sub-signal 1: Liquidation imbalance z-score
        if "liq_imbalance" in df.columns:
            roll = df["liq_imbalance"].rolling(self.LIQ_ZSCORE_WINDOW, min_periods=10)
            df["liq_z"] = (df["liq_imbalance"] - roll.mean()) / (roll.std() + 1e-10)
            # Positive z = more long liqs = shorts winning → SHORT
            # Negative z = more short liqs = longs winning → LONG
            sub_signals["liq_signal"] = np.where(
                df["liq_z"] > self.LIQ_ENTRY_Z, -1,
                np.where(df["liq_z"] < -self.LIQ_ENTRY_Z, 1, 0)
            )
        else:
            df["liq_z"] = 0
            sub_signals["liq_signal"] = 0

        # Sub-signal 2: OI-Price divergence
        if "open_interest_close" in df.columns:
            df["oi_chg"] = df["open_interest_close"].pct_change(self.OI_CHANGE_WINDOW)
            df["price_chg"] = df["price"].pct_change(self.PRICE_CHANGE_WINDOW)

            oi_sig = np.zeros(len(df))
            for i in range(len(df)):
                oi_c = df["oi_chg"].iloc[i]
                p_c = df["price_chg"].iloc[i]
                if pd.isna(oi_c) or pd.isna(p_c):
                    continue
                if oi_c > 0.02 and p_c < -0.01:
                    # OI rising + price falling = shorts building → squeeze risk → LONG
                    oi_sig[i] = 1.0
                elif oi_c > 0.02 and p_c > 0.01:
                    # OI rising + price rising = longs building → correction risk → SHORT
                    oi_sig[i] = -0.5
                elif oi_c < -0.02:
                    # OI falling = deleveraging → neutral
                    oi_sig[i] = 0
            sub_signals["oi_divergence"] = oi_sig
        else:
            df["oi_chg"] = 0
            df["price_chg"] = 0
            sub_signals["oi_divergence"] = 0

        # Sub-signal 3: L/S ratio contrarian
        if "ls_ratio" in df.columns:
            roll = df["ls_ratio"].rolling(self.LS_ZSCORE_WINDOW, min_periods=20)
            df["ls_z"] = (df["ls_ratio"] - roll.mean()) / (roll.std() + 1e-10)
            sub_signals["ls_signal"] = np.where(
                df["ls_z"] > self.LS_ENTRY_Z, -1,  # Crowd very long → contrarian SHORT
                np.where(df["ls_z"] < -self.LS_ENTRY_Z, 1, 0)  # Crowd very short → contrarian LONG
            )
        else:
            df["ls_z"] = 0
            sub_signals["ls_signal"] = 0

        # Combine sub-signals
        sig_cols = [c for c in sub_signals.columns]
        df["combined_score"] = sub_signals[sig_cols].sum(axis=1)

        # Smooth slightly
        df["score_ema"] = df["combined_score"].ewm(span=2, min_periods=1).mean()

        signal = np.zeros(len(df))
        confidence = np.zeros(len(df))

        for i in range(1, len(df)):
            s = df["score_ema"].iloc[i]
            if pd.isna(s):
                continue
            if s > self.LONG_THRESHOLD:
                signal[i] = 1
                confidence[i] = min(0.55 + (s - self.LONG_THRESHOLD) * 0.1, 0.85)
            elif s < self.SHORT_THRESHOLD:
                signal[i] = -1
                confidence[i] = min(0.55 + abs(s - self.SHORT_THRESHOLD) * 0.1, 0.85)

        df["signal"] = signal.astype(int)
        df["confidence"] = confidence

        return df[["date", "signal", "confidence", "combined_score", "liq_z",
                    "ls_z"]].copy()

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient positioning data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        details = {
            "liq_zscore": f"{last['liq_z']:.2f}",
            "ls_ratio_zscore": f"{last['ls_z']:.2f}",
            "combined_score": f"{last['combined_score']:+.2f}",
        }

        if direction == "LONG":
            expl = f"Positioning favors longs (score={last['combined_score']:+.1f})"
        elif direction == "SHORT":
            expl = f"Positioning favors shorts (score={last['combined_score']:+.1f})"
        else:
            expl = f"Positioning neutral (score={last['combined_score']:+.1f})"

        return StrategySignal(direction, conf, expl, details)
