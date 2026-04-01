"""Strategy 6 (Backup): Spot-Futures Basis Reversion.

Thesis: The perpetual premium index mean-reverts. Extreme positive basis
= market overleveraged long. Extreme negative = overleveraged short.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class BasisReversion(BaseStrategy):

    name = "Basis Reversion"
    description = "Fades extreme spot-futures basis/premium"
    data_files = [
        "binance_premium_index_klines.csv",
    ]

    # Tunable parameters
    RESAMPLE_PERIOD = "4h"
    ZSCORE_WINDOW = 336      # 336 × 4h = 56 days
    ENTRY_Z = 2.5
    EXIT_Z = 0.8
    LONG_ZSCORE_WINDOW = 1008  # 1008 × 4h = 168 days

    def _build_resampled_data(self, data: dict) -> pd.DataFrame:
        """Resample 5m premium index to 4h bars."""
        df = data.get("binance_premium_index_klines.csv", pd.DataFrame())
        if df.empty or "close" not in df.columns:
            return pd.DataFrame()

        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce")
        df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.sort_values("datetime")

        # Resample to 4h
        df = df.set_index("datetime")
        resampled = df["close"].resample(self.RESAMPLE_PERIOD).last().dropna()
        result = resampled.reset_index()
        result.columns = ["datetime", "premium"]
        result["date"] = result["datetime"].dt.date
        return result

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        df = self._build_resampled_data(data)
        if df.empty or len(df) < 50:
            return pd.DataFrame()

        # Z-scores
        roll = df["premium"].rolling(self.ZSCORE_WINDOW, min_periods=30)
        df["z_score"] = (df["premium"] - roll.mean()) / (roll.std() + 1e-12)

        roll_long = df["premium"].rolling(self.LONG_ZSCORE_WINDOW, min_periods=50)
        df["z_long"] = (df["premium"] - roll_long.mean()) / (roll_long.std() + 1e-12)

        # Generate signals
        signal = np.zeros(len(df))
        confidence = np.zeros(len(df))
        in_position = 0

        for i in range(1, len(df)):
            z = df["z_score"].iloc[i]
            z_l = df["z_long"].iloc[i]

            if pd.isna(z):
                signal[i] = in_position
                confidence[i] = 0.0
                continue

            if z > self.ENTRY_Z:
                signal[i] = -1  # SHORT — extreme premium
                confidence[i] = min(0.55 + (z - self.ENTRY_Z) * 0.1, 0.85)
                in_position = -1
            elif z < -self.ENTRY_Z:
                signal[i] = 1  # LONG — extreme discount
                confidence[i] = min(0.55 + abs(z + self.ENTRY_Z) * 0.1, 0.85)
                in_position = 1
            elif in_position != 0:
                if abs(z) < self.EXIT_Z:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = in_position
                    confidence[i] = confidence[i - 1] * 0.98
            else:
                signal[i] = 0
                confidence[i] = 0.0

        df["signal"] = signal.astype(int)
        df["confidence"] = confidence

        # Resample to daily for consistent backtesting
        df_daily = df.groupby("date").agg(
            signal=("signal", "last"),
            confidence=("confidence", "last"),
            premium=("premium", "last"),
            z_score=("z_score", "last"),
        ).reset_index()

        return df_daily

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient basis data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        details = {
            "premium": f"{last['premium']:.6f}",
            "z_score": f"{last['z_score']:.2f}",
        }

        if direction == "LONG":
            expl = f"Extreme basis discount (z={last['z_score']:.1f}, premium={last['premium']:.6f})"
        elif direction == "SHORT":
            expl = f"Extreme basis premium (z={last['z_score']:.1f}, premium={last['premium']:.6f})"
        else:
            expl = f"Basis in normal range (z={last['z_score']:.1f})"

        return StrategySignal(direction, conf, expl, details)
