"""Taker Flow Imbalance Strategy.

Thesis: When aggressive buyers significantly outnumber sellers (or vice versa),
the order flow imbalance is directional — indicating conviction in the move.
Computed as a z-score of normalized taker buy/sell imbalance over a 4h rolling window.

Data: binance_taker_buy_sell.csv (5-min buckets of taker buy/sell volume)
Frequency: ~2-4 signals/week at z-score ≥ 1.5 threshold
Correlation: Essentially zero with all 6 existing strategies (independent signal source)
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class TakerFlowImbalance(BaseStrategy):

    name = "Taker Flow Imbalance"
    description = "Detects directional conviction via buy/sell order flow z-score"
    data_files = [
        "binance_taker_buy_sell.csv",
    ]

    # Tunable parameters
    ZSCORE_WINDOW = 48    # 48 × 5min = 4h rolling window
    MIN_PERIODS = 12      # 12 × 5min = 1h minimum to produce a z-score
    ENTRY_Z = 1.5         # Z-score threshold to enter
    EXIT_Z = 0.5          # Z-score threshold to exit
    MAX_CONF = 0.82       # Confidence ceiling

    def _build_imbalance(self, data: dict) -> pd.DataFrame:
        """Parse 5-min taker buy/sell data and compute normalized imbalance."""
        df = data.get("binance_taker_buy_sell.csv", pd.DataFrame())
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        if "timestamp_ms" in df.columns:
            df["datetime"] = pd.to_datetime(
                pd.to_numeric(df["timestamp_ms"], errors="coerce"),
                unit="ms", utc=True,
            )
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], utc=True)
        else:
            return pd.DataFrame()

        for col in ["buy_vol", "sell_vol"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "buy_vol" not in df.columns or "sell_vol" not in df.columns:
            return pd.DataFrame()

        df = df.dropna(subset=["datetime", "buy_vol", "sell_vol"])
        df = df.sort_values("datetime").reset_index(drop=True)

        # Normalized imbalance: +1 = all buys, -1 = all sells
        total = df["buy_vol"] + df["sell_vol"]
        df["imbalance"] = np.where(
            total > 0,
            (df["buy_vol"] - df["sell_vol"]) / total,
            0.0,
        )

        # Rolling z-score (computed on available rows — ignores time gaps)
        roll = df["imbalance"].rolling(self.ZSCORE_WINDOW, min_periods=self.MIN_PERIODS)
        df["z_score"] = (df["imbalance"] - roll.mean()) / (roll.std() + 1e-12)
        df["date"] = df["datetime"].dt.date

        return df

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        """Compute full signal history (for backtesting), aggregated to daily."""
        df = self._build_imbalance(data)
        if df.empty or "z_score" not in df.columns:
            return pd.DataFrame()

        signal = np.zeros(len(df))
        confidence = np.zeros(len(df))
        in_position = 0

        for i in range(len(df)):
            z = df["z_score"].iloc[i]
            if pd.isna(z):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.97 if i > 0 else 0.0
                continue

            if z >= self.ENTRY_Z:
                signal[i] = 1  # LONG — buyers overpowering sellers
                confidence[i] = min(0.55 + (z - self.ENTRY_Z) * 0.08, self.MAX_CONF)
                in_position = 1
            elif z <= -self.ENTRY_Z:
                signal[i] = -1  # SHORT — sellers overpowering buyers
                confidence[i] = min(0.55 + (abs(z) - self.ENTRY_Z) * 0.08, self.MAX_CONF)
                in_position = -1
            elif in_position != 0 and abs(z) < self.EXIT_Z:
                signal[i] = 0
                confidence[i] = 0.0
                in_position = 0
            else:
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.97 if i > 0 else 0.0

        df["signal"] = signal.astype(int)
        df["confidence"] = confidence

        # Aggregate to daily for consistent backtest interface
        df_daily = df.groupby("date").agg(
            signal=("signal", "last"),
            confidence=("confidence", "last"),
            imbalance=("imbalance", "last"),
            z_score=("z_score", "last"),
        ).reset_index()

        return df_daily

    def compute_signal(self, data: dict) -> StrategySignal:
        """Return the current signal for the live pipeline."""
        df = self._build_imbalance(data)
        if df.empty or len(df) < self.MIN_PERIODS:
            return StrategySignal("INACTIVE", 0.0, "Insufficient taker flow data", {})

        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient taker flow data", {})

        last_daily = series.iloc[-1]
        sig = int(last_daily["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last_daily["confidence"])

        # Use latest intraday row for real-time z-score display
        latest = df.iloc[-1]
        z_now = float(latest["z_score"]) if not pd.isna(latest["z_score"]) else 0.0
        imb_now = float(latest["imbalance"])

        details = {
            "z_score": f"{z_now:.2f}",
            "imbalance": f"{imb_now:+.3f}",
            "rows_used": str(len(df)),
        }

        if direction == "LONG":
            expl = (f"Aggressive buyers dominating order flow "
                    f"(z={z_now:.2f}, imbalance={imb_now:+.3f})")
        elif direction == "SHORT":
            expl = (f"Aggressive sellers dominating order flow "
                    f"(z={z_now:.2f}, imbalance={imb_now:+.3f})")
        else:
            expl = f"Order flow balanced (z={z_now:.2f})"

        return StrategySignal(direction, conf, expl, details)
