"""Strategy: Commodity Risk Appetite.

Thesis: The copper/gold ratio is the cleanest cross-asset signal for global risk
appetite. Copper prices rise when industrial demand is strong (risk-on) while
gold rises in risk-off regimes. BTC behaves as a risk-on asset in macro regimes
where copper is outperforming gold. Crude oil provides a second-order confirmation:
rising oil confirms economic activity (risk-on), crashing oil signals recession fear
(risk-off). This strategy captures the macro regime shift BEFORE it appears in
crypto-specific data.

Differentiation from existing MacroRegime strategy: MacroRegime uses SPX returns,
DXY, VIX, yield curve, and credit spreads — financial market signals. This strategy
uses physical commodity prices, which represent real-world industrial demand dynamics
and have a different timing relative to BTC price moves.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class CommodityRiskAppetite(BaseStrategy):

    name = "Commodity Risk Appetite"
    description = (
        "Copper/gold ratio momentum as a real-economy risk-on/off signal for BTC"
    )
    data_files = [
        "macro_commodities.csv",
        "binance_futures_klines_5m.csv",
    ]

    # Tunable parameters
    RETURN_WINDOW = 3        # Days for return calculation (fast-moving signal)
    ZSCORE_WINDOW = 20       # Rolling window for z-score normalization
    ENTRY_Z = 0.75           # Z-score threshold for signal (both directions)
    EXIT_Z = 0.15            # Z-score threshold to exit position
    OIL_CRASH_THRESHOLD = -0.08  # Oil 5d return < -8% → skip new entries (crisis)
    PRICE_CONFIRM_WINDOW = 5     # BTC price confirmation days
    BULL_MARKET_WINDOW = 60      # Days for major BTC bull market filter
    BULL_MARKET_THRESHOLD = 0.30 # If BTC up 30%+ in 60 days, skip SHORT signals
    MIN_HISTORY = 40         # Minimum rows before generating signals

    def _load_commodities(self, data: dict) -> pd.DataFrame:
        df = data.get("macro_commodities.csv", pd.DataFrame())
        if df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

        for col in ["HGF_close", "GCF_close", "CLF_close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("date").dropna(subset=["HGF_close", "GCF_close"])
        return df

    def _load_price(self, data: dict) -> pd.DataFrame:
        klines = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()

        klines = klines.copy()
        klines["close"] = pd.to_numeric(klines["close"], errors="coerce")
        klines["ts_ms"] = pd.to_numeric(klines["open_time_ms"], errors="coerce")
        klines["date"] = pd.to_datetime(klines["ts_ms"], unit="ms", utc=True).dt.date
        price = klines.groupby("date")["close"].last().reset_index()
        price.columns = ["date", "btc_close"]
        return price

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        comm = self._load_commodities(data)
        if comm.empty or len(comm) < self.MIN_HISTORY:
            return pd.DataFrame()

        price = self._load_price(data)

        # Core ratio: copper / gold (risk-on when rising)
        comm["cu_au_ratio"] = comm["HGF_close"] / comm["GCF_close"]

        # N-day returns for the ratio and components
        comm["cu_au_ret"] = comm["cu_au_ratio"].pct_change(self.RETURN_WINDOW)
        comm["oil_ret"] = comm["CLF_close"].pct_change(self.RETURN_WINDOW) if "CLF_close" in comm.columns else 0.0

        # Z-score of copper/gold return (normalizes across regime changes)
        roll = comm["cu_au_ret"].rolling(self.ZSCORE_WINDOW, min_periods=15)
        comm["cu_au_z"] = (comm["cu_au_ret"] - roll.mean()) / (roll.std() + 1e-10)

        # Merge BTC price for confirmation
        if not price.empty:
            comm = comm.merge(price, on="date", how="left")
            comm["btc_close"] = comm["btc_close"].ffill()
            comm["btc_ret"] = comm["btc_close"].pct_change(self.PRICE_CONFIRM_WINDOW)
            comm["btc_long_ret"] = comm["btc_close"].pct_change(self.BULL_MARKET_WINDOW)
        else:
            comm["btc_ret"] = 0.0
            comm["btc_long_ret"] = np.nan

        # Generate signals
        signal = np.zeros(len(comm))
        confidence = np.zeros(len(comm))
        in_position = 0

        for i in range(self.MIN_HISTORY, len(comm)):
            z = comm["cu_au_z"].iloc[i]
            oil_ret = comm["oil_ret"].iloc[i] if "oil_ret" in comm.columns else 0.0
            btc_ret = comm["btc_ret"].iloc[i] if "btc_ret" in comm.columns else 0.0

            btc_long_ret = comm["btc_long_ret"].iloc[i] if "btc_long_ret" in comm.columns else np.nan

            if pd.isna(z):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.95 if in_position != 0 else 0.0
                continue

            # Oil crash filter: skip new entries during severe oil crashes
            # (indicates a macro dislocation where copper/gold signal unreliable)
            oil_crisis = (not pd.isna(oil_ret)) and (oil_ret < self.OIL_CRASH_THRESHOLD)
            # Bull market filter: skip SHORT entries when BTC is in a strong uptrend
            bull_mkt = (not pd.isna(btc_long_ret)) and (btc_long_ret > self.BULL_MARKET_THRESHOLD)

            if z > self.ENTRY_Z and not oil_crisis:
                # Risk-on: copper outperforming gold → LONG BTC
                # Strengthen if BTC price is already confirming
                signal[i] = 1
                base_conf = min(0.55 + (z - self.ENTRY_Z) * 0.12, 0.84)
                # Oil confirmation: if oil also up, stronger signal
                oil_boost = 0.04 if (not pd.isna(oil_ret) and oil_ret > 0.03) else 0.0
                # BTC price confirmation
                price_boost = 0.04 if (not pd.isna(btc_ret) and btc_ret > 0) else 0.0
                confidence[i] = min(base_conf + oil_boost + price_boost, 0.87)
                in_position = 1

            elif z < -self.ENTRY_Z and not oil_crisis and not bull_mkt:
                # Risk-off: gold outperforming copper → SHORT BTC
                signal[i] = -1
                base_conf = min(0.55 + (abs(z) - self.ENTRY_Z) * 0.12, 0.84)
                # Oil crashing adds to bearish conviction
                oil_boost = 0.04 if (not pd.isna(oil_ret) and oil_ret < -0.03) else 0.0
                price_boost = 0.04 if (not pd.isna(btc_ret) and btc_ret < 0) else 0.0
                confidence[i] = min(base_conf + oil_boost + price_boost, 0.87)
                in_position = -1

            elif in_position != 0:
                # Hold position until z-score normalizes
                if abs(z) < self.EXIT_Z:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = in_position
                    confidence[i] = confidence[i - 1] * 0.97
            else:
                signal[i] = 0
                confidence[i] = 0.0

        comm["signal"] = signal.astype(int)
        comm["confidence"] = confidence

        cols = ["date", "signal", "confidence", "cu_au_z", "cu_au_ret", "oil_ret", "btc_long_ret"]
        return comm[[c for c in cols if c in comm.columns]].copy()

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient commodity data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        z = last.get("cu_au_z", 0.0)
        cu_au_ret = last.get("cu_au_ret", 0.0)
        oil_ret = last.get("oil_ret", 0.0)

        details = {
            "cu_au_z": f"{z:.2f}",
            "cu_au_5d_ret": f"{cu_au_ret:.3f}" if not pd.isna(cu_au_ret) else "n/a",
            "oil_5d_ret": f"{oil_ret:.3f}" if not pd.isna(oil_ret) else "n/a",
        }

        if direction == "LONG":
            expl = (
                f"Copper/gold ratio z={z:.1f} (risk-on macro: copper outperforming gold)"
            )
        elif direction == "SHORT":
            expl = (
                f"Copper/gold ratio z={z:.1f} (risk-off macro: gold outperforming copper)"
            )
        else:
            expl = f"Copper/gold ratio in neutral zone (z={z:.1f})"

        return StrategySignal(direction, conf, expl, details)
