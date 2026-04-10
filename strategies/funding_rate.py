"""Strategy 1: Funding Rate Mean Reversion.

Thesis: Extreme cross-exchange funding rates create economic pressure for
positioning to reverse. Overleveraged longs (high positive funding) → selling
pressure → price drops. Overleveraged shorts (high negative funding) → buying
pressure → price rises.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class FundingRateReversion(BaseStrategy):

    name = "Funding Rate Mean Reversion"
    description = "Fades extreme cross-exchange funding rates"
    data_files = [
        "binance_funding_rates.csv",
        "bybit_funding_rates.csv",
        "coinalyze_funding_daily.csv",
        "coinalyze_oi_daily.csv",
        "binance_futures_klines_5m.csv",
    ]

    # Tunable parameters
    SHORT_ZSCORE = 60   # rolling window for short-term z-score (in funding periods)
    LONG_ZSCORE = 180   # rolling window for long-term z-score
    ENTRY_Z_SHORT = 2.0
    ENTRY_Z_LONG = 1.5
    EXIT_Z = 0.5
    OI_BOOST_Z = 1.0   # OI z-score threshold for confidence boost
    RATE_EXTREME = 0.0003  # Absolute rate must exceed this for entry
    PRICE_TREND_WINDOW = 7  # Days for price trend confirmation

    def _merge_funding_rates(self, data: dict) -> pd.DataFrame:
        """Merge Binance and Bybit funding rates into a weighted average series."""
        frames = []
        for fname, ts_col, rate_col in [
            ("binance_funding_rates.csv", "funding_time_ms", "funding_rate"),
            ("bybit_funding_rates.csv", "funding_time_ms", "funding_rate"),
        ]:
            df = data.get(fname, pd.DataFrame())
            if df.empty or rate_col not in df.columns:
                continue
            df = df[[ts_col, rate_col]].copy()
            df[rate_col] = pd.to_numeric(df[rate_col], errors="coerce")
            df["ts_ms"] = pd.to_numeric(df[ts_col], errors="coerce")
            df = df.dropna(subset=["ts_ms", rate_col])
            df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.date
            # Keep 8h resolution — group by date + 8h bucket
            df["period"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.floor("8h")
            agg = df.groupby("period")[rate_col].last().reset_index()
            agg.columns = ["period", fname.split("_")[0] + "_rate"]
            frames.append(agg)

        if not frames:
            return pd.DataFrame()

        merged = frames[0]
        for f in frames[1:]:
            merged = merged.merge(f, on="period", how="outer")
        merged = merged.sort_values("period").reset_index(drop=True)

        rate_cols = [c for c in merged.columns if c.endswith("_rate")]
        merged["avg_funding"] = merged[rate_cols].mean(axis=1)
        merged = merged.dropna(subset=["avg_funding"])
        return merged

    def _add_coinalyze_daily(self, funding: pd.DataFrame,
                              data: dict) -> pd.DataFrame:
        """Merge Coinalyze daily funding and OI for confirmation."""
        funding["date"] = funding["period"].dt.date

        # Coinalyze daily OI
        oi = data.get("coinalyze_oi_daily.csv", pd.DataFrame())
        if not oi.empty and "open_interest_close" in oi.columns:
            oi["ts_ms"] = pd.to_numeric(oi["timestamp_ms"], errors="coerce")
            oi["date"] = pd.to_datetime(oi["ts_ms"], unit="ms", utc=True).dt.date
            oi = oi[["date", "open_interest_close"]].drop_duplicates("date", keep="last")
            oi["open_interest_close"] = pd.to_numeric(oi["open_interest_close"], errors="coerce")
            funding = funding.merge(oi, on="date", how="left")
            funding["open_interest_close"] = funding["open_interest_close"].ffill()

        return funding

    def _add_price_data(self, funding: pd.DataFrame, data: dict) -> pd.DataFrame:
        """Add price trend for confirmation filter."""
        bnc = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if bnc.empty:
            # Try loading directly
            from pathlib import Path
            path = Path("raw_data") / "binance_futures_klines_5m.csv"
            if path.exists():
                bnc = pd.read_csv(path)
        if bnc.empty:
            funding["price_trend"] = 0
            return funding

        bnc["close"] = pd.to_numeric(bnc["close"], errors="coerce")
        bnc["ts_ms"] = pd.to_numeric(bnc["open_time_ms"], errors="coerce")
        bnc["date"] = pd.to_datetime(bnc["ts_ms"], unit="ms", utc=True).dt.date
        price_daily = bnc.groupby("date")["close"].last().reset_index()
        price_daily.columns = ["date", "btc_price"]

        # 7-day price return
        price_daily["price_ret_7d"] = price_daily["btc_price"].pct_change(self.PRICE_TREND_WINDOW)
        funding = funding.merge(price_daily[["date", "price_ret_7d"]], on="date", how="left")
        funding["price_ret_7d"] = funding["price_ret_7d"].ffill()
        return funding

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        funding = self._merge_funding_rates(data)
        if funding.empty:
            return pd.DataFrame()

        funding = self._add_coinalyze_daily(funding, data)
        funding = self._add_price_data(funding, data)

        # Compute z-scores
        roll_short = funding["avg_funding"].rolling(self.SHORT_ZSCORE, min_periods=20)
        roll_long = funding["avg_funding"].rolling(self.LONG_ZSCORE, min_periods=40)
        funding["z_short"] = (funding["avg_funding"] - roll_short.mean()) / (roll_short.std() + 1e-10)
        funding["z_long"] = (funding["avg_funding"] - roll_long.mean()) / (roll_long.std() + 1e-10)

        # Funding rate momentum (is the extreme reverting?)
        funding["rate_momentum"] = funding["avg_funding"].diff(3)

        # OI z-score for confidence boost
        if "open_interest_close" in funding.columns:
            oi_roll = funding["open_interest_close"].rolling(90, min_periods=20)
            funding["oi_z"] = (funding["open_interest_close"] - oi_roll.mean()) / (oi_roll.std() + 1e-10)
        else:
            funding["oi_z"] = 0.0

        # Generate signals with price trend confirmation
        signal = np.zeros(len(funding))
        confidence = np.zeros(len(funding))
        in_position = 0

        for i in range(1, len(funding)):
            z_s = funding["z_short"].iloc[i]
            z_l = funding["z_long"].iloc[i]
            oi_z = funding["oi_z"].iloc[i]
            rate = funding["avg_funding"].iloc[i]
            rate_mom = funding["rate_momentum"].iloc[i] if "rate_momentum" in funding.columns else 0
            price_ret = funding["price_ret_7d"].iloc[i] if "price_ret_7d" in funding.columns else 0

            if pd.isna(z_s) or pd.isna(z_l):
                signal[i] = in_position
                confidence[i] = 0.0
                continue

            # Entry conditions — require z-score extreme + rate level + momentum confirmation
            rate_extreme = abs(rate) > self.RATE_EXTREME

            if (z_s > self.ENTRY_Z_SHORT and z_l > self.ENTRY_Z_LONG
                    and rate_extreme and rate > 0):
                # Only SHORT if price is not in a strong uptrend (avoid fighting momentum)
                if pd.isna(price_ret) or price_ret < 0.05:  # Not up >5% in 7d
                    # Prefer when rate is starting to decline (momentum turning)
                    mom_ok = pd.isna(rate_mom) or rate_mom <= 0
                    signal[i] = -1  # SHORT
                    base_conf = min(0.55 + abs(z_s) * 0.08, 0.85)
                    if mom_ok:
                        base_conf += 0.05
                    confidence[i] = min(base_conf + (0.05 if oi_z > self.OI_BOOST_Z else 0), 0.90)
                    in_position = -1
                else:
                    signal[i] = in_position
                    confidence[i] = confidence[i - 1] * 0.9 if in_position != 0 else 0
            elif (z_s < -self.ENTRY_Z_SHORT and z_l < -self.ENTRY_Z_LONG
                      and rate_extreme and rate < 0):
                # Only LONG if price is not in a strong downtrend
                if pd.isna(price_ret) or price_ret > -0.05:
                    mom_ok = pd.isna(rate_mom) or rate_mom >= 0
                    signal[i] = 1  # LONG
                    base_conf = min(0.55 + abs(z_s) * 0.08, 0.85)
                    if mom_ok:
                        base_conf += 0.05
                    confidence[i] = min(base_conf + (0.05 if oi_z > self.OI_BOOST_Z else 0), 0.90)
                    in_position = 1
                else:
                    signal[i] = in_position
                    confidence[i] = confidence[i - 1] * 0.9 if in_position != 0 else 0
            elif in_position != 0:
                # Exit when z-score normalizes OR confidence has decayed too far
                decayed_conf = confidence[i - 1] * 0.95
                if abs(z_s) < self.EXIT_Z or decayed_conf < 0.40:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = in_position
                    confidence[i] = decayed_conf
            else:
                signal[i] = 0
                confidence[i] = 0.0

        funding["signal"] = signal.astype(int)
        funding["confidence"] = confidence
        funding["date_col"] = funding["period"]

        return funding[["date_col", "signal", "confidence", "z_short", "z_long",
                         "avg_funding", "oi_z"]].rename(columns={"date_col": "date"})

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient funding data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        details = {
            "avg_funding": f"{last['avg_funding']:.6f}",
            "z_short": f"{last['z_short']:.2f}",
            "z_long": f"{last['z_long']:.2f}",
            "oi_zscore": f"{last['oi_z']:.2f}",
        }

        if direction == "LONG":
            expl = f"Cross-exchange funding z-score at {last['z_short']:.1f} (overleveraged shorts)"
        elif direction == "SHORT":
            expl = f"Cross-exchange funding z-score at {last['z_short']:.1f} (overleveraged longs)"
        else:
            expl = f"Funding rates in normal range (z={last['z_short']:.1f})"

        return StrategySignal(direction, conf, expl, details)
