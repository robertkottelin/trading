"""Strategy: Funding Rate Carry Momentum.

Thesis: When the perpetual futures market shows PERSISTENT moderate positive funding
(longs consistently paying carry for 2-3+ days), it signals a "bull carry regime"
where directional momentum is confirmed by the derivatives market. This is distinct
from the existing Funding Rate Reversion strategy which FADES extremes — this
strategy FOLLOWS persistent carry trends that are not yet extreme enough to trigger
a reversion.

Economic logic: If longs are willing to keep paying 0.01-0.02%/8h for multiple
periods without the price reversing, it means the market has genuine directional
conviction — not just a short-term crowded spike. Conversely, persistent negative
funding (shorts paying) means sellers have conviction. This captures trend
continuation in the derivatives market.

Differentiation: The existing FundingRateReversion strategy fires when z_short > 2.0
(extreme zone). This strategy targets the 0.5-1.5 z-score range — moderate but
persistent carry. They should rarely fire simultaneously.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class FundingCarryMomentum(BaseStrategy):

    name = "Funding Carry Momentum"
    description = (
        "Follows persistent moderate funding rate direction as trend continuation signal"
    )
    data_files = [
        "binance_funding_rates.csv",
        "bybit_funding_rates.csv",
        "binance_futures_klines_5m.csv",
    ]

    # Tunable parameters — use moderate thresholds to avoid overlap with FundingReversion
    CARRY_WINDOW = 6       # Periods to sum for "persistent carry" (6 × 8h = 2 days)
    TREND_WINDOW = 14      # Periods for trend average (14 × 8h = 4.7 days)
    ENTRY_CUM = 0.00030    # 2-day cumulative funding threshold for LONG (2d × avg 0.015%/8h)
    ENTRY_CUM_SHORT = -0.00015  # 2-day cumulative funding threshold for SHORT
    EXIT_CUM_LONG = 0.00001     # Tight exit — escape quickly on reversals
    EXIT_CUM_SHORT = -0.00001   # Mirror tight exit for SHORT
    EXTREME_Z = 1.5        # Skip new entries if FundingReversion is about to fire
    ZSCORE_WINDOW = 180    # Periods for z-score computation (matches FundingReversion)
    PRICE_WINDOW = 5       # Days for price trend confirmation
    SMA_WINDOW = 100       # BTC 100-day SMA trend filter — only LONG when BTC above trend
    MIN_HISTORY = 60       # Minimum periods before generating signals (ensures SMA warm)

    def _load_funding(self, data: dict) -> pd.DataFrame:
        """Load and merge Binance + Bybit funding rates."""
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
        merged["date"] = merged["period"].dt.date
        return merged

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
        funding = self._load_funding(data)
        if funding.empty or len(funding) < self.MIN_HISTORY:
            return pd.DataFrame()

        price = self._load_price(data)

        # --- Core carry metrics ---

        # 1. Cumulative funding over CARRY_WINDOW periods (3 days)
        funding["cum_carry"] = funding["avg_funding"].rolling(
            self.CARRY_WINDOW, min_periods=self.CARRY_WINDOW // 2
        ).sum()

        # 2. 7-day trend in funding
        funding["avg_trend"] = funding["avg_funding"].rolling(
            self.TREND_WINDOW, min_periods=self.TREND_WINDOW // 2
        ).mean()

        # 3. Z-score for "extreme zone" detection — avoid conflict with FundingReversion
        roll_z = funding["avg_funding"].rolling(self.ZSCORE_WINDOW, min_periods=30)
        funding["z_short"] = (
            (funding["avg_funding"] - roll_z.mean()) / (roll_z.std() + 1e-10)
        )

        # 4. Carry momentum: rate of change of cumulative carry
        funding["carry_momentum"] = funding["cum_carry"].diff(3)

        # Merge BTC price for confirmation
        if not price.empty:
            funding = funding.merge(price, on="date", how="left")
            funding["btc_close"] = funding["btc_close"].ffill()
            funding["btc_ret"] = funding["btc_close"].pct_change(self.PRICE_WINDOW)
            funding["btc_14d_ret"] = funding["btc_close"].pct_change(14)
            funding["btc_sma50"] = funding["btc_close"].rolling(self.SMA_WINDOW).mean()
        else:
            funding["btc_ret"] = 0.0
            funding["btc_14d_ret"] = np.nan
            funding["btc_sma50"] = np.nan

        # Generate signals
        signal = np.zeros(len(funding))
        confidence = np.zeros(len(funding))
        in_position = 0

        for i in range(self.MIN_HISTORY, len(funding)):
            cum = funding["cum_carry"].iloc[i]
            avg_t = funding["avg_trend"].iloc[i]
            z = funding["z_short"].iloc[i]
            carry_mom = funding["carry_momentum"].iloc[i]
            btc_ret = funding["btc_ret"].iloc[i] if "btc_ret" in funding.columns else 0.0
            btc_14d_ret = funding["btc_14d_ret"].iloc[i] if "btc_14d_ret" in funding.columns else np.nan
            btc_close_val = funding["btc_close"].iloc[i] if "btc_close" in funding.columns else np.nan
            btc_sma50 = funding["btc_sma50"].iloc[i] if "btc_sma50" in funding.columns else np.nan
            # True when BTC is in a confirmed uptrend (above 50d SMA) — LONGs only valid here
            btc_uptrend = not pd.isna(btc_close_val) and not pd.isna(btc_sma50) and btc_close_val > btc_sma50

            if pd.isna(cum) or pd.isna(z):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.95 if in_position != 0 else 0.0
                continue

            # Skip new entries if funding is already at extreme (FundingReversion territory)
            in_extreme = abs(z) >= self.EXTREME_Z

            if in_position == 1:
                # Exit LONG: cumulative carry collapsed or entered extreme short territory
                if cum < self.EXIT_CUM_LONG or z < -self.EXTREME_Z:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = 1
                    confidence[i] = confidence[i - 1] * 0.98

            elif in_position == -1:
                # Exit SHORT: cumulative carry flipped positive or entered extreme long territory
                if cum > self.EXIT_CUM_SHORT or z > self.EXTREME_Z:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = -1
                    confidence[i] = confidence[i - 1] * 0.98

            else:
                # Flat: look for new entries
                btc_ok_long = pd.isna(btc_ret) or btc_ret >= 0
                btc_ok_short = pd.isna(btc_ret) or btc_ret <= 0
                trend_long = not pd.isna(avg_t) and avg_t > 0
                trend_short = not pd.isna(avg_t) and avg_t < 0
                # carry_momentum must be building (not decelerating into a reversal)
                mom_ok_long = not pd.isna(carry_mom) and carry_mom > 0
                mom_ok_short = not pd.isna(carry_mom) and carry_mom < 0
                # Macro crash protection: skip LONG if BTC already down >10% over 14 days
                btc_crashing = not pd.isna(btc_14d_ret) and btc_14d_ret < -0.10

                if (cum > self.ENTRY_CUM
                        and trend_long
                        and btc_ok_long
                        and mom_ok_long
                        and btc_uptrend
                        and not in_extreme
                        and not btc_crashing):
                    # Persistent + accelerating positive carry → bull carry regime → LONG
                    signal[i] = 1
                    confidence[i] = min(0.55 + (cum - self.ENTRY_CUM) / self.ENTRY_CUM * 0.10, 0.87)
                    in_position = 1

                elif (cum < self.ENTRY_CUM_SHORT
                        and trend_short
                        and btc_ok_short
                        and mom_ok_short
                        and not in_extreme):
                    # Persistent + accelerating negative carry → bear carry regime → SHORT
                    signal[i] = -1
                    confidence[i] = min(0.55 + (abs(cum) - abs(self.ENTRY_CUM_SHORT)) / abs(self.ENTRY_CUM_SHORT) * 0.10, 0.87)
                    in_position = -1

        funding["signal"] = signal.astype(int)
        funding["confidence"] = confidence

        cols = ["date", "signal", "confidence", "cum_carry", "avg_trend", "z_short"]
        available = [c for c in cols if c in funding.columns]

        # Deduplicate to daily (take last signal per day)
        result = funding[available].copy()
        result["date"] = pd.to_datetime(result["date"]).dt.date
        result = result.sort_values("date").drop_duplicates(subset="date", keep="last")
        return result.reset_index(drop=True)

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient funding data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        cum = last.get("cum_carry", 0.0)
        z = last.get("z_short", 0.0)
        avg_t = last.get("avg_trend", 0.0)

        details = {
            "cum_carry_3d": f"{cum:.6f}",
            "avg_trend_7d": f"{avg_t:.6f}",
            "z_short": f"{z:.2f}",
        }

        if direction == "LONG":
            expl = (
                f"Persistent positive carry: 3d cumfunding={cum:.5f} "
                f"(longs paying carry steadily → bull momentum regime)"
            )
        elif direction == "SHORT":
            expl = (
                f"Persistent negative carry: 3d cumfunding={cum:.5f} "
                f"(shorts paid carry steadily → bear momentum regime)"
            )
        else:
            expl = (
                f"No persistent carry momentum (3d cumfunding={cum:.5f}, z={z:.1f})"
            )

        return StrategySignal(direction, conf, expl, details)
