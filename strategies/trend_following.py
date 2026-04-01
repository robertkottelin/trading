"""Strategy 7 (Backup): Multi-Timeframe Trend Following.

Thesis: When short, medium, and long-term trends align, the move has momentum.
When they diverge, expect reversal. ADX confirms trend strength.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class TrendFollowing(BaseStrategy):

    name = "Trend Following"
    description = "Multi-timeframe EMA alignment with ADX confirmation"
    data_files = [
        "binance_futures_klines_5m.csv",
    ]

    # Tunable parameters (in 5-minute candles)
    SHORT_FAST = 144    # 12h EMA
    SHORT_SLOW = 288    # 24h EMA
    MED_FAST = 864      # 3d EMA
    MED_SLOW = 2016     # 7d EMA
    LONG_FAST = 2016    # 7d EMA
    LONG_SLOW = 6048    # 21d EMA
    ADX_PERIOD = 288    # 24h ADX (on 5m bars)
    ADX_TREND = 25      # Above this = trending
    ADX_RANGE = 20      # Below this = ranging

    def _build_daily_data(self, data: dict) -> pd.DataFrame:
        """Compute EMAs and ADX on 5m data, then resample to daily."""
        df = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if df.empty or "close" not in df.columns:
            return pd.DataFrame()

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce")
        df = df.sort_values("ts_ms").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df["date"] = df["datetime"].dt.date

        # EMAs
        df["ema_short_fast"] = df["close"].ewm(span=self.SHORT_FAST, min_periods=50).mean()
        df["ema_short_slow"] = df["close"].ewm(span=self.SHORT_SLOW, min_periods=100).mean()
        df["ema_med_fast"] = df["close"].ewm(span=self.MED_FAST, min_periods=200).mean()
        df["ema_med_slow"] = df["close"].ewm(span=self.MED_SLOW, min_periods=500).mean()
        df["ema_long_fast"] = df["close"].ewm(span=self.LONG_FAST, min_periods=500).mean()
        df["ema_long_slow"] = df["close"].ewm(span=self.LONG_SLOW, min_periods=1500).mean()

        # Trend signals
        df["short_trend"] = np.sign(df["ema_short_fast"] - df["ema_short_slow"])
        df["med_trend"] = np.sign(df["ema_med_fast"] - df["ema_med_slow"])
        df["long_trend"] = np.sign(df["ema_long_fast"] - df["ema_long_slow"])

        # ADX calculation (simplified Wilder's ADX)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1))
            )
        )
        df["dm_plus"] = np.where(
            (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
            np.maximum(df["high"] - df["high"].shift(1), 0), 0
        )
        df["dm_minus"] = np.where(
            (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
            np.maximum(df["low"].shift(1) - df["low"], 0), 0
        )

        period = self.ADX_PERIOD
        df["atr"] = df["tr"].ewm(span=period, min_periods=period // 2).mean()
        df["di_plus"] = 100 * (df["dm_plus"].ewm(span=period, min_periods=period // 2).mean() /
                                (df["atr"] + 1e-10))
        df["di_minus"] = 100 * (df["dm_minus"].ewm(span=period, min_periods=period // 2).mean() /
                                 (df["atr"] + 1e-10))
        df["dx"] = 100 * abs(df["di_plus"] - df["di_minus"]) / (df["di_plus"] + df["di_minus"] + 1e-10)
        df["adx"] = df["dx"].ewm(span=period, min_periods=period // 2).mean()

        # Resample to daily (end of day values)
        daily = df.groupby("date").agg(
            close=("close", "last"),
            short_trend=("short_trend", "last"),
            med_trend=("med_trend", "last"),
            long_trend=("long_trend", "last"),
            adx=("adx", "last"),
        ).reset_index()

        return daily

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        df = self._build_daily_data(data)
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        signal = np.zeros(len(df))
        confidence = np.zeros(len(df))

        for i in range(len(df)):
            st = df["short_trend"].iloc[i]
            mt = df["med_trend"].iloc[i]
            lt = df["long_trend"].iloc[i]
            adx = df["adx"].iloc[i]

            if pd.isna(st) or pd.isna(mt) or pd.isna(lt) or pd.isna(adx):
                continue

            all_bull = (st > 0) and (mt > 0) and (lt > 0)
            all_bear = (st < 0) and (mt < 0) and (lt < 0)

            if all_bull and adx > self.ADX_TREND:
                signal[i] = 1
                confidence[i] = min(0.55 + (adx - self.ADX_TREND) * 0.005, 0.85)
            elif all_bear and adx > self.ADX_TREND:
                signal[i] = -1
                confidence[i] = min(0.55 + (adx - self.ADX_TREND) * 0.005, 0.85)
            # Ranging market or divergent trends → INACTIVE

        df["signal"] = signal.astype(int)
        df["confidence"] = confidence
        df["alignment"] = df["short_trend"] + df["med_trend"] + df["long_trend"]

        return df[["date", "signal", "confidence", "adx", "alignment", "close"]].copy()

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient price data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        details = {
            "adx": f"{last['adx']:.1f}",
            "alignment": f"{last['alignment']:+.0f}/3",
        }

        if direction == "LONG":
            expl = f"All timeframes bullish, ADX={last['adx']:.0f} (strong trend)"
        elif direction == "SHORT":
            expl = f"All timeframes bearish, ADX={last['adx']:.0f} (strong trend)"
        else:
            expl = f"Trend divergence or weak trend (ADX={last['adx']:.0f}, alignment={last['alignment']:+.0f}/3)"

        return StrategySignal(direction, conf, expl, details)
