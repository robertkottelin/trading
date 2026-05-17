"""Strategy: Stochastic Oscillator EMA Cross (4h).

Thesis: Stochastic(8,3,3) K/D crossovers from oversold zones (<40) on 4h bars
combined with the local 4h uptrend (EMA5 > EMA13) and price above the daily EMA50.
Using Stoch(8) instead of (14) makes K more responsive on 4h bars, firing more
frequently while still filtering out sub-hour noise.

The EMA50 daily filter keeps entries anchored to intermediate-term uptrends. No
golden cross requirement — this allows entries even when EMA200 is still catching
up, which is valid in early bull-market stages. Zone requirement ensures K was
genuinely oversold (< 40) at some point in the last 48h before the crossover.

SHORTs omitted entirely — Stochastic oscillator SHORT signals on BTC have
historically low win rates due to sustained momentum moves.

Differentiation from MomentumComposite:
  MC uses Stochastic(14,3,3) daily as 1-of-6 composite components. This uses
  Stoch(8,3,3) on 4h as the sole primary trigger with EMA confirmation —
  different parameters, different timeframe, different philosophy.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


def _stoch(high: np.ndarray, low: np.ndarray, close: np.ndarray,
           k_period: int, d_period: int, smooth: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    raw_k = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        hh = high[i - k_period + 1: i + 1].max()
        ll = low[i - k_period + 1: i + 1].min()
        rng = hh - ll
        raw_k[i] = (close[i] - ll) / (rng + 1e-10) * 100
    k_line = pd.Series(raw_k).rolling(smooth, min_periods=1).mean().values
    d_line = pd.Series(k_line).rolling(d_period, min_periods=1).mean().values
    return k_line, d_line


class StochasticEMACross(BaseStrategy):

    name = "Stochastic EMA Cross"
    description = "4h Stoch(8,3,3) K/D cross from oversold, LONG-only, daily EMA50 + local EMA5/13"
    data_files = ["binance_futures_klines_5m.csv"]

    STOCH_K = 8             # Faster K period for 4h bars
    STOCH_D = 3
    STOCH_SMOOTH = 3
    EMA_FAST = 5            # 4h EMA5 for local trend
    EMA_SLOW = 13           # 4h EMA13 for local trend
    DAILY_EMA_FAST = 50     # Price must be above this for LONG entries
    DAILY_EMA_SLOW = 200    # Golden cross check (for signal details only)
    OVERSOLD = 40           # K threshold for oversold zone entry
    EXIT_OB = 78            # Exit LONG when K overbought
    ZONE_LOOKBACK = 12      # 4h bars to look back (= 2 days)

    BULL_WINDOW = 20
    BULL_THRESH = 0.20

    MIN_HISTORY = 60

    # ------------------------------------------------------------------ #

    def _load_4h(self, data: dict) -> pd.DataFrame:
        klines = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()
        klines = klines.copy()
        for col in ["open", "high", "low", "close"]:
            if col in klines.columns:
                klines[col] = pd.to_numeric(klines[col], errors="coerce")
        klines["ts_ms"] = pd.to_numeric(klines["open_time_ms"], errors="coerce")
        klines["dt"] = pd.to_datetime(klines["ts_ms"], unit="ms", utc=True)
        klines["period"] = klines["dt"].dt.floor("4h")
        bars = klines.groupby("period").agg(
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).reset_index().rename(columns={"period": "ts"})
        bars = bars.sort_values("ts").reset_index(drop=True)
        bars = bars.dropna(subset=["close"])
        bars["date"] = bars["ts"].dt.date
        return bars

    def _daily_context(self, data: dict) -> pd.DataFrame:
        klines = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()
        klines = klines.copy()
        klines["close"] = pd.to_numeric(klines["close"], errors="coerce")
        klines["ts_ms"] = pd.to_numeric(klines["open_time_ms"], errors="coerce")
        klines["date"] = pd.to_datetime(klines["ts_ms"], unit="ms", utc=True).dt.date
        d = klines.groupby("date")["close"].last().reset_index()
        d.columns = ["date", "d_close"]
        d = d.sort_values("date").reset_index(drop=True)
        d["d_ema50"] = d["d_close"].ewm(span=self.DAILY_EMA_FAST, adjust=False).mean()
        d["d_ema200"] = d["d_close"].ewm(span=self.DAILY_EMA_SLOW, adjust=False).mean()
        d["date_next"] = (pd.to_datetime(d["date"]) + pd.Timedelta(days=1)).dt.date
        return d[["date_next", "d_ema50", "d_ema200"]].rename(columns={"date_next": "date"})

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._load_4h(data)
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()

        daily = self._daily_context(data)
        if daily.empty:
            return pd.DataFrame()

        h = bars["high"].values
        lo = bars["low"].values
        c = bars["close"].values
        n = len(bars)

        k_line, d_line = _stoch(h, lo, c, self.STOCH_K, self.STOCH_D, self.STOCH_SMOOTH)
        ema5 = pd.Series(c).ewm(span=self.EMA_FAST, adjust=False).mean().values
        ema13 = pd.Series(c).ewm(span=self.EMA_SLOW, adjust=False).mean().values

        bars = bars.merge(daily, on="date", how="left")
        bars["d_ema50"] = bars["d_ema50"].ffill()
        bars["d_ema200"] = bars["d_ema200"].ffill()

        signal = np.zeros(n, dtype=int)
        confidence = np.zeros(n, dtype=float)
        in_position = 0

        for i in range(self.MIN_HISTORY, n):
            k = k_line[i]
            d_val = d_line[i]
            k_p = k_line[i - 1]
            d_p = d_line[i - 1]

            if pd.isna(k) or pd.isna(d_val):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.97 if in_position else 0.0
                continue

            close_i = c[i]
            d_ema50 = bars["d_ema50"].iloc[i]
            d_ema200 = bars["d_ema200"].iloc[i]

            if pd.isna(d_ema50):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.97 if in_position else 0.0
                continue

            above_ema50 = close_i > d_ema50
            golden_cross = (not pd.isna(d_ema200)) and d_ema50 > d_ema200
            ema_bull = ema5[i] > ema13[i]

            k_cross_up = k > d_val and k_p <= d_p

            # Zone requirement: K must have been below OVERSOLD at some point in ZONE_LOOKBACK
            look_start = max(self.MIN_HISTORY, i - self.ZONE_LOOKBACK)
            was_os = any(not pd.isna(k_line[j]) and k_line[j] < self.OVERSOLD
                         for j in range(look_start, i))

            if in_position == 1:
                # Exit: 4h EMA reverses OR K overbought OR price below daily EMA50
                if not ema_bull or k > self.EXIT_OB or not above_ema50:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = 1
                    confidence[i] = min(confidence[i - 1] * 0.996, 0.85)

            else:
                # LONG ONLY: K/D cross from oversold + local 4h uptrend + above EMA50
                # No golden cross required — allows early-cycle and intermediate entries
                if k_cross_up and was_os and above_ema50 and ema_bull:
                    signal[i] = 1
                    os_depth = max(0, self.OVERSOLD - min(
                        (k_line[j] for j in range(look_start, i)
                         if not pd.isna(k_line[j])), default=self.OVERSOLD))
                    confidence[i] = min(0.58 + os_depth * 0.003, 0.85)
                    in_position = 1

        bars["signal"] = signal.astype(int)
        bars["confidence"] = confidence
        bars["stoch_k"] = k_line
        bars["stoch_d"] = d_line

        result = bars[["date", "signal", "confidence", "stoch_k", "stoch_d"]].copy()
        result["date"] = pd.to_datetime(result["date"]).dt.date
        result = (result.sort_values("date")
                  .drop_duplicates("date", keep="last")
                  .reset_index(drop=True))
        return result

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])
        k = float(last.get("stoch_k", 50))
        d = float(last.get("stoch_d", 50))

        details = {
            "stoch_k": f"{k:.1f}",
            "stoch_d": f"{d:.1f}",
            "zone": "OVERSOLD" if k < self.OVERSOLD else ("OVERBOUGHT" if k > self.EXIT_OB else "NEUTRAL"),
        }
        if direction == "LONG":
            expl = (f"4h Stoch(8,3,3) K({k:.1f}) crossed above D({d:.1f}) from oversold "
                    f"(<{self.OVERSOLD}), above daily EMA50, local 4h EMA5>EMA13")
        else:
            expl = (f"No oversold bounce entry: Stoch K={k:.1f} D={d:.1f}, "
                    f"waiting for oversold dip + EMA5>EMA13 + above daily EMA50")
        return StrategySignal(direction, conf, expl, details)
