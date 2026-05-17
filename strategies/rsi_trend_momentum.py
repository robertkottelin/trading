"""Strategy: RSI Trend Momentum (4h).

Thesis: Fast RSI(9) crossing the 50 centerline on 4h bars after having been in
an extreme zone (below 40 for bullish, above 60 for bearish) signals genuine
momentum shift. RSI(9) is more responsive than classic RSI(14) — catches momentum
transitions 1-2 days earlier. Daily EMA50+EMA200 golden cross ensures we're only
fading oversold dips in macro bull trends.

The "from-extreme" requirement with a 6-bar lookback (= 1 day on 4h) ensures we
catch real momentum builds. Combined with daily EMA50 + EMA200 golden cross, this
avoids the classic RSI false signal in persistent downtrends.

Differentiation:
- MomentumComposite: RSI(14) daily as 1-of-6 oscillators for level extremes
- EMATrendRegime: uses RSI(9) as a gate/filter, not the primary trigger
- This: RSI(9) 50-cross after extreme as primary trigger + golden cross macro filter
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_g / (avg_l + 1e-10)
    return (100 - 100 / (1 + rs)).values


class RSITrendMomentum(BaseStrategy):

    name = "RSI Trend Momentum"
    description = "4h RSI(9) 50-centerline cross after extreme with daily EMA50/200 golden cross gate"
    data_files = ["binance_futures_klines_5m.csv"]

    RSI_PERIOD = 9         # Faster RSI for 4h bars
    DAILY_EMA_FAST = 50
    DAILY_EMA_SLOW = 200   # Golden cross filter
    RSI_UPPER_CROSS = 50
    RSI_LOWER_CROSS = 50
    PRIOR_LOW = 40         # Must have been below this before LONG cross
    PRIOR_HIGH = 60        # Must have been above this before SHORT cross
    PRIOR_LOOKBACK = 6     # 4h bars to look back (= 1 day)
    RSI_EXIT_LONG = 70     # Exit LONG when overbought
    RSI_EXIT_SHORT = 30    # Exit SHORT when oversold

    BULL_WINDOW = 20
    BULL_THRESH = 0.20

    MIN_HISTORY = 50

    # ------------------------------------------------------------------ #

    def _load_4h(self, data: dict) -> pd.DataFrame:
        klines = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()
        klines = klines.copy()
        for col in ["close"]:
            if col in klines.columns:
                klines[col] = pd.to_numeric(klines[col], errors="coerce")
        klines["ts_ms"] = pd.to_numeric(klines["open_time_ms"], errors="coerce")
        klines["dt"] = pd.to_datetime(klines["ts_ms"], unit="ms", utc=True)
        klines["period"] = klines["dt"].dt.floor("4h")
        bars = klines.groupby("period").agg(
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
        d["d_ret20"] = d["d_close"].pct_change(self.BULL_WINDOW)
        d["date_next"] = (pd.to_datetime(d["date"]) + pd.Timedelta(days=1)).dt.date
        return d[["date_next", "d_ema50", "d_ema200", "d_ret20"]].rename(
            columns={"date_next": "date"})

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._load_4h(data)
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()

        daily = self._daily_context(data)
        if daily.empty:
            return pd.DataFrame()

        c = bars["close"].values
        n = len(bars)

        rsi = _rsi(c, self.RSI_PERIOD)

        bars = bars.merge(daily, on="date", how="left")
        bars["d_ema50"] = bars["d_ema50"].ffill()
        bars["d_ema200"] = bars["d_ema200"].ffill()
        bars["d_ret20"] = bars["d_ret20"].ffill()

        signal = np.zeros(n, dtype=int)
        confidence = np.zeros(n, dtype=float)
        in_position = 0

        for i in range(self.MIN_HISTORY, n):
            r = rsi[i]
            r_p = rsi[i - 1] if i > 0 else r
            close_i = c[i]

            if pd.isna(r):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.98 if in_position else 0.0
                continue

            d_ema50 = bars["d_ema50"].iloc[i]
            d_ema200 = bars["d_ema200"].iloc[i]
            d_ret = bars["d_ret20"].iloc[i]

            if pd.isna(d_ema50):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.98 if in_position else 0.0
                continue

            bull_mkt = (not pd.isna(d_ret)) and d_ret > self.BULL_THRESH
            above_ema50 = close_i > d_ema50
            below_ema50 = close_i < d_ema50
            golden_cross = (not pd.isna(d_ema200)) and d_ema50 > d_ema200

            cross_up = r >= self.RSI_UPPER_CROSS and r_p < self.RSI_UPPER_CROSS
            cross_dn = r <= self.RSI_LOWER_CROSS and r_p > self.RSI_LOWER_CROSS

            look_start = max(0, i - self.PRIOR_LOOKBACK)
            was_low = any(rsi[j] < self.PRIOR_LOW
                          for j in range(look_start, i) if not pd.isna(rsi[j]))
            was_high = any(rsi[j] > self.PRIOR_HIGH
                           for j in range(look_start, i) if not pd.isna(rsi[j]))

            if in_position == 1:
                if r > self.RSI_EXIT_LONG or not above_ema50:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = 1
                    confidence[i] = min(confidence[i - 1] * 0.997, 0.87)

            else:
                # LONG ONLY: RSI(9) crosses 50 from below + was < 40 + above EMA50 + golden cross
                if cross_up and was_low and above_ema50 and golden_cross:
                    signal[i] = 1
                    min_rsi = min((rsi[j] for j in range(look_start, i)
                                   if not pd.isna(rsi[j])), default=self.PRIOR_LOW)
                    depth = max(0, self.PRIOR_LOW - min_rsi)
                    confidence[i] = min(0.58 + depth * 0.004, 0.86)
                    in_position = 1

        bars["signal"] = signal.astype(int)
        bars["confidence"] = confidence
        bars["rsi9_4h"] = rsi

        result = bars[["date", "signal", "confidence", "rsi9_4h"]].copy()
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
        r = float(last.get("rsi9_4h", 50))

        details = {
            "rsi9_4h": f"{r:.1f}",
            "rsi_zone": ("oversold" if r < self.PRIOR_LOW else
                         ("overbought" if r > self.PRIOR_HIGH else "neutral")),
        }
        if direction == "LONG":
            expl = (f"4h RSI(9)={r:.1f} crossed above 50 from oversold "
                    f"(below {self.PRIOR_LOW}), above daily EMA50+EMA200 golden cross")
        elif direction == "SHORT":
            expl = (f"4h RSI(9)={r:.1f} crossed below 50 from overbought "
                    f"(above {self.PRIOR_HIGH}), below daily EMA50")
        else:
            expl = f"RSI(9)={r:.1f}, no 50-cross from extreme or macro trend not aligned"
        return StrategySignal(direction, conf, expl, details)
