"""Strategy: Bollinger Band Breakout with OBV Momentum (4h).

Thesis: When 4h price breaks out of a Bollinger Band(15,1.8) AND OBV confirms
direction AND volatility is expanding (ATR in trending regime), the breakout is
high-quality. Tighter bands (1.8σ) fire more frequently. ATR regime filter
(fast ATR > slow ATR = expanding volatility = trending) avoids false breakouts
during choppy range-bound markets. Daily EMA50+EMA200 golden cross gates LONGs
to confirmed macro bull regimes only.

Exit uses 4h EMA5/EMA13 cross (borrowed from EMATrendRegime) rather than
BB midline — this holds through consolidations and exits on trend reversal.

Differentiation:
- MomentumComposite: fades BB extremes (opposite logic)
- SupertrendOBV: OBV confirmation but Supertrend (not BB) with TBR data
- EMATrendRegime: EMA5/13 cross as primary trigger; BB breakout here is primary
- This: BB(15,1.8) breakout + OBV + ATR trending filter + EMA5/13 exit
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


def _wilder_atr(high, low, close, period):
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class BBBreakoutOBV(BaseStrategy):

    name = "BB Breakout OBV"
    description = "4h BB(15,1.8) breakout + OBV + ATR trending regime + daily EMA50/200 golden cross"
    data_files = ["binance_futures_klines_5m.csv"]

    BB_PERIOD = 15
    BB_STD = 1.8
    OBV_EMA_PERIOD = 20
    ATR_FAST = 7          # 7 × 4h = 28h fast ATR
    ATR_SLOW = 28         # 28 × 4h = 112h slow ATR
    EMA_FAST = 5          # 4h EMA5 for exit signal
    EMA_SLOW = 13         # 4h EMA13 for exit signal
    DAILY_EMA_FAST = 50
    DAILY_EMA_SLOW = 200  # Golden cross for LONG
    BULL_WINDOW = 20
    BULL_THRESH = 0.20

    MIN_HISTORY = 60

    # ------------------------------------------------------------------ #

    def _load_4h(self, data: dict) -> pd.DataFrame:
        klines = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()
        klines = klines.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in klines.columns:
                klines[col] = pd.to_numeric(klines[col], errors="coerce")
        klines["ts_ms"] = pd.to_numeric(klines["open_time_ms"], errors="coerce")
        klines["dt"] = pd.to_datetime(klines["ts_ms"], unit="ms", utc=True)
        klines["period"] = klines["dt"].dt.floor("4h")
        bars = klines.groupby("period").agg(
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
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

        h = bars["high"].values
        lo = bars["low"].values
        c = bars["close"].values
        v = bars["volume"].values
        n = len(bars)

        # Bollinger Bands (4h)
        bb_mid = pd.Series(c).rolling(self.BB_PERIOD).mean().values
        bb_std = pd.Series(c).rolling(self.BB_PERIOD).std().values
        bb_upper = bb_mid + self.BB_STD * bb_std
        bb_lower = bb_mid - self.BB_STD * bb_std

        # OBV on 4h
        obv = np.zeros(n)
        for i in range(1, n):
            if c[i] > c[i - 1]:
                obv[i] = obv[i - 1] + v[i]
            elif c[i] < c[i - 1]:
                obv[i] = obv[i - 1] - v[i]
            else:
                obv[i] = obv[i - 1]
        obv_ema = pd.Series(obv).ewm(span=self.OBV_EMA_PERIOD, adjust=False).mean().values

        # ATR regime: trending when fast ATR > slow ATR
        atr_fast = _wilder_atr(h, lo, c, self.ATR_FAST)
        atr_slow = _wilder_atr(h, lo, c, self.ATR_SLOW)

        # 4h EMA5/EMA13 for trend-based exits
        ema5 = pd.Series(c).ewm(span=self.EMA_FAST, adjust=False).mean().values
        ema13 = pd.Series(c).ewm(span=self.EMA_SLOW, adjust=False).mean().values

        bars = bars.merge(daily, on="date", how="left")
        bars["d_ema50"] = bars["d_ema50"].ffill()
        bars["d_ema200"] = bars["d_ema200"].ffill()
        bars["d_ret20"] = bars["d_ret20"].ffill()

        signal = np.zeros(n, dtype=int)
        confidence = np.zeros(n, dtype=float)
        in_position = 0

        for i in range(self.MIN_HISTORY, n):
            if pd.isna(bb_upper[i]) or pd.isna(obv_ema[i]):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.97 if in_position else 0.0
                continue

            close_i = c[i]
            bb_u = bb_upper[i]
            bb_l = bb_lower[i]
            obv_i = obv[i]
            obv_ema_i = obv_ema[i]
            d_ema50 = bars["d_ema50"].iloc[i]
            d_ema200 = bars["d_ema200"].iloc[i]
            d_ret = bars["d_ret20"].iloc[i]
            ef = ema5[i]
            es = ema13[i]
            atr_f = atr_fast[i]
            atr_s = atr_slow[i]

            if pd.isna(d_ema50) or pd.isna(ef):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.97 if in_position else 0.0
                continue

            obv_bull = obv_i > obv_ema_i
            obv_bear = obv_i < obv_ema_i
            above_ema50 = close_i > d_ema50
            below_ema50 = close_i < d_ema50
            golden_cross = (not pd.isna(d_ema200)) and d_ema50 > d_ema200
            bull_mkt = (not pd.isna(d_ret)) and d_ret > self.BULL_THRESH
            ema_bull = ef > es       # 4h EMA5 > EMA13 = local uptrend
            ema_bear = ef < es
            # ATR trending regime: fast ATR > slow ATR = expanding volatility
            trending = atr_f > atr_s if atr_s > 0 else True

            if in_position == 1:
                # Exit: 4h EMA reverses OR price falls below daily EMA50
                if not ema_bull or not above_ema50:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = 1
                    decay = 0.996 if trending else 0.97
                    confidence[i] = min(confidence[i - 1] * decay, 0.87)

            elif in_position == -1:
                # Exit: 4h EMA reverses OR price rises above daily EMA50
                if not ema_bear or not below_ema50:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = -1
                    decay = 0.996 if trending else 0.97
                    confidence[i] = min(confidence[i - 1] * decay, 0.87)

            else:
                # LONG: BB breakout + OBV confirms + above EMA50 + golden cross + trending
                if close_i > bb_u and obv_bull and above_ema50 and golden_cross and trending:
                    signal[i] = 1
                    atr_boost = max(0.0, min((atr_f / (atr_s + 1e-10) - 1.0) * 0.05, 0.05))
                    confidence[i] = min(0.63 + atr_boost, 0.87)
                    in_position = 1
                # SHORT: BB breakdown + OBV confirms + below EMA50 + trending
                elif close_i < bb_l and obv_bear and below_ema50 and not bull_mkt and trending:
                    signal[i] = -1
                    atr_boost = max(0.0, min((atr_f / (atr_s + 1e-10) - 1.0) * 0.05, 0.05))
                    confidence[i] = min(0.63 + atr_boost, 0.87)
                    in_position = -1

        bars["signal"] = signal.astype(int)
        bars["confidence"] = confidence
        bars["bb_upper"] = bb_upper
        bars["bb_lower"] = bb_lower
        bars["bb_mid"] = bb_mid

        result = bars[["date", "signal", "confidence",
                        "bb_upper", "bb_lower", "bb_mid"]].copy()
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
        bb_u = float(last.get("bb_upper", 0))
        bb_l = float(last.get("bb_lower", 0))
        bb_m = float(last.get("bb_mid", 0))

        details = {
            "bb_upper": f"{bb_u:.0f}",
            "bb_lower": f"{bb_l:.0f}",
            "bb_mid": f"{bb_m:.0f}",
        }
        if direction == "LONG":
            expl = (f"4h price broke above BB upper ({bb_u:.0f}) with OBV bullish, "
                    f"ATR trending, above EMA50+EMA200 golden cross")
        elif direction == "SHORT":
            expl = (f"4h price broke below BB lower ({bb_l:.0f}) with OBV bearish, "
                    f"ATR trending, below EMA50")
        else:
            expl = (f"No qualifying BB breakout — price inside bands or trend/ATR filters not met "
                    f"({bb_l:.0f}–{bb_u:.0f}, mid={bb_m:.0f})")
        return StrategySignal(direction, conf, expl, details)
