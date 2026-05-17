"""Strategy: VWAP ATR Channel Breakout (4h).

Thesis: Combining 8-day rolling VWAP with ATR creates dynamic support/resistance
channels: VWAP ± 1.5*ATR defines fair value boundaries. When price breaks above
the upper channel (VWAP + 1.5*ATR), it signals both volume-weighted momentum AND
volatility expansion — a higher-quality breakout signal than BB alone (which uses
std deviation and ignores volume-weighting). The VWAP provides volume-weighted
fair value; the ATR channel width adapts to current volatility conditions.

Entry confirmation: price > channel upper AND RSI(9) in momentum zone (40-70)
AND 4h EMA5>EMA13 (local trend aligned) AND daily golden cross (macro bull).

Exit: 4h EMA5/EMA13 reversal OR price falls below daily EMA50 — same proven
exit logic as EMATrendRegime.

Uniqueness: VWAP is used by none of the 13 existing strategies. ATR-bands (not
ATR as stop-loss but as dynamic channel) is also distinct. The volume-weighted
channel approach differs from both Bollinger Bands (std-based) and Supertrend
(ATR trailing support line).
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


class VWAPRSIReversion(BaseStrategy):

    name = "VWAP RSI Reversion"
    description = "4h VWAP(48-bar) + ATR channel breakout with RSI(9) momentum and daily golden cross"
    data_files = ["binance_futures_klines_5m.csv"]

    VWAP_PERIOD = 48       # 48 × 4h = 8-day rolling VWAP
    ATR_PERIOD = 14        # ATR period for channel width
    ATR_MULT = 1.5         # VWAP ± 1.5*ATR for channel bands
    RSI_PERIOD = 9         # Fast RSI(9) on 4h
    RSI_LONG_MIN = 40      # RSI must be above this for LONG
    RSI_LONG_MAX = 72      # RSI must be below this for LONG
    RSI_SHORT_MIN = 28     # RSI must be above this for SHORT
    RSI_SHORT_MAX = 60     # RSI ceiling for SHORT
    EMA_FAST = 5           # 4h EMA5 for local trend and exit
    EMA_SLOW = 13          # 4h EMA13 for local trend and exit
    DAILY_EMA_FAST = 50
    DAILY_EMA_SLOW = 200   # Golden cross
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

        # Rolling VWAP on 4h bars
        pv = c * v
        pv_series = pd.Series(pv)
        v_series = pd.Series(v)
        vwap = (pv_series.rolling(self.VWAP_PERIOD, min_periods=self.VWAP_PERIOD // 2).sum()
                / v_series.rolling(self.VWAP_PERIOD, min_periods=self.VWAP_PERIOD // 2).sum()).values

        # ATR for channel width
        atr = _wilder_atr(h, lo, c, self.ATR_PERIOD)

        # RSI(9) on 4h
        rsi = _rsi(c, self.RSI_PERIOD)

        # 4h EMA5/EMA13 for local trend direction and exit
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
            r = rsi[i]
            vw = vwap[i]
            atr_i = atr[i]

            if pd.isna(r) or pd.isna(vw) or atr_i <= 0:
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.97 if in_position else 0.0
                continue

            close_i = c[i]
            d_ema50 = bars["d_ema50"].iloc[i]
            d_ema200 = bars["d_ema200"].iloc[i]
            d_ret = bars["d_ret20"].iloc[i]
            ef = ema5[i]
            es = ema13[i]

            if pd.isna(d_ema50) or pd.isna(ef):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.97 if in_position else 0.0
                continue

            # VWAP ATR channels
            vwap_upper = vw + self.ATR_MULT * atr_i
            vwap_lower = vw - self.ATR_MULT * atr_i

            above_ema50 = close_i > d_ema50
            below_ema50 = close_i < d_ema50
            golden_cross = (not pd.isna(d_ema200)) and d_ema50 > d_ema200
            bull_mkt = (not pd.isna(d_ret)) and d_ret > self.BULL_THRESH
            ema_bull = ef > es
            ema_bear = ef < es

            rsi_long_ok = self.RSI_LONG_MIN <= r <= self.RSI_LONG_MAX
            rsi_short_ok = self.RSI_SHORT_MIN <= r <= self.RSI_SHORT_MAX

            if in_position == 1:
                # Exit: 4h EMA reverses or drops below daily EMA50
                if not ema_bull or not above_ema50:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = 1
                    confidence[i] = min(confidence[i - 1] * 0.996, 0.87)

            elif in_position == -1:
                # Exit: 4h EMA reverses or rises above daily EMA50
                if not ema_bear or not below_ema50:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = -1
                    confidence[i] = min(confidence[i - 1] * 0.996, 0.87)

            else:
                # LONG: price breaks above VWAP+ATR channel + RSI momentum + golden cross
                if close_i > vwap_upper and rsi_long_ok and ema_bull and above_ema50 and golden_cross:
                    signal[i] = 1
                    dev = (close_i - vwap_upper) / (atr_i + 1e-10)
                    conf = min(0.61 + dev * 0.02, 0.86)
                    confidence[i] = conf
                    in_position = 1

                # SHORT: price breaks below VWAP-ATR channel + RSI momentum + below EMA50
                elif close_i < vwap_lower and rsi_short_ok and ema_bear and below_ema50 and not bull_mkt:
                    signal[i] = -1
                    dev = (vwap_lower - close_i) / (atr_i + 1e-10)
                    conf = min(0.61 + dev * 0.02, 0.86)
                    confidence[i] = conf
                    in_position = -1

        bars["signal"] = signal.astype(int)
        bars["confidence"] = confidence
        bars["vwap"] = vwap
        bars["rsi9_4h"] = rsi

        result = bars[["date", "signal", "confidence", "vwap", "rsi9_4h"]].copy()
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
        vw = float(last.get("vwap", 0))
        r = float(last.get("rsi9_4h", 50))

        details = {
            "vwap_8d": f"{vw:.0f}",
            "rsi9_4h": f"{r:.1f}",
            "channel": f"VWAP±{self.ATR_MULT}×ATR",
        }
        if direction == "LONG":
            expl = (f"4h price broke above VWAP({vw:.0f})+{self.ATR_MULT}×ATR channel, "
                    f"RSI(9)={r:.1f} momentum zone, daily golden cross")
        elif direction == "SHORT":
            expl = (f"4h price broke below VWAP({vw:.0f})-{self.ATR_MULT}×ATR channel, "
                    f"RSI(9)={r:.1f} momentum zone, below daily EMA50")
        else:
            expl = (f"Price inside VWAP±ATR channel or filters not met: "
                    f"VWAP={vw:.0f}, RSI(9)={r:.1f}")
        return StrategySignal(direction, conf, expl, details)
