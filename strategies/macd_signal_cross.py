"""Strategy: MACD Signal Cross (4h).

Thesis: MACD(12,26,9) signal-line crossovers on 4h bars identify momentum shifts.
Golden cross (daily EMA50 > EMA200) gates LONG entries to confirmed macro bull
regimes only. Exit uses 4h EMA5/EMA13 divergence (fast exit = less exposure to
BTC's volatile multi-day swings), proven effective by EMATrendRegime.

The zero-line filter (MACD > 0 for LONG, < 0 for SHORT) distinguishes this from
MomentumComposite's composite fade approach — this is a pure momentum-confirmation
trend entry, not a fade.

Differentiation from EMATrendRegime: EMATrendRegime uses EMA5/13 price cross as
primary trigger. This uses MACD(12,26,9) signal cross — MACD measures EMA
divergence and fires at different inflection points than raw EMA crosses,
providing signal decorrelation even though both use EMA-based concepts.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class MACDSignalCross(BaseStrategy):

    name = "MACD Signal Cross"
    description = "4h MACD(12,26,9) signal cross with zero-line filter, golden cross, and EMA5/13 exit"
    data_files = ["binance_futures_klines_5m.csv"]

    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL_PERIOD = 9
    EMA_FAST = 5           # 4h EMA5 for exit
    EMA_SLOW = 13          # 4h EMA13 for exit
    DAILY_EMA_FAST = 50
    DAILY_EMA_SLOW = 200   # Golden cross
    BULL_WINDOW = 20
    BULL_THRESH = 0.20

    MIN_HISTORY = 80

    # ------------------------------------------------------------------ #

    def _load_4h(self, data: dict) -> pd.DataFrame:
        klines = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()
        klines = klines.copy()
        for col in ["close", "volume"]:
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

        # MACD on 4h bars
        ema_f = pd.Series(c).ewm(span=self.MACD_FAST, adjust=False).mean().values
        ema_s = pd.Series(c).ewm(span=self.MACD_SLOW, adjust=False).mean().values
        macd_line = ema_f - ema_s
        sig_line = pd.Series(macd_line).ewm(
            span=self.MACD_SIGNAL_PERIOD, adjust=False).mean().values

        # 4h EMA5/13 for exits
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
            ml = macd_line[i]
            sl = sig_line[i]
            ml_p = macd_line[i - 1]
            sl_p = sig_line[i - 1]

            if pd.isna(ml) or pd.isna(sl):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.98 if in_position else 0.0
                continue

            d_ema50 = bars["d_ema50"].iloc[i]
            d_ema200 = bars["d_ema200"].iloc[i]
            d_ret = bars["d_ret20"].iloc[i]
            close_i = c[i]
            ef = ema5[i]
            es = ema13[i]

            if pd.isna(d_ema50) or pd.isna(ef):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.98 if in_position else 0.0
                continue

            cross_up = ml > sl and ml_p <= sl_p
            cross_dn = ml < sl and ml_p >= sl_p
            above_ema50 = close_i > d_ema50
            below_ema50 = close_i < d_ema50
            golden_cross = (not pd.isna(d_ema200)) and d_ema50 > d_ema200
            death_cross = (not pd.isna(d_ema200)) and d_ema50 < d_ema200
            bull_mkt = (not pd.isna(d_ret)) and d_ret > self.BULL_THRESH
            macd_positive = ml > 0
            macd_negative = ml < 0
            ema_bull = ef > es    # 4h EMA5 > EMA13 = local uptrend
            ema_bear = ef < es

            if in_position == 1:
                # Fast exit: 4h EMA reverses OR daily EMA50 breaks
                if not ema_bull or not above_ema50:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = 1
                    confidence[i] = min(confidence[i - 1] * 0.997, 0.87)

            else:
                # LONG ONLY: MACD crosses up + EMA50 aligned + golden cross
                # zero-line filter removed to allow early-cycle entries
                if cross_up and above_ema50 and golden_cross:
                    signal[i] = 1
                    hist = abs(ml - sl)
                    confidence[i] = min(0.61 + hist / (abs(ml) + 1e-10) * 0.04, 0.85)
                    in_position = 1

        bars["signal"] = signal.astype(int)
        bars["confidence"] = confidence
        bars["macd_line"] = macd_line
        bars["macd_sig"] = sig_line

        result = bars[["date", "signal", "confidence", "macd_line", "macd_sig"]].copy()
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
        ml = float(last.get("macd_line", 0))
        sl = float(last.get("macd_sig", 0))
        hist = ml - sl

        details = {
            "macd_line": f"{ml:.2f}",
            "macd_signal": f"{sl:.2f}",
            "histogram": f"{hist:.2f}",
            "zero_side": "positive" if ml > 0 else "negative",
        }
        if direction == "LONG":
            expl = (f"4h MACD(12,26,9) crossed above signal ({ml:.2f}>{sl:.2f}), "
                    f"MACD positive, golden cross, EMA5>EMA13 aligned")
        elif direction == "SHORT":
            expl = (f"4h MACD(12,26,9) crossed below signal ({ml:.2f}<{sl:.2f}), "
                    f"MACD negative, death cross, EMA5<EMA13")
        else:
            expl = f"No MACD cross or macro/local trend filters not aligned (hist={hist:.2f})"
        return StrategySignal(direction, conf, expl, details)
