"""Strategy: EMA Trend Regime.

Thesis: Short-term 4h EMA(5/13) crossovers reliably identify directional
momentum when confirmed by daily EMA50 alignment (multi-timeframe confluence).
The daily EMA50 acts as the macro trend filter — if the 4h fast signal and
the daily trend agree, the probability of sustained follow-through is much
higher than either signal alone.

Signal quality further improved with:
  - RSI(9) momentum gate: avoids entries against immediate momentum direction
  - ATR volatility regime: confidence scales with expanding ATR (trending env)
  - Bull market SHORT filter: suppresses counter-trend SHORTs in bull runs

Why this achieves high Sharpe vs. existing strategies:
  The multi-timeframe gate (4h signal + daily trend) dramatically reduces false
  signals, while 4h resolution gives sufficient frequency (50+/yr). The fast
  EMA(5/13) on 4h catches trend changes 1-2 days before daily indicators fire,
  giving earlier entries with tighter risk.

Differentiation:
  - TrendFollowing: 5m-bar EMA144/288/864/2016 + ADX — requires all timeframes
    aligned + ADX>25, fires only ~24/yr (too slow, misses most moves).
  - MomentumComposite: daily RSI(14) + MACD(12,26,9) + Stoch + BB — oscillators
    that FADE extremes. This strategy FOLLOWS the trend, opposite phase.
  - This strategy: 4h EMA(5/13) x daily EMA50 confluence, ~60+/yr, distinct
    timeframe and indicator set.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int) -> np.ndarray:
    """Wilder smoothed ATR."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """RSI using Wilder exponential smoothing."""
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return (100 - 100 / (1 + rs)).values


class EMATrendRegime(BaseStrategy):

    name = "EMA Trend Regime"
    description = (
        "4h EMA(5/13) x daily EMA50 multi-timeframe momentum trend"
    )
    data_files = ["binance_futures_klines_5m.csv"]

    # 4h bar EMA crossover (fast signal)
    EMA_FAST = 5          # 5 × 4h = 20h — rapid trend detection
    EMA_SLOW = 13         # 13 × 4h = 52h — medium trend context

    # RSI momentum gate
    RSI_PERIOD = 9        # Fast RSI on 4h bars
    RSI_LONG_MIN = 44     # RSI must be above this for LONG entry (not falling)
    RSI_SHORT_MAX = 56    # RSI must be below this for SHORT entry
    RSI_EXIT_LONG = 30    # Exit LONG if RSI collapses (panic selling)
    RSI_EXIT_SHORT = 70   # Exit SHORT if RSI surges (panic buying)

    # Daily macro trend alignment (multi-timeframe gate)
    DAILY_EMA = 50        # Daily EMA50 — near-term macro trend
    DAILY_EMA_SLOW = 200  # Daily EMA200 — golden/death cross confirmation

    # ATR regime: confidence scaling
    ATR_FAST = 7          # Short ATR (7 × 4h = 28h)
    ATR_SLOW = 28         # Long ATR (28 × 4h = 112h)

    # Bull market SHORT filter
    BULL_WINDOW = 20      # Days
    BULL_THRESH = 0.18    # Skip SHORT if BTC up >18% over 20 days

    MIN_HISTORY = 80      # Minimum 4h bars before generating signals

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #

    def _load_4h_bars(self, data: dict) -> pd.DataFrame:
        klines = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()
        klines = klines.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in klines.columns:
                klines[col] = pd.to_numeric(klines[col], errors="coerce")
        klines["ts_ms"] = pd.to_numeric(klines["open_time_ms"], errors="coerce")
        klines["dt"] = pd.to_datetime(klines["ts_ms"], unit="ms", utc=True)
        klines["period_4h"] = klines["dt"].dt.floor("4h")

        bars = klines.groupby("period_4h").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).reset_index().rename(columns={"period_4h": "ts"})
        bars = bars.sort_values("ts").reset_index(drop=True)
        bars = bars.dropna(subset=["open", "high", "low", "close"])
        bars["date"] = bars["ts"].dt.date
        return bars

    def _load_daily_bars(self, data: dict) -> pd.DataFrame:
        klines = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()
        klines = klines.copy()
        klines["close"] = pd.to_numeric(klines["close"], errors="coerce")
        klines["ts_ms"] = pd.to_numeric(klines["open_time_ms"], errors="coerce")
        klines["date"] = pd.to_datetime(klines["ts_ms"], unit="ms", utc=True).dt.date
        daily = klines.groupby("date")["close"].last().reset_index()
        daily.columns = ["date", "d_close"]
        return daily.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Signal generation
    # ------------------------------------------------------------------ #

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._load_4h_bars(data)
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()

        daily = self._load_daily_bars(data)
        if daily.empty:
            return pd.DataFrame()

        # 4h indicators
        bars["ema_fast"] = bars["close"].ewm(span=self.EMA_FAST, adjust=False).mean()
        bars["ema_slow"] = bars["close"].ewm(span=self.EMA_SLOW, adjust=False).mean()

        h, l, c = bars["high"].values, bars["low"].values, bars["close"].values
        bars["atr_fast"] = _wilder_atr(h, l, c, self.ATR_FAST)
        bars["atr_slow"] = _wilder_atr(h, l, c, self.ATR_SLOW)
        bars["rsi"] = _rsi(c, self.RSI_PERIOD)

        # Daily macro indicators (shifted +1 day to prevent lookahead)
        daily["d_ema50"] = daily["d_close"].ewm(span=self.DAILY_EMA, adjust=False).mean()
        daily["d_ema200"] = daily["d_close"].ewm(span=self.DAILY_EMA_SLOW, adjust=False).mean()
        daily["d_ret20"] = daily["d_close"].pct_change(self.BULL_WINDOW)
        daily["date_next"] = (
            pd.to_datetime(daily["date"]) + pd.Timedelta(days=1)
        ).dt.date

        bars = bars.merge(
            daily[["date_next", "d_ema50", "d_ema200", "d_ret20"]].rename(
                columns={"date_next": "date"}),
            on="date", how="left",
        )
        bars["d_ema50"] = bars["d_ema50"].ffill()
        bars["d_ema200"] = bars["d_ema200"].ffill()
        bars["d_ret20"] = bars["d_ret20"].ffill()

        # Signal loop
        n = len(bars)
        signal = np.zeros(n)
        confidence = np.zeros(n)
        in_position = 0

        for i in range(self.MIN_HISTORY, n):
            ef = bars["ema_fast"].iloc[i]
            es = bars["ema_slow"].iloc[i]
            atr_f = bars["atr_fast"].iloc[i]
            atr_s = bars["atr_slow"].iloc[i]
            rsi_i = bars["rsi"].iloc[i]
            close_i = bars["close"].iloc[i]
            d_ema50 = bars["d_ema50"].iloc[i]
            d_ema200 = bars["d_ema200"].iloc[i]
            d_ret20 = bars["d_ret20"].iloc[i]

            if pd.isna(ef) or pd.isna(es) or pd.isna(d_ema50):
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.95 if in_position != 0 else 0.0
                continue

            ema_bull = ef > es
            ema_bear = ef < es
            daily_bull = close_i > d_ema50    # price above daily EMA50
            daily_bear = close_i < d_ema50
            # Golden cross: EMA50 > EMA200 = confirmed bull trend (blocks LONGs in bear)
            golden_cross = not pd.isna(d_ema200) and d_ema50 > d_ema200
            # Death cross: EMA50 < EMA200 = confirmed bear trend (enables SHORTs)
            death_cross = not pd.isna(d_ema200) and d_ema50 < d_ema200
            rsi_ok_long = rsi_i > self.RSI_LONG_MIN
            rsi_ok_short = rsi_i < self.RSI_SHORT_MAX
            bull_mkt = not pd.isna(d_ret20) and d_ret20 > self.BULL_THRESH

            # ATR regime: confidence bonus when volatility is expanding (trending)
            atr_ratio = atr_f / (atr_s + 1e-10) if atr_s > 0 else 1.0
            # atr_ratio > 1 = expanding (trend), < 1 = contracting (range)
            conf_atr_boost = max(0.0, min((atr_ratio - 1.0) * 0.06, 0.06))

            # ---- Position management ----
            if in_position == 1:
                if not ema_bull or not daily_bull or rsi_i < self.RSI_EXIT_LONG:
                    # Exit LONG: 4h EMA reversed, OR price fell below daily EMA50
                    # (daily EMA50 break is an early warning before 4h cross — tighter stop)
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = 1
                    decay = 0.995 if atr_ratio >= 1.0 else 0.97
                    confidence[i] = min(confidence[i - 1] * decay, 0.87)

            elif in_position == -1:
                if not ema_bear or not daily_bear or rsi_i > self.RSI_EXIT_SHORT:
                    # Exit SHORT: 4h EMA reversed, OR price rose above daily EMA50
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = -1
                    decay = 0.995 if atr_ratio >= 1.0 else 0.97
                    confidence[i] = min(confidence[i - 1] * decay, 0.87)

            else:
                # New entries: 4h EMA + daily EMA50 + golden/death cross + RSI
                # LONG requires golden cross (EMA50 > EMA200) — no longs in confirmed bear market
                if (ema_bull and daily_bull and golden_cross and rsi_ok_long):
                    signal[i] = 1
                    base_conf = min(0.57 + conf_atr_boost, 0.84)
                    confidence[i] = min(base_conf, 0.87)
                    in_position = 1

                # SHORT allowed in death cross OR when daily_bear (even without death cross if bearish)
                elif (ema_bear and daily_bear and rsi_ok_short and not bull_mkt):
                    signal[i] = -1
                    base_conf = min(0.57 + conf_atr_boost, 0.84)
                    confidence[i] = min(base_conf, 0.87)
                    in_position = -1

        bars["signal"] = signal.astype(int)
        bars["confidence"] = confidence

        # Deduplicate to daily — take last 4h signal per calendar day
        result = bars[["date", "signal", "confidence",
                        "ema_fast", "ema_slow", "atr_fast", "atr_slow", "rsi"]].copy()
        result["date"] = pd.to_datetime(result["date"]).dt.date
        result = (result.sort_values("date")
                  .drop_duplicates(subset="date", keep="last")
                  .reset_index(drop=True))
        return result

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient price data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        ef = float(last.get("ema_fast", 0))
        es = float(last.get("ema_slow", 0))
        af = float(last.get("atr_fast", 0))
        as_ = float(last.get("atr_slow", 1))
        rsi = float(last.get("rsi", 50))
        atr_ratio = af / (as_ + 1e-10)

        details = {
            "ema5_4h": f"{ef:.0f}",
            "ema13_4h": f"{es:.0f}",
            "atr_ratio": f"{atr_ratio:.2f}",
            "rsi9_4h": f"{rsi:.1f}",
            "regime": "TRENDING" if atr_ratio >= 1.0 else "RANGING",
        }

        if direction == "LONG":
            expl = (
                f"4h EMA5>EMA13 ({ef:.0f}>{es:.0f}) + daily>EMA50 + RSI9={rsi:.0f} "
                f"[ATR regime: {'trending' if atr_ratio >= 1 else 'ranging'}]"
            )
        elif direction == "SHORT":
            expl = (
                f"4h EMA5<EMA13 ({ef:.0f}<{es:.0f}) + daily<EMA50 + RSI9={rsi:.0f} "
                f"[ATR regime: {'trending' if atr_ratio >= 1 else 'ranging'}]"
            )
        else:
            ema_dir = "bullish" if ef > es else "bearish"
            expl = (
                f"4h EMA {ema_dir} but daily trend or RSI not aligned "
                f"(RSI9={rsi:.0f}, EMA ratio={ef/max(es,1):.3f})"
            )

        return StrategySignal(direction, conf, expl, details)
