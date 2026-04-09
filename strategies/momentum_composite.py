"""Strategy 6: Technical Momentum Composite.

Combines 6 indicator families (RSI, MACD, Stochastic, Bollinger, Fisher, CCI)
into a composite score. Captures short-term momentum swings and mean-reversion
from oversold/overbought extremes on daily bars.

Uncorrelated with the existing Trend Following strategy which uses long-term
EMAs (12h-21d) + ADX. This strategy uses short-term oscillators (14-20 period)
that detect momentum shifts within trends.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class MomentumComposite(BaseStrategy):

    name = "Technical Momentum"
    description = "Multi-indicator momentum composite (RSI, MACD, Stochastic, Bollinger, Fisher, CCI)"
    data_files = ["binance_futures_klines_5m.csv"]

    # Indicator parameters
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    STOCH_K = 14
    STOCH_D = 3
    STOCH_SMOOTH = 3
    BB_PERIOD = 20
    BB_STD = 2
    FISHER_PERIOD = 9
    CCI_PERIOD = 20

    # Signal thresholds
    LONG_THRESHOLD = 1.2
    SHORT_THRESHOLD = -1.2
    EXIT_THRESHOLD = 0.3  # Close position when |score| drops below this

    # ── Indicator computations ──────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - 100 / (1 + rs)

    @staticmethod
    def _macd(close: pd.Series, fast: int, slow: int, signal: int):
        ema_fast = close.ewm(span=fast, min_periods=fast).mean()
        ema_slow = close.ewm(span=slow, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                     k_period: int, d_period: int, smooth: int):
        lowest_low = low.rolling(k_period, min_periods=k_period).min()
        highest_high = high.rolling(k_period, min_periods=k_period).max()
        raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        k = raw_k.rolling(smooth, min_periods=1).mean()  # Smoothed %K
        d = k.rolling(d_period, min_periods=1).mean()     # %D
        return k, d

    @staticmethod
    def _bollinger(close: pd.Series, period: int, std_mult: float):
        sma = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std()
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        pct_b = (close - lower) / (upper - lower + 1e-10)
        width = (upper - lower) / (sma + 1e-10)
        return pct_b, width, sma

    @staticmethod
    def _fisher_transform(high: pd.Series, low: pd.Series, period: int):
        """Ehlers Fisher Transform (standard implementation).

        Step 1: smooth the normalised price value[i] = 0.5*raw[i] + 0.5*value[i-1]
        Step 2: Fisher[i] = 0.5*ln((1+value)/(1-value)) + 0.5*Fisher[i-1]

        The previous code incorrectly used Fisher[i-1] (the log-transformed output,
        which can be large) in place of value[i-1] (the smoothed raw in [-1,1]).
        When Fisher grew beyond ~1 the sigmoid rescaling drove val to the clip
        boundary every bar, locking the indicator at its extreme and preventing
        reversion — producing permanently wrong cross signals.
        """
        mid = (high + low) / 2
        lowest = mid.rolling(period, min_periods=period).min()
        highest = mid.rolling(period, min_periods=period).max()
        raw = 2 * (mid - lowest) / (highest - lowest + 1e-10) - 1
        raw = raw.clip(-0.999, 0.999)

        raw_arr = raw.to_numpy()
        fisher_arr = np.zeros(len(raw_arr))
        value = 0.0  # smoothed normalised price — kept separate from Fisher output
        for i in range(1, len(raw_arr)):
            r = raw_arr[i]
            if np.isnan(r):
                fisher_arr[i] = fisher_arr[i - 1]
                continue
            value = np.clip(0.5 * r + 0.5 * value, -0.999, 0.999)
            fisher_arr[i] = (0.5 * np.log((1 + value) / (1 - value + 1e-10))
                             + 0.5 * fisher_arr[i - 1])

        fisher = pd.Series(fisher_arr, index=raw.index)
        signal = fisher.shift(1)
        return fisher, signal

    @staticmethod
    def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
        tp = (high + low + close) / 3
        sma = tp.rolling(period, min_periods=period).mean()
        mad = tp.rolling(period, min_periods=period).apply(
            lambda x: np.abs(x - x.mean()).mean(), raw=True
        )
        return (tp - sma) / (0.015 * mad + 1e-10)

    # ── Build daily data ────────────────────────────────────────────────

    def _build_daily(self, data: dict) -> pd.DataFrame:
        df = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if df.empty or "close" not in df.columns:
            return pd.DataFrame()

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce")
        df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df["date"] = df["datetime"].dt.date
        df = df.sort_values("ts_ms")

        daily = df.groupby("date").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        ).reset_index()
        daily = daily.sort_values("date").reset_index(drop=True)
        return daily

    # ── Scoring sub-signals ─────────────────────────────────────────────

    def _score_momentum(self, rsi: pd.Series, stoch_k: pd.Series,
                         stoch_d: pd.Series) -> pd.Series:
        """Sub-signal A: Momentum Score from RSI + Stochastic."""
        score = pd.Series(0.0, index=rsi.index)

        # RSI direction (3-day change)
        rsi_dir = rsi.diff(3)
        score += np.where(rsi_dir > 0, 0.5, np.where(rsi_dir < 0, -0.5, 0))

        # RSI level
        score += np.where((rsi >= 60) & (rsi <= 75), 0.5,
                 np.where(rsi < 30, 0.5,  # Oversold → bullish reversal potential
                 np.where(rsi > 75, -0.5,
                 np.where((rsi >= 30) & (rsi < 40), -0.5, 0))))

        # Stochastic crossover
        score += np.where(stoch_k > stoch_d, 0.5,
                 np.where(stoch_k < stoch_d, -0.5, 0))

        # Stochastic extreme crossovers (bonus)
        oversold_cross = (stoch_k > stoch_d) & (stoch_k < 25)
        overbought_cross = (stoch_k < stoch_d) & (stoch_k > 75)
        score += np.where(oversold_cross, 0.5, 0)
        score += np.where(overbought_cross, -0.5, 0)

        return score.clip(-2, 2)

    def _score_acceleration(self, macd_hist: pd.Series,
                             cci: pd.Series) -> pd.Series:
        """Sub-signal B: Trend Acceleration from MACD histogram + CCI."""
        score = pd.Series(0.0, index=macd_hist.index)

        # MACD histogram sign
        score += np.where(macd_hist > 0, 0.5, np.where(macd_hist < 0, -0.5, 0))

        # MACD histogram momentum (3-day change)
        hist_mom = macd_hist.diff(3)
        score += np.where(hist_mom > 0, 0.5, np.where(hist_mom < 0, -0.5, 0))

        # CCI direction (3-day)
        cci_dir = cci.diff(3)
        score += np.where(cci_dir > 0, 0.5, np.where(cci_dir < 0, -0.5, 0))

        # CCI extreme mean reversion
        score += np.where((cci < -200) & (cci_dir > 0), 0.5, 0)
        score += np.where((cci > 200) & (cci_dir < 0), -0.5, 0)

        return score.clip(-2, 2)

    def _score_volatility(self, pct_b: pd.Series, bb_width: pd.Series,
                           close: pd.Series, bb_mid: pd.Series) -> pd.Series:
        """Sub-signal C: Volatility Context from Bollinger Bands.

        Uses mean-reversion interpretation at extremes, consistent with the
        RSI (oversold=bullish) and Fisher Transform (extreme low=bullish)
        sub-signals.  Price above upper band = overbought (bearish); price
        below lower band = oversold (bullish).
        """
        score = pd.Series(0.0, index=pct_b.index)

        # %B position — approaching band extremes signals overextension
        score += np.where(pct_b > 0.7, -0.5, np.where(pct_b < 0.3, 0.5, 0))

        # %B outside bands — strong mean-reversion signal (matches RSI/Fisher logic)
        score += np.where(pct_b > 1.0, -1.0, np.where(pct_b < 0.0, 1.0, 0))

        # BB width squeeze + price position
        width_pctile = bb_width.rolling(60, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        squeeze = width_pctile < 0.20
        above_mid = close > bb_mid
        score += np.where(squeeze & above_mid, 0.5,
                 np.where(squeeze & ~above_mid, -0.5, 0))

        return score.clip(-2, 2)

    def _score_extreme(self, fisher: pd.Series,
                        fisher_signal: pd.Series) -> pd.Series:
        """Sub-signal D: Extreme/Reversal from Fisher Transform."""
        score = pd.Series(0.0, index=fisher.index)

        # Fisher crosses signal
        cross_up = (fisher > fisher_signal) & (fisher.shift(1) <= fisher_signal.shift(1))
        cross_down = (fisher < fisher_signal) & (fisher.shift(1) >= fisher_signal.shift(1))
        score += np.where(cross_up, 1.0, 0)
        score += np.where(cross_down, -1.0, 0)

        # Fisher extreme reversal
        fisher_dir = fisher.diff(1)
        score += np.where((fisher < -2.0) & (fisher_dir > 0), 0.5, 0)
        score += np.where((fisher > 2.0) & (fisher_dir < 0), -0.5, 0)

        return score.clip(-2, 2)

    # ── Main signal series ──────────────────────────────────────────────

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        daily = self._build_daily(data)
        if daily.empty or len(daily) < 50:
            return pd.DataFrame()

        # Compute all indicators
        rsi = self._rsi(daily["close"], self.RSI_PERIOD)
        _, _, macd_hist = self._macd(daily["close"], self.MACD_FAST,
                                      self.MACD_SLOW, self.MACD_SIGNAL)
        stoch_k, stoch_d = self._stochastic(daily["high"], daily["low"],
                                              daily["close"], self.STOCH_K,
                                              self.STOCH_D, self.STOCH_SMOOTH)
        pct_b, bb_width, bb_mid = self._bollinger(daily["close"],
                                                    self.BB_PERIOD, self.BB_STD)
        fisher, fisher_sig = self._fisher_transform(daily["high"], daily["low"],
                                                      self.FISHER_PERIOD)
        cci = self._cci(daily["high"], daily["low"], daily["close"],
                         self.CCI_PERIOD)

        # Score each sub-signal
        s_momentum = self._score_momentum(rsi, stoch_k, stoch_d)
        s_accel = self._score_acceleration(macd_hist, cci)
        s_vol = self._score_volatility(pct_b, bb_width, daily["close"], bb_mid)
        s_extreme = self._score_extreme(fisher, fisher_sig)

        daily["total_score"] = s_momentum + s_accel + s_vol + s_extreme
        daily["rsi"] = rsi
        daily["macd_hist"] = macd_hist
        daily["stoch_k"] = stoch_k
        daily["cci"] = cci
        daily["fisher"] = fisher
        daily["pct_b"] = pct_b

        # Generate signals with position tracking and exit logic
        signal = np.zeros(len(daily))
        confidence = np.zeros(len(daily))
        in_position = 0

        for i in range(1, len(daily)):
            score = daily["total_score"].iloc[i]
            if pd.isna(score):
                signal[i] = in_position
                confidence[i] = 0.0
                continue

            if score > self.LONG_THRESHOLD:
                signal[i] = 1
                confidence[i] = min(0.55 + (score - self.LONG_THRESHOLD) * 0.05, 0.90)
                in_position = 1
            elif score < self.SHORT_THRESHOLD:
                signal[i] = -1
                confidence[i] = min(0.55 + abs(score - self.SHORT_THRESHOLD) * 0.05, 0.90)
                in_position = -1
            elif in_position != 0:
                # Exit when score fades toward neutral
                if abs(score) < self.EXIT_THRESHOLD:
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    # Hold but decay confidence
                    signal[i] = in_position
                    confidence[i] = confidence[i - 1] * 0.95
            else:
                signal[i] = 0
                confidence[i] = 0.0

        daily["signal"] = signal.astype(int)
        daily["confidence"] = confidence

        return daily[["date", "signal", "confidence", "total_score",
                       "rsi", "macd_hist", "stoch_k", "cci",
                       "fisher", "pct_b"]].copy()

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient price data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        details = {
            "score": f"{last['total_score']:+.2f}",
            "rsi": f"{last['rsi']:.1f}",
            "macd_hist": f"{last['macd_hist']:+.0f}",
            "stoch_k": f"{last['stoch_k']:.0f}",
            "cci": f"{last['cci']:+.0f}",
            "fisher": f"{last['fisher']:+.2f}",
            "bb_pct_b": f"{last['pct_b']:.2f}",
        }

        if direction == "LONG":
            expl = (f"Bullish momentum composite (score={last['total_score']:+.1f}, "
                    f"RSI={last['rsi']:.0f}, Stoch={last['stoch_k']:.0f})")
        elif direction == "SHORT":
            expl = (f"Bearish momentum composite (score={last['total_score']:+.1f}, "
                    f"RSI={last['rsi']:.0f}, Stoch={last['stoch_k']:.0f})")
        else:
            expl = (f"Momentum neutral (score={last['total_score']:+.1f}, "
                    f"RSI={last['rsi']:.0f})")

        return StrategySignal(direction, conf, expl, details)
