"""Strategy: Supertrend + OBV + Taker Flow.

Thesis: The Supertrend indicator (ATR-based dynamic support/resistance line)
identifies the macro directional regime on daily bars and flips cleanly when
a trend changes. Unlike oscillators (RSI, MACD) it does not fade moves — it
rides them until the ATR-adjusted band is broken. Combined with:

  1. On-Balance Volume (OBV): cumulative volume confirming price direction.
     Rising OBV during a price uptrend = genuine accumulation. OBV diverging
     from price = potential reversal warning.

  2. Taker Buy Ratio (TBR): fraction of volume that hits the ask (aggressive
     buyers) vs the bid (aggressive sellers). TBR > 0.5 = buyers in control;
     TBR < 0.5 = sellers in control. This is a microstructure signal derived
     directly from exchange data, unavailable to most indicators.

Differentiation:
  - TrendFollowing: 5-minute-bar EMA stacks — requires all 3 timeframes aligned
    + ADX>25 (fires only ~24/yr, too slow).
  - MomentumComposite: RSI, MACD, Stochastic, BB, Fisher — all oscillators that
    FADE moves and fire at price extremes. This strategy does the opposite: it
    FOLLOWS the trend once confirmed by volume and microstructure.
  - CommodityRisk: macro cross-asset (copper/gold/oil) — completely different data.

Signal logic:
  LONG:  Supertrend BULLISH + OBV above 20d EMA + TBR(20d avg) > 0.50
  SHORT: Supertrend BEARISH + OBV below 20d EMA + TBR(20d avg) < 0.50
  Hold:  Supertrend stays in same direction (OBV/TBR used for confidence)
  Exit:  Supertrend flips direction
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class SupertrendOBV(BaseStrategy):

    name = "Supertrend OBV"
    description = (
        "ATR-based Supertrend trend-following with OBV and taker-buy-ratio confirmation"
    )
    data_files = ["binance_futures_klines_5m.csv"]

    # Supertrend parameters
    ST_PERIOD = 7       # ATR lookback — faster than classic 10 reduces lag on volatile BTC
    ST_MULT = 2.0       # ATR multiplier — tighter band, quicker flips on trend reversals

    # OBV trend filter
    OBV_EMA = 20        # EMA period for OBV trend direction

    # Taker buy pressure
    TBR_WINDOW = 20     # Rolling window for taker buy ratio average

    # Bull market SHORT filter — skip new SHORTs during strong uptrends
    BULL_RETURN_WINDOW = 60    # BTC return lookback (days)
    BULL_RETURN_THRESH = 0.20  # If BTC up >20% over 60d, skip new SHORT entries

    # Misc
    MIN_HISTORY = 80    # Minimum daily bars before generating signals

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #

    def _load_daily_ohlcv(self, data: dict) -> pd.DataFrame:
        """Aggregate 5-minute klines to daily OHLCV with taker buy volume."""
        klines = data.get("binance_futures_klines_5m.csv", pd.DataFrame())
        if klines.empty:
            return pd.DataFrame()

        klines = klines.copy()
        for col in ["open", "high", "low", "close", "volume", "taker_buy_volume"]:
            if col in klines.columns:
                klines[col] = pd.to_numeric(klines[col], errors="coerce")

        klines["ts_ms"] = pd.to_numeric(klines["open_time_ms"], errors="coerce")
        klines["date"] = pd.to_datetime(klines["ts_ms"], unit="ms", utc=True).dt.date

        agg = klines.groupby("date").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            taker_buy_vol=("taker_buy_volume", "sum"),
        ).reset_index()

        agg = agg.sort_values("date").reset_index(drop=True)
        agg = agg.dropna(subset=["open", "high", "low", "close", "volume"])
        return agg

    # ------------------------------------------------------------------ #
    #  Indicator calculation
    # ------------------------------------------------------------------ #

    def _compute_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Supertrend with Wilder ATR. Returns df with new columns."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        n = len(df)

        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # Wilder ATR: seed with simple average, then smooth
        atr = np.zeros(n)
        if n >= self.ST_PERIOD:
            atr[self.ST_PERIOD - 1] = tr[: self.ST_PERIOD].mean()
            for i in range(self.ST_PERIOD, n):
                atr[i] = (atr[i - 1] * (self.ST_PERIOD - 1) + tr[i]) / self.ST_PERIOD

        hl2 = (high + low) / 2.0

        # Raw bands
        raw_upper = hl2 + self.ST_MULT * atr
        raw_lower = hl2 - self.ST_MULT * atr

        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        st_line = np.full(n, np.nan)
        direction = np.zeros(n, dtype=int)  # +1 = bullish, -1 = bearish

        # Sequential adjustment — bands only tighten, never widen against trend
        upper[0] = raw_upper[0]
        lower[0] = raw_lower[0]

        for i in range(1, n):
            if atr[i] == 0:
                upper[i] = upper[i - 1]
                lower[i] = lower[i - 1]
                direction[i] = direction[i - 1]
                st_line[i] = st_line[i - 1]
                continue

            # Upper band: only tighten (can only decrease), reset if price closes above it
            upper[i] = (
                raw_upper[i]
                if raw_upper[i] < upper[i - 1] or close[i - 1] > upper[i - 1]
                else upper[i - 1]
            )
            # Lower band: only tighten (can only increase), reset if price closes below it
            lower[i] = (
                raw_lower[i]
                if raw_lower[i] > lower[i - 1] or close[i - 1] < lower[i - 1]
                else lower[i - 1]
            )

            # Determine trend direction
            if close[i] > upper[i]:
                direction[i] = 1
            elif close[i] < lower[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]  # carry forward

            st_line[i] = lower[i] if direction[i] == 1 else upper[i]

        df = df.copy()
        df["st_direction"] = direction
        df["st_line"] = st_line
        df["atr"] = atr
        return df

    def _compute_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute OBV and its EMA trend."""
        close = df["close"].values
        volume = df["volume"].values
        n = len(df)

        obv = np.zeros(n)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        df = df.copy()
        df["obv"] = obv
        df["obv_ema"] = df["obv"].ewm(span=self.OBV_EMA, adjust=False).mean()
        return df

    def _compute_tbr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute taker buy ratio (TBR) and its rolling average."""
        df = df.copy()
        volume = df["volume"].replace(0, np.nan)
        tbv = df["taker_buy_vol"]
        df["tbr"] = (tbv / volume).clip(0, 1)
        df["tbr_avg"] = df["tbr"].rolling(self.TBR_WINDOW, min_periods=5).mean()
        return df

    # ------------------------------------------------------------------ #
    #  Signal generation
    # ------------------------------------------------------------------ #

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        daily = self._load_daily_ohlcv(data)
        if daily.empty or len(daily) < self.MIN_HISTORY:
            return pd.DataFrame()

        daily = self._compute_supertrend(daily)
        daily = self._compute_obv(daily)
        daily = self._compute_tbr(daily)

        # Bull market filter: 60-day BTC return
        daily["btc_60d_ret"] = daily["close"].pct_change(self.BULL_RETURN_WINDOW)

        signal = np.zeros(len(daily))
        confidence = np.zeros(len(daily))
        in_position = 0

        for i in range(self.MIN_HISTORY, len(daily)):
            st_dir = int(daily["st_direction"].iloc[i])
            prev_dir = int(daily["st_direction"].iloc[i - 1])

            obv_i = daily["obv"].iloc[i]
            obv_ema_i = daily["obv_ema"].iloc[i]
            tbr_avg = daily["tbr_avg"].iloc[i]
            close_i = daily["close"].iloc[i]
            st_line_i = daily["st_line"].iloc[i]
            atr_i = daily["atr"].iloc[i]
            btc_60d_ret = daily["btc_60d_ret"].iloc[i]

            if pd.isna(obv_ema_i) or pd.isna(tbr_avg) or atr_i == 0:
                signal[i] = in_position
                confidence[i] = confidence[i - 1] * 0.94 if in_position != 0 else 0.0
                continue

            # Volume/microstructure conditions
            obv_bullish = obv_i > obv_ema_i
            obv_bearish = obv_i < obv_ema_i
            tbr_bullish = tbr_avg > 0.505     # slight positive bias (buyers marginally dominant)
            tbr_bearish = tbr_avg < 0.495     # slight negative bias (sellers marginally dominant)

            # Bull market filter: avoid shorting when BTC is in a strong 60-day uptrend
            bull_mkt = not pd.isna(btc_60d_ret) and btc_60d_ret > self.BULL_RETURN_THRESH

            # Supertrend distance from price (in ATR units) — used for confidence
            st_dist = abs(close_i - st_line_i) / atr_i if not pd.isna(st_line_i) else 0.0

            # ---- Position management ----
            if in_position == 1:
                if st_dir == -1 or not obv_bullish:
                    # Exit LONG: Supertrend flipped bearish OR OBV distribution started
                    # (OBV divergence is an early warning before Supertrend lags behind)
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    # Both ST and OBV still bullish — hold, TBR modulates confidence
                    signal[i] = 1
                    confidence[i] = confidence[i - 1] * (0.98 if tbr_bullish else 0.93)

            elif in_position == -1:
                if st_dir == 1 or not obv_bearish:
                    # Exit SHORT: Supertrend flipped bullish OR OBV accumulation started
                    signal[i] = 0
                    confidence[i] = 0.0
                    in_position = 0
                else:
                    signal[i] = -1
                    confidence[i] = confidence[i - 1] * (0.98 if tbr_bearish else 0.93)

            else:
                # --- New entries: require Supertrend + OBV + TBR all agree ---
                if st_dir == 1 and obv_bullish and tbr_bullish:
                    # All three confirm bullish: Supertrend + volume accumulation + buyer aggression
                    signal[i] = 1
                    base_conf = min(0.55 + min(st_dist, 4.0) * 0.05, 0.84)
                    confidence[i] = min(base_conf, 0.87)
                    in_position = 1

                elif st_dir == -1 and obv_bearish and tbr_bearish and not bull_mkt:
                    # All three confirm bearish + not in a macro bull run
                    signal[i] = -1
                    base_conf = min(0.55 + min(st_dist, 4.0) * 0.05, 0.84)
                    confidence[i] = min(base_conf, 0.87)
                    in_position = -1

        daily["signal"] = signal.astype(int)
        daily["confidence"] = confidence

        cols = ["date", "signal", "confidence", "st_direction", "st_line", "atr",
                "obv", "obv_ema", "tbr_avg"]
        return daily[[c for c in cols if c in daily.columns]].copy()

    def compute_signal(self, data: dict) -> StrategySignal:
        series = self.compute_signal_series(data)
        if series.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient price data", {})

        last = series.iloc[-1]
        sig = int(last["signal"])
        direction = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        conf = float(last["confidence"])

        st_dir = int(last.get("st_direction", 0))
        st_line = last.get("st_line", 0.0)
        atr = last.get("atr", 0.0)
        obv_i = last.get("obv", 0.0)
        obv_ema_i = last.get("obv_ema", 0.0)
        tbr = last.get("tbr_avg", 0.5)

        details = {
            "st_direction": "bullish" if st_dir == 1 else "bearish",
            "st_line": f"{st_line:.0f}",
            "atr_14d": f"{atr:.0f}",
            "obv_vs_ema": "above" if obv_i > obv_ema_i else "below",
            "tbr_20d": f"{tbr:.3f}",
        }

        if direction == "LONG":
            expl = (
                f"Supertrend bullish (line=${st_line:.0f}), OBV "
                f"{'above' if obv_i > obv_ema_i else 'below'} EMA, "
                f"taker-buy-ratio={tbr:.3f}"
            )
        elif direction == "SHORT":
            expl = (
                f"Supertrend bearish (line=${st_line:.0f}), OBV "
                f"{'below' if obv_i < obv_ema_i else 'above'} EMA, "
                f"taker-buy-ratio={tbr:.3f}"
            )
        else:
            expl = (
                f"No confluence: ST={'bullish' if st_dir == 1 else 'bearish'}, "
                f"OBV {'above' if obv_i > obv_ema_i else 'below'} EMA, TBR={tbr:.3f}"
            )

        return StrategySignal(direction, conf, expl, details)
