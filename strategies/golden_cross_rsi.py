"""GoldenCrossRSI (daily) — combine golden cross with RSI entry timing.

Pattern from winners: regime filter (golden cross 50/200) + entry timing
(RSI dip). The regime tells us WHEN to look for entries; RSI tells us WHEN
to actually enter.

Logic:
  Regime: ema50 > ema200 (bullish regime)
  Entry: RSI crosses up from below 45  (after a healthy pullback)
  Exit: RSI > 75 OR ema50 < ema200 (regime breaks)

Long-only. Trades are scarce but high-quality (~25-50/yr historically).
Vol-target sizing in overlay caps exposure during high-vol periods.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class GoldenCrossRSI(BaseStrategy):
    name = "Golden Cross RSI"
    description = "Long inside golden-cross regime when RSI(14) crosses up from oversold"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    EMA_FAST = 50
    EMA_SLOW = 200
    RSI_PERIOD = 14
    RSI_ENTRY_BELOW = 45    # RSI must dip below this to arm entry
    RSI_ENTRY_CROSS = 50    # then cross up through this to fire
    RSI_EXIT = 75
    MIN_HISTORY = 220

    def _daily(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(close=("close", "last")).reset_index()
        return agg.rename(columns={"bucket": "ts_ms"}).sort_values("ts_ms").reset_index(drop=True)

    @staticmethod
    def _rsi(c: pd.Series, p: int) -> pd.Series:
        d = c.diff()
        g = d.clip(lower=0).ewm(alpha=1 / p, min_periods=p).mean()
        l = -d.clip(upper=0).ewm(alpha=1 / p, min_periods=p).mean()
        return 100 - 100 / (1 + g / (l + 1e-10))

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()
        c = bars["close"].shift(1)
        ema_f = c.ewm(span=self.EMA_FAST, adjust=False, min_periods=self.EMA_FAST).mean()
        ema_s = c.ewm(span=self.EMA_SLOW, adjust=False, min_periods=self.EMA_SLOW).mean()
        rsi = self._rsi(c, self.RSI_PERIOD)

        signal = np.zeros(len(bars), dtype=int)
        pos = 0
        armed = False  # set True after RSI dips below RSI_ENTRY_BELOW
        for i in range(self.MIN_HISTORY, len(bars)):
            r_i = rsi.iloc[i]
            r_prev = rsi.iloc[i - 1]
            regime = (ema_f.iloc[i] > ema_s.iloc[i])
            if pd.isna(r_i) or pd.isna(r_prev) or pd.isna(ema_f.iloc[i]):
                signal[i] = pos
                continue

            if not regime:
                # Regime broken
                pos = 0
                armed = False
            else:
                if r_i < self.RSI_ENTRY_BELOW:
                    armed = True
                if pos == 0 and armed and r_prev < self.RSI_ENTRY_CROSS \
                        and r_i >= self.RSI_ENTRY_CROSS:
                    pos = 1
                    armed = False  # consume the trigger
                elif pos == 1 and r_i > self.RSI_EXIT:
                    pos = 0
            signal[i] = pos

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = np.where(signal == 1, 0.65, 0.0)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Golden cross + RSI dip", {})
