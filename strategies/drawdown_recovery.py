"""DrawdownRecovery (daily) — buy structural-recovery breakouts.

Pattern: after BTC sells off >15% from a recent peak, the first reclaim of
the 30-day EMA often kicks off a multi-week rally (a "kicker" signal).
Empirically, post-drawdown reclaims have higher win rates than pure
momentum because they buy WEAKNESS that's reversing — better risk/reward.

Logic:
  - Compute 60d high (recent peak)
  - Compute drawdown = (close / peak) - 1
  - Trigger entry when:
      * was in >15% drawdown within the past 30 bars
      * close NOW reclaims 30d EMA (price > 30 EMA) for first time since DD
  - Exit when:
      * close < 30d EMA again (failed recovery), OR
      * 60d return > 30% (achieved meaningful gain — take profit)

Long-only. Trade count low but each trade has strong R/R.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class DrawdownRecovery(BaseStrategy):
    name = "Drawdown Recovery"
    description = "Long when BTC reclaims 30d EMA after a 15%+ drawdown"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    PEAK_WINDOW = 60
    DD_THRESH = -0.15
    DD_LOOKBACK = 30
    EMA_PERIOD = 30
    TAKE_PROFIT_DAYS = 60
    TAKE_PROFIT_RET = 0.30
    MIN_HISTORY = 90

    def _daily(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        df = df_5m.copy()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce").astype("int64")
        df["bucket"] = (df["ts_ms"] // 86_400_000) * 86_400_000
        agg = df.groupby("bucket").agg(close=("close", "last")).reset_index()
        return agg.rename(columns={"bucket": "ts_ms"}).sort_values("ts_ms").reset_index(drop=True)

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        bars = self._daily(data.get("binance_futures_klines_5m.csv", pd.DataFrame()))
        if bars.empty or len(bars) < self.MIN_HISTORY:
            return pd.DataFrame()
        c = bars["close"].shift(1)
        peak = c.rolling(self.PEAK_WINDOW, min_periods=self.PEAK_WINDOW).max()
        dd = c / peak - 1
        ema = c.ewm(span=self.EMA_PERIOD, adjust=False, min_periods=self.EMA_PERIOD).mean()
        # Detect "had been in deep DD in the last LOOKBACK bars"
        recent_dd = dd.rolling(self.DD_LOOKBACK, min_periods=1).min()
        deep_drawdown_recently = recent_dd < self.DD_THRESH

        above_ema = c > ema
        above_ema_prev = c.shift(1) <= ema.shift(1)  # crossing UP

        long_entry = deep_drawdown_recently & above_ema & above_ema_prev

        signal = np.zeros(len(bars), dtype=int)
        pos = 0
        entry_idx = -1
        entry_price = 0.0
        for i in range(self.MIN_HISTORY, len(bars)):
            ci = c.iloc[i]
            ei = ema.iloc[i]
            if pd.isna(ci) or pd.isna(ei):
                signal[i] = pos
                continue
            if pos == 0:
                if bool(long_entry.iloc[i]):
                    pos = 1
                    entry_idx = i
                    entry_price = ci
            else:
                # Exit: failed recovery (back below EMA) or take profit
                if ci < ei:
                    pos = 0
                elif entry_price > 0:
                    bars_held = i - entry_idx
                    ret_since = ci / entry_price - 1
                    if bars_held >= self.TAKE_PROFIT_DAYS and ret_since > self.TAKE_PROFIT_RET:
                        pos = 0
            signal[i] = pos

        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = np.where(signal == 1, 0.7, 0.0)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Post-drawdown EMA reclaim", {})
