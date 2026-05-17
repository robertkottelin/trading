"""TripleMomentum (daily) — three-horizon time-series momentum consensus.

Long-only when ALL THREE momentum horizons (10d, 30d, 90d) are positive.
This triple-AND filter is highly selective — only fires during well-
established trends. Combined with vol-targeting, delivers high Sharpe.

Inspired by Antonacci's "dual momentum" but with 3 horizons for more
robustness. The short horizon (10d) prevents entering near tops; the
long horizon (90d) prevents entering in fake breakouts.

Exit when ANY horizon goes negative — opposite of triple-AND entry.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class TripleMomentum(BaseStrategy):
    name = "Triple Momentum"
    description = "Long when 10d, 30d AND 90d returns all positive; exit if any flip"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    H1 = 10
    H2 = 30
    H3 = 90
    EXIT_H = 30      # exit horizon (slow exit avoids whipsaws)
    MIN_HISTORY = 110

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
        r1 = c / c.shift(self.H1) - 1
        r2 = c / c.shift(self.H2) - 1
        r3 = c / c.shift(self.H3) - 1
        r_exit = c / c.shift(self.EXIT_H) - 1

        signal = np.zeros(len(bars), dtype=int)
        pos = 0
        for i in range(self.MIN_HISTORY, len(bars)):
            v1, v2, v3, ve = r1.iloc[i], r2.iloc[i], r3.iloc[i], r_exit.iloc[i]
            if any(pd.isna(x) for x in (v1, v2, v3, ve)):
                signal[i] = pos
                continue
            if pos == 0:
                if v1 > 0 and v2 > 0 and v3 > 0:
                    pos = 1
            else:
                # Exit if the slow-exit horizon turns negative
                if ve < 0:
                    pos = 0
            signal[i] = pos

        # Confidence ~ strength of slowest momentum
        conf = np.clip(0.5 + r3.fillna(0) * 1.0, 0.0, 0.9)
        conf = np.where(signal == 1, conf, 0.0)
        out = bars[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = conf
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Insufficient bars", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = "LONG" if sig == 1 else "INACTIVE"
        return StrategySignal(d, float(last["confidence"]),
                              "Triple-horizon momentum (10d/30d/90d)", {})
