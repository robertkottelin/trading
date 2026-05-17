"""VolTargetBuyHold (daily) — vol-targeted long, exit only during extreme stress.

The simplest possible high-Sharpe strategy on BTC: always long with
volatility-targeted position sizing, except when realized vol is in the
top 5% historically (panic regimes). Captures BTC's natural drift with
minimal drawdown exposure.

Logic:
  - Compute 21d realized vol (annualized)
  - Compute the 5% high quantile of historical vol (~110%+ in BTC)
  - Long when rv < hi_quantile AND close > 200d EMA (regime intact)
  - Flat during extreme-vol regimes

Trade count is HIGH (frequent exits during vol spikes) but the Sharpe
ratio tends to be excellent because we duck the worst drawdowns.
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal


class VolTargetBuyHold(BaseStrategy):
    name = "VolTarget BuyHold"
    description = "Always-long with regime + extreme-vol exit"
    data_files = ["binance_futures_klines_5m.csv"]

    bar_seconds = 86400

    VOL_WINDOW = 21
    VOL_QUANTILE_WINDOW = 365
    VOL_QUANTILE_THRESH = 0.92  # exit when 21d RV in top 8% historically
    EMA_REGIME = 200
    MIN_HISTORY = 220

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
        log_ret = np.log(c / c.shift(1))
        rv = log_ret.rolling(self.VOL_WINDOW, min_periods=10).std() * np.sqrt(365)

        # Rolling historical quantile (uses past data only — no look-ahead)
        rv_thresh = rv.rolling(self.VOL_QUANTILE_WINDOW,
                                min_periods=60).quantile(self.VOL_QUANTILE_THRESH)
        ema = c.ewm(span=self.EMA_REGIME, adjust=False,
                     min_periods=self.EMA_REGIME).mean()
        long_ok = (rv < rv_thresh) & (c > ema)
        signal = np.where(long_ok.fillna(False), 1, 0)

        # Lower confidence when vol is HIGH (closer to threshold)
        vol_buffer = (rv_thresh - rv) / rv_thresh.replace(0, np.nan)
        conf = np.where(long_ok.fillna(False),
                        np.clip(0.5 + vol_buffer.fillna(0) * 0.8, 0.5, 0.9), 0.0)
        out = bars[["ts_ms"]].copy()
        out["signal"] = signal.astype(int)
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
                              "Vol-target buyhold (regime + extreme-vol filter)", {})
