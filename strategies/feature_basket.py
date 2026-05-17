"""FeatureBasket (daily) — composite signal from pre-computed parquet features.

Uses the rich feature set in processed_data/btc_training_dataset.parquet to
build a multi-factor score:

  • Momentum cluster:  return_288 (1d), return_8064 (4w), rsi_14, macd_hist
  • Vol regime:        atr_ratio_96, garch_vol_fast
  • Microstructure:    taker_buy_ratio, dydx_oi_pct_48
  • Sentiment:         sent_btc_dominance_change_7d, sent_fng (if avail)

Each component is z-scored vs its own history; we LONG when the composite
score is positive and exceeds a threshold. Daily signals; uses no future info
because all features are aligned to the 5-min bar and we read the daily-close
row only.

This is essentially "ML-lite" — handcrafted weights instead of a fitted model.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, StrategySignal

log = logging.getLogger(__name__)

PARQUET_PATH = "processed_data/btc_training_dataset.parquet"

# Module-level cache to avoid reloading the 0.8GB parquet every optimizer trial
_FEATURE_CACHE: dict[str, pd.DataFrame] = {}


def _load_features() -> pd.DataFrame:
    cached = _FEATURE_CACHE.get("daily")
    if cached is not None:
        return cached
    if not os.path.exists(PARQUET_PATH):
        return pd.DataFrame()
    cols_wanted = [
        "open_time_ms", "close",
        "return_288", "momentum_rsi", "trend_macd_diff", "atr_ratio_96",
        "garch_vol_fast", "oi_dydx_pct_48",
        "sent_fng_value", "sent_fng_zscore_30d",
    ]
    df = pd.read_parquet(PARQUET_PATH, columns=cols_wanted)
    # Resample to daily — keep last 5m row per UTC day
    df["bucket"] = (df["open_time_ms"].astype("int64") // 86_400_000) * 86_400_000
    daily = df.groupby("bucket").last().reset_index()
    daily = daily.rename(columns={"bucket": "ts_ms"})
    daily = daily.sort_values("ts_ms").reset_index(drop=True)
    _FEATURE_CACHE["daily"] = daily
    return daily


class FeatureBasket(BaseStrategy):
    name = "Feature Basket"
    description = "Composite z-score across momentum/vol/flow features from parquet"
    data_files: list[str] = []  # uses the parquet directly

    bar_seconds = 86400

    Z_WIN = 60
    LONG_THRESH = 0.6
    EXIT_THRESH = -0.2
    ALLOW_SHORT = False
    SHORT_THRESH = -0.8
    MIN_HISTORY = 80

    # Per-feature weight + zscore direction (+1: high=bull, -1: high=bear)
    FEATURE_WEIGHTS = {
        "return_288": (+0.25, +1),
        "momentum_rsi": (+0.10, +1),
        "trend_macd_diff": (+0.15, +1),
        "atr_ratio_96": (-0.10, +1),     # high vol = caution (subtract)
        "garch_vol_fast": (-0.10, +1),
        "oi_dydx_pct_48": (+0.10, +1),
        "sent_fng_value": (+0.15, +1),   # higher greed = trend confirms
        "sent_fng_zscore_30d": (-0.05, +1),  # contrarian when extreme
    }

    def compute_signal_series(self, data: dict) -> pd.DataFrame:
        feat = _load_features()
        if feat.empty or len(feat) < self.MIN_HISTORY:
            return pd.DataFrame()

        score = pd.Series(0.0, index=feat.index)
        # Shift each feature by 1 so we use yesterday's value (no look-ahead)
        for col, (w, sign) in self.FEATURE_WEIGHTS.items():
            if col not in feat.columns:
                continue
            x = pd.to_numeric(feat[col], errors="coerce").shift(1)
            mu = x.rolling(self.Z_WIN, min_periods=10).mean()
            sd = x.rolling(self.Z_WIN, min_periods=10).std()
            z = (x - mu) / sd.replace(0, np.nan)
            score = score + w * sign * z.fillna(0.0)

        signal = np.zeros(len(feat), dtype=int)
        pos = 0
        for i in range(self.MIN_HISTORY, len(feat)):
            s_i = float(score.iloc[i])
            if pos == 0:
                if s_i > self.LONG_THRESH:
                    pos = 1
                elif self.ALLOW_SHORT and s_i < self.SHORT_THRESH:
                    pos = -1
            elif pos == 1:
                if s_i < self.EXIT_THRESH:
                    pos = 0
            elif pos == -1:
                if s_i > -self.EXIT_THRESH:
                    pos = 0
            signal[i] = pos

        out = feat[["ts_ms"]].copy()
        out["signal"] = signal
        out["confidence"] = np.clip(np.abs(score.fillna(0)) / 1.5, 0.0, 0.9)
        out["confidence"] = np.where(signal != 0, out["confidence"], 0.0)
        return out

    def compute_signal(self, data: dict) -> StrategySignal:
        s = self.compute_signal_series(data)
        if s.empty:
            return StrategySignal("INACTIVE", 0.0, "Parquet feature data unavailable", {})
        last = s.iloc[-1]
        sig = int(last["signal"])
        d = {1: "LONG", -1: "SHORT"}.get(sig, "INACTIVE")
        return StrategySignal(d, float(last["confidence"]),
                              "Composite feature score (parquet)", {})
