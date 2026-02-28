"""Sentiment features: Fear & Greed, Google Trends, market data (~12 features).

Sources: sentiment_fear_greed.csv, sentiment_google_trends.csv, sentiment_market.csv
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_daily, rolling_zscore


def build_sentiment_features(grid: pd.DataFrame) -> pd.DataFrame:
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    # --- Fear & Greed Index (daily) ---
    fng = load_csv("sentiment_fear_greed.csv")
    fng_aligned = align_daily(fng, gms, "date",
                              ["fng_value"], "sent_", lag_days=1)

    result["sent_fng_value"] = fng_aligned["sent_fng_value"]

    # 7-day momentum (7 * 288 = 2016 candles)
    result["sent_fng_momentum_7d"] = (
        result["sent_fng_value"].diff(7 * 288).astype(np.float32)
    )

    # Z-score (30-day rolling = 30 * 288 = 8640 candles)
    result["sent_fng_zscore_30d"] = rolling_zscore(
        result["sent_fng_value"], 30 * 288)

    # Regime flags
    result["sent_fng_extreme_fear"] = (
        (result["sent_fng_value"] < 25).astype(np.float32)
    )
    result["sent_fng_extreme_greed"] = (
        (result["sent_fng_value"] > 75).astype(np.float32)
    )

    # Mean reversion signal: distance from 50 (neutral)
    result["sent_fng_mean_rev"] = (
        (50.0 - result["sent_fng_value"]) / 50.0
    ).astype(np.float32)

    # --- Google Trends (daily/weekly) ---
    trends = load_csv("sentiment_google_trends.csv")
    trends_aligned = align_daily(trends, gms, "date",
                                 ["bitcoin"], "sent_", lag_days=1)

    result["sent_gtrends_bitcoin"] = trends_aligned["sent_bitcoin"]
    result["sent_gtrends_momentum_7d"] = (
        result["sent_gtrends_bitcoin"].diff(7 * 288).astype(np.float32)
    )

    # --- Market data (daily, ~366 rows) ---
    try:
        mkt = load_csv("sentiment_market.csv")
        mkt_cols = [c for c in ["btc_dominance", "total_market_cap", "btc_market_cap"]
                    if c in mkt.columns]
        if mkt_cols:
            m = align_daily(mkt, gms, "date", mkt_cols,
                            "sent_", lag_days=1)
            if "sent_btc_dominance" in m.columns:
                result["sent_btc_dominance"] = m["sent_btc_dominance"]
                result["sent_btc_dominance_change_7d"] = (
                    result["sent_btc_dominance"].diff(7 * 288).astype(np.float32)
                )
            if "sent_total_market_cap" in m.columns:
                result["sent_total_mcap"] = m["sent_total_market_cap"]
                result["sent_total_mcap_change_7d"] = (
                    result["sent_total_mcap"].pct_change(7 * 288).astype(np.float32)
                )
    except (FileNotFoundError, KeyError):
        pass

    return result
