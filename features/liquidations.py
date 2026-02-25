"""Liquidation features from OKX (~4 features).

Source: okx_liquidations.csv (4.6K rows, tick-level events)

Aggregates liquidation events into 5-minute buckets.
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_events, rolling_zscore


def build_liquidation_features(grid: pd.DataFrame) -> pd.DataFrame:
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    try:
        liq = load_csv("okx_liquidations.csv")

        # Separate long and short liquidations
        # OKX API: "sell" = long liquidated (forced sell), "buy" = short liquidated (forced buy)
        side_lower = liq["side"].str.lower()
        liq["is_long"] = (side_lower.isin(["long", "sell"])).astype(float)
        liq["is_short"] = (side_lower.isin(["short", "buy"])).astype(float)
        liq["long_vol"] = liq["size"] * liq["is_long"]
        liq["short_vol"] = liq["size"] * liq["is_short"]

        agg = align_events(liq, gms, "timestamp_ms",
                           {"long_vol": "sum", "short_vol": "sum", "size": "count"},
                           "okx_liq_")

        result["okx_liq_long_vol"] = agg["okx_liq_long_vol"]
        result["okx_liq_short_vol"] = agg["okx_liq_short_vol"]

        total = result["okx_liq_long_vol"] + result["okx_liq_short_vol"]
        result["okx_liq_ratio"] = (
            result["okx_liq_long_vol"] / total.replace(0, np.nan)
        ).astype(np.float32)

        result["okx_liq_total_zscore"] = rolling_zscore(total, 288)

    except (FileNotFoundError, KeyError):
        pass

    return result
