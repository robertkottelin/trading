"""Open interest features from dYdX + Bybit (~8 features).

Sources:
- dydx_candles_5m.csv (starting_oi column)
- bybit_open_interest.csv (57K rows, 5m, recent)

Dropped sources (insufficient API history):
- binance_open_interest.csv: 30-day API retention limit
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_5m, align_ffill, rolling_zscore


def build_open_interest_features(grid: pd.DataFrame) -> pd.DataFrame:
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    # --- dYdX OI (from candles, 5m direct, ~27% coverage from 2023-11) ---
    dydx = load_csv("dydx_candles_5m.csv")
    dydx_aligned = align_5m(dydx, gms, "timestamp",
                            ["starting_oi"], "oi_dydx_")
    result["oi_dydx"] = dydx_aligned["oi_dydx_starting_oi"]

    # Pct change at different horizons
    result["oi_dydx_pct_12"] = (
        result["oi_dydx"].pct_change(12).astype(np.float32)
    )
    result["oi_dydx_pct_48"] = (
        result["oi_dydx"].pct_change(48).astype(np.float32)
    )

    # --- Bybit OI (57K rows, 5m, recent data) ---
    try:
        bybit_oi = load_csv("bybit_open_interest.csv")
        bybit_aligned = align_ffill(bybit_oi, gms, "timestamp_ms",
                                    ["open_interest"], "oi_bybit_")
        result["oi_bybit"] = bybit_aligned["oi_bybit_open_interest"]
        result["oi_bybit_pct_12"] = (
            result["oi_bybit"].pct_change(12).astype(np.float32)
        )
        result["oi_bybit_pct_48"] = (
            result["oi_bybit"].pct_change(48).astype(np.float32)
        )
        result["oi_bybit_zscore_288"] = rolling_zscore(result["oi_bybit"], 288)
    except (FileNotFoundError, KeyError):
        pass

    return result
