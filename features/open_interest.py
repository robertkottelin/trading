"""Open interest features from dYdX (~3 features).

Sources:
- dydx_candles_5m.csv (starting_oi column)

Dropped sources (insufficient API history):
- binance_open_interest.csv: 30-day API retention limit
- bybit_open_interest.csv: ~200-day API retention limit — would create
  a degenerate post-2025-10 feature for ML training
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_5m


def build_open_interest_features(grid: pd.DataFrame) -> pd.DataFrame:
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    # --- dYdX OI (from candles, 5m direct, available from 2023-11) ---
    dydx = load_csv("dydx_candles_5m.csv")
    dydx_aligned = align_5m(dydx, gms, "timestamp",
                            ["starting_oi"], "oi_dydx_")
    result["oi_dydx"] = dydx_aligned["oi_dydx_starting_oi"]

    # Pct change at different horizons
    result["oi_dydx_pct_12"] = (
        result["oi_dydx"].pct_change(12, fill_method=None).astype(np.float32)
    )
    result["oi_dydx_pct_48"] = (
        result["oi_dydx"].pct_change(48, fill_method=None).astype(np.float32)
    )

    return result
