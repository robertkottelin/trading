"""Positioning features: CFTC COT (6 features).

Source: cftc_cot_bitcoin.csv (weekly, ~92% coverage from 2018-04)

Dropped sources (insufficient API history):
- binance_global_ls_ratio.csv: 30-day API retention limit
- binance_top_ls_accounts.csv: 30-day API retention limit
- binance_top_ls_positions.csv: 30-day API retention limit
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_daily


def build_positioning_features(grid: pd.DataFrame) -> pd.DataFrame:
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    # --- CFTC COT (weekly) ---
    cot = load_csv("cftc_cot_bitcoin.csv")

    # Leveraged money net positioning
    cot["lev_money_net"] = cot["lev_money_long"] - cot["lev_money_short"]
    # Asset manager net positioning
    cot["asset_mgr_net"] = cot["asset_mgr_long"] - cot["asset_mgr_short"]

    cot_aligned = align_daily(cot, gms, "date",
                              ["lev_money_net", "asset_mgr_net",
                               "open_interest"],
                              "cot_", lag_days=0)

    result["cot_lev_money_net"] = cot_aligned["cot_lev_money_net"]
    result["cot_asset_mgr_net"] = cot_aligned["cot_asset_mgr_net"]
    result["cot_oi"] = cot_aligned["cot_open_interest"]

    # COT weekly changes (data is weekly, ffilled to 5m grid: 7 * 288 = 2016 bars)
    result["cot_lev_money_change"] = (
        result["cot_lev_money_net"].diff(7 * 288).astype(np.float32)
    )
    result["cot_asset_mgr_change"] = (
        result["cot_asset_mgr_net"].diff(7 * 288).astype(np.float32)
    )

    # Net positioning as % of OI
    result["cot_lev_money_pct_oi"] = (
        result["cot_lev_money_net"] /
        result["cot_oi"].replace(0, np.nan)
    ).astype(np.float32)

    return result
