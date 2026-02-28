"""Implied volatility features from Deribit DVOL (7 features).

Source: deribit_dvol.csv (~58% coverage from 2021-03)

Dropped sources (insufficient API history):
- deribit_historical_vol.csv: 16-day rolling window, no historical API
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_ffill, rolling_zscore


def build_implied_vol_features(grid: pd.DataFrame,
                               base_df: pd.DataFrame) -> pd.DataFrame:
    """Build implied vol features.

    Args:
        grid: DataFrame with open_time_ms
        base_df: Full base parquet with existing GARCH/realized vol columns
    """
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    # --- Deribit DVOL (hourly OHLC → ffill to 5m) ---
    dvol = load_csv("deribit_dvol.csv")
    dvol_aligned = align_ffill(dvol, gms, "timestamp_ms",
                               ["dvol_close", "dvol_high", "dvol_low", "dvol_open"],
                               "")

    result["dvol_close"] = dvol_aligned["dvol_close"]
    result["dvol_momentum_12"] = result["dvol_close"].diff(12).astype(np.float32)
    result["dvol_zscore_288"] = rolling_zscore(result["dvol_close"], 288)

    # Intraday range of DVOL
    result["dvol_intraday_range"] = (
        dvol_aligned["dvol_high"] - dvol_aligned["dvol_low"]
    ).astype(np.float32)

    # DVOL acceleration
    result["dvol_acceleration"] = (
        result["dvol_close"].diff().diff().astype(np.float32)
    )

    # DVOL vs existing GARCH from base parquet
    if "garch_vol_fast" in base_df.columns:
        garch = base_df.set_index("open_time_ms")["garch_vol_fast"].reindex(gms.values)
        # garch_vol_fast is already daily vol (per-candle * sqrt(288) in ta_core).
        # DVOL is annualized %. To annualize daily vol: multiply by sqrt(365.25) * 100.
        garch_annual = garch * np.sqrt(365.25) * 100
        result["dvol_vs_garch"] = (
            (result["dvol_close"].values - garch_annual.values).astype(np.float32)
        )
    else:
        result["dvol_vs_garch"] = np.float32(np.nan)

    # DVOL vs existing realized vol from base parquet
    if "realized_vol_288" in base_df.columns:
        rv = base_df.set_index("open_time_ms")["realized_vol_288"].reindex(gms.values)
        # realized_vol_288 is already daily vol (std * sqrt(288) in ta_core).
        # Annualize: multiply by sqrt(365.25) * 100.
        rv_annual = rv * np.sqrt(365.25) * 100
        result["dvol_vs_realized"] = (
            (result["dvol_close"].values - rv_annual.values).astype(np.float32)
        )
    else:
        result["dvol_vs_realized"] = np.float32(np.nan)

    return result
