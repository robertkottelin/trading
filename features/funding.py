"""Multi-exchange funding rate features (18 features).

Sources: binance/bybit_funding_rates.csv (8h),
         deribit/dydx/hyperliquid_funding_rates.csv (1h)

Dropped sources (insufficient API history):
- okx_funding_rates.csv: API retains only ~107 days
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_ffill, rolling_zscore


def _load_funding_source(filename: str, ts_col: str, rate_col: str,
                         grid_ms: pd.Series, prefix: str) -> pd.Series:
    """Load a single funding rate source, align to grid, return Series."""
    df = load_csv(filename)
    aligned = align_ffill(df, grid_ms, ts_col, [rate_col], prefix)
    return aligned[f"{prefix}{rate_col}"]


def build_funding_features(grid: pd.DataFrame) -> pd.DataFrame:
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    # Load funding sources (5 exchanges)
    result["funding_binance"] = _load_funding_source(
        "binance_funding_rates.csv", "funding_time_ms", "funding_rate", gms, "")
    result["funding_bybit"] = _load_funding_source(
        "bybit_funding_rates.csv", "funding_time_ms", "funding_rate", gms, "")
    result["funding_dydx"] = _load_funding_source(
        "dydx_funding_rates.csv", "timestamp", "rate", gms, "")
    result["funding_hyperliquid"] = _load_funding_source(
        "hyperliquid_funding_rates.csv", "timestamp_ms", "funding_rate", gms, "")
    result["funding_deribit"] = _load_funding_source(
        "deribit_funding_rates.csv", "timestamp_ms", "interest_8h", gms, "")

    # Cross-exchange statistics
    funding_cols = ["funding_binance", "funding_bybit",
                    "funding_dydx", "funding_hyperliquid", "funding_deribit"]
    funding_matrix = result[funding_cols].values.astype(np.float64)

    result["funding_cross_mean"] = np.nanmean(funding_matrix, axis=1).astype(np.float32)
    result["funding_cross_std"] = np.nanstd(funding_matrix, axis=1).astype(np.float32)
    result["funding_cross_max"] = np.nanmax(funding_matrix, axis=1).astype(np.float32)
    result["funding_cross_min"] = np.nanmin(funding_matrix, axis=1).astype(np.float32)
    result["funding_cross_range"] = (
        result["funding_cross_max"] - result["funding_cross_min"]
    ).astype(np.float32)

    # Momentum
    result["funding_binance_momentum_3"] = (
        result["funding_binance"].diff(3).astype(np.float32)
    )
    result["funding_cross_mean_momentum_3"] = (
        result["funding_cross_mean"].diff(3).astype(np.float32)
    )

    # Z-scores
    result["funding_binance_zscore_30"] = rolling_zscore(
        result["funding_binance"], 30)
    result["funding_cross_mean_zscore_30"] = rolling_zscore(
        result["funding_cross_mean"], 30)

    # Cross-exchange spread
    result["funding_dydx_vs_binance"] = (
        result["funding_dydx"] - result["funding_binance"]
    ).astype(np.float32)

    # Hourly acceleration
    result["funding_hourly_accel"] = (
        result["funding_dydx"].diff().diff().astype(np.float32)
    )

    # Binance cumulative sum (rolling 24h)
    result["funding_binance_cumsum_24h"] = (
        result["funding_binance"]
        .rolling(288, min_periods=1).sum().astype(np.float32)
    )

    # Positive/negative ratio
    result["funding_pos_neg_ratio_10"] = (
        (result["funding_binance"] > 0).astype(np.float64)
        .rolling(10, min_periods=1).mean().astype(np.float32)
    )

    return result
