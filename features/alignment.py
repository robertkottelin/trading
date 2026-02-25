"""Grid alignment utilities for merging multi-resolution data onto 5-minute grid.

All joins use open_time_ms as the key. Only forward-fill is used (never bfill
or interpolate) to prevent look-ahead bias.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = "raw_data"


def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from raw_data/ directory."""
    path = os.path.join(DATA_DIR, filename)
    return pd.read_csv(path)


def _datetime_series_to_ms(dt_series: pd.Series) -> pd.Series:
    """Convert a datetime64 Series to milliseconds since epoch (int64).

    Handles both datetime64[ns] and datetime64[us] resolutions.
    """
    # Use the Unix epoch approach which works regardless of resolution
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return ((dt_series - epoch) // pd.Timedelta(milliseconds=1)).astype(np.int64)


def ts_col_to_ms(df: pd.DataFrame, col: str) -> pd.Series:
    """Normalize any timestamp column to milliseconds (int64).

    Handles:
    - open_time_ms / timestamp_ms / funding_time_ms: already ms, passthrough
    - ISO strings like "2023-11-13T15:25:00.000Z": parse to ms
    - date strings like "2023-11-13": parse to midnight UTC ms
    """
    s = df[col]
    # If already numeric (ms), just return as int
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.int64)
    # String timestamps — parse with pandas
    parsed = pd.to_datetime(s, utc=True)
    return _datetime_series_to_ms(parsed)


def align_5m(df: pd.DataFrame, grid_ms: pd.Series,
             ts_col: str, value_cols: list[str],
             prefix: str) -> pd.DataFrame:
    """Direct merge for 5-minute resolution sources.

    Args:
        df: Source DataFrame
        grid_ms: Series of open_time_ms values from the master grid
        ts_col: Name of the timestamp column in df (will be converted to ms)
        value_cols: Columns to extract from df
        prefix: Prefix to add to column names (e.g. "bybit_")

    Returns:
        DataFrame with open_time_ms + prefixed value columns
    """
    src = df[value_cols + [ts_col]].copy()
    src["open_time_ms"] = ts_col_to_ms(src, ts_col)
    src = src.drop(columns=[ts_col])

    # Rename value columns with prefix
    rename = {c: f"{prefix}{c}" for c in value_cols}
    src = src.rename(columns=rename)

    # Merge onto grid
    grid = pd.DataFrame({"open_time_ms": grid_ms})
    result = grid.merge(src, on="open_time_ms", how="left")

    # Cast to float32
    for c in rename.values():
        if c in result.columns:
            result[c] = result[c].astype(np.float32)
    return result


def align_ffill(df: pd.DataFrame, grid_ms: pd.Series,
                ts_col: str, value_cols: list[str],
                prefix: str) -> pd.DataFrame:
    """Forward-fill alignment for coarser-than-5m sources (hourly, 8h, etc).

    The source timestamp is floored to the nearest 5-minute boundary,
    merged onto the grid, and then forward-filled.
    """
    src = df[value_cols + [ts_col]].copy()
    src["ts_ms"] = ts_col_to_ms(src, ts_col)
    src = src.drop(columns=[ts_col])

    # Floor to 5-minute boundary (300_000 ms)
    src["open_time_ms"] = (src["ts_ms"] // 300_000) * 300_000
    src = src.drop(columns=["ts_ms"])

    # Deduplicate: keep last value per 5m bucket
    src = src.drop_duplicates(subset="open_time_ms", keep="last")

    # Rename
    rename = {c: f"{prefix}{c}" for c in value_cols}
    src = src.rename(columns=rename)

    # Merge onto grid and ffill
    grid = pd.DataFrame({"open_time_ms": grid_ms})
    result = grid.merge(src, on="open_time_ms", how="left")
    for c in rename.values():
        if c in result.columns:
            result[c] = result[c].ffill().astype(np.float32)
    return result


def align_daily(df: pd.DataFrame, grid_ms: pd.Series,
                date_col: str, value_cols: list[str],
                prefix: str, lag_days: int = 0) -> pd.DataFrame:
    """Align daily data onto 5-minute grid with forward-fill.

    Args:
        lag_days: Number of days to shift forward (e.g. 1 for macro data
                  where market close is available next day)
    """
    src = df[value_cols + [date_col]].copy()

    # Parse date to midnight UTC milliseconds
    dates = pd.to_datetime(src[date_col], utc=True)
    if lag_days > 0:
        dates = dates + pd.Timedelta(days=lag_days)
    src["open_time_ms"] = _datetime_series_to_ms(dates)
    src = src.drop(columns=[date_col])

    # Floor to 5m boundary (should already be at midnight, but be safe)
    src["open_time_ms"] = (src["open_time_ms"] // 300_000) * 300_000

    # Deduplicate
    src = src.drop_duplicates(subset="open_time_ms", keep="last")

    # Rename
    rename = {c: f"{prefix}{c}" for c in value_cols}
    src = src.rename(columns=rename)

    # Merge and ffill
    grid = pd.DataFrame({"open_time_ms": grid_ms})
    result = grid.merge(src, on="open_time_ms", how="left")
    for c in rename.values():
        if c in result.columns:
            result[c] = result[c].ffill().astype(np.float32)
    return result


def align_events(df: pd.DataFrame, grid_ms: pd.Series,
                 ts_col: str, agg_dict: dict,
                 prefix: str) -> pd.DataFrame:
    """Aggregate tick-level events into 5-minute buckets.

    Args:
        agg_dict: Dict of {column: agg_func} e.g. {"size": "sum", "side": "count"}
        prefix: Prefix for output column names
    """
    src = df.copy()
    src["ts_ms"] = ts_col_to_ms(src, ts_col)
    src["open_time_ms"] = (src["ts_ms"] // 300_000) * 300_000

    # Aggregate per 5m bucket
    agg = src.groupby("open_time_ms").agg(agg_dict).reset_index()

    # Rename columns
    rename = {c: f"{prefix}{c}" for c in agg_dict.keys()}
    agg = agg.rename(columns=rename)

    # Merge onto grid — fill missing buckets with 0 (no events = zero)
    grid = pd.DataFrame({"open_time_ms": grid_ms})
    result = grid.merge(agg, on="open_time_ms", how="left")
    for c in rename.values():
        if c in result.columns:
            result[c] = result[c].fillna(0).astype(np.float32)
    return result


def rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std."""
    m = s.rolling(window, min_periods=max(1, window // 2)).mean()
    st = s.rolling(window, min_periods=max(1, window // 2)).std()
    return ((s - m) / st.replace(0, np.nan)).astype(np.float32)
