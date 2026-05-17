"""Data loaders for the explorer — CSV / Parquet / JSON readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.lib import state


def list_data_files() -> dict[str, list[Path]]:
    """Group readable data files by their parent directory."""
    groups: dict[str, list[Path]] = {}
    for label, root in [
        ("market_context_data", state.MARKET_DATA_DIR),
        ("processed_data", state.PROCESSED_DATA_DIR),
        ("raw_data", state.RAW_DATA_DIR),
    ]:
        if not root.exists():
            continue
        files = []
        for ext in ("*.csv", "*.parquet", "*.json", "*.jsonl"):
            files.extend(root.glob(ext))
        files.sort(key=lambda p: p.name)
        if files:
            groups[label] = files
    return groups


def file_size_kb(path: Path) -> int:
    try:
        return path.stat().st_size // 1024
    except OSError:
        return 0


def load_dataframe(path: Path, head_only: bool = False,
                   nrows: int | None = None) -> pd.DataFrame:
    """Load a CSV / Parquet / JSON file into a DataFrame.

    Args:
        path: file to load.
        head_only: if True, read only first 1000 rows (CSV).
        nrows: explicit row cap (CSV only); overrides head_only.
    """
    suffix = path.suffix.lower()
    cap = nrows if nrows is not None else (1000 if head_only else None)
    if suffix == ".csv":
        kwargs: dict[str, Any] = {}
        if cap is not None:
            kwargs["nrows"] = cap
        return pd.read_csv(path, **kwargs)
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        if cap is not None and len(df) > cap:
            df = df.head(cap)
        return df
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if cap is not None and i >= cap:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    import json
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return pd.DataFrame(rows)
    if suffix == ".json":
        import json
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            # one-row "record" view
            return pd.DataFrame([data])
        return pd.DataFrame()
    raise ValueError(f"Unsupported file type: {path.suffix}")


def detect_timestamp_col(df: pd.DataFrame) -> str | None:
    """Heuristically pick the column most likely to be a timestamp."""
    candidates = [c for c in df.columns
                  if c.lower() in ("timestamp", "time", "datetime", "date", "ts",
                                   "open_time", "close_time", "openTime", "closeTime")]
    if candidates:
        return candidates[0]
    # Fallback: any datetime dtype column
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def coerce_timestamp(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Return a copy of df with ts_col coerced to UTC datetime."""
    out = df.copy()
    # Numeric epochs (seconds or ms)
    if pd.api.types.is_numeric_dtype(out[ts_col]):
        max_val = out[ts_col].dropna().max()
        if pd.notna(max_val):
            # Heuristic — ms vs s
            if max_val > 1e12:
                out[ts_col] = pd.to_datetime(out[ts_col], unit="ms", utc=True)
            else:
                out[ts_col] = pd.to_datetime(out[ts_col], unit="s", utc=True)
            return out
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    return out


def numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
