"""Tests for data_loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dashboard.lib import data_loaders, state


@pytest.fixture
def sample_files(tmp_path, monkeypatch):
    market = tmp_path / "market_context_data"
    market.mkdir()
    # CSV
    (market / "candles.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-05-16T00:00:00,77000,77500,76500,77100,123.4\n"
        "2026-05-16T00:05:00,77100,77600,77000,77400,150.0\n"
    )
    # JSONL
    rows = [{"ts": 1, "value": 10.0}, {"ts": 2, "value": 11.0}]
    (market / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # JSON (single object)
    (market / "config.json").write_text(json.dumps({"a": 1, "b": 2}))
    # Parquet
    df = pd.DataFrame({"t": pd.date_range("2026-05-16", periods=5),
                       "x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    df.to_parquet(market / "frame.parquet")

    monkeypatch.setattr(state, "MARKET_DATA_DIR", market)
    monkeypatch.setattr(state, "PROCESSED_DATA_DIR", tmp_path / "processed_data")
    monkeypatch.setattr(state, "RAW_DATA_DIR", tmp_path / "raw_data")

    return market


def test_list_data_files(sample_files):
    groups = data_loaders.list_data_files()
    assert "market_context_data" in groups
    names = [p.name for p in groups["market_context_data"]]
    assert "candles.csv" in names
    assert "events.jsonl" in names
    assert "frame.parquet" in names


def test_load_csv(sample_files):
    p = sample_files / "candles.csv"
    df = data_loaders.load_dataframe(p)
    assert len(df) == 2
    assert "close" in df.columns


def test_load_csv_with_nrows_cap(sample_files):
    p = sample_files / "candles.csv"
    df = data_loaders.load_dataframe(p, nrows=1)
    assert len(df) == 1


def test_load_parquet(sample_files):
    p = sample_files / "frame.parquet"
    df = data_loaders.load_dataframe(p)
    assert len(df) == 5
    assert "x" in df.columns


def test_load_jsonl(sample_files):
    p = sample_files / "events.jsonl"
    df = data_loaders.load_dataframe(p)
    assert len(df) == 2
    assert "ts" in df.columns


def test_load_json_dict(sample_files):
    p = sample_files / "config.json"
    df = data_loaders.load_dataframe(p)
    assert len(df) == 1
    assert "a" in df.columns


def test_detect_timestamp_col(sample_files):
    df = data_loaders.load_dataframe(sample_files / "candles.csv")
    assert data_loaders.detect_timestamp_col(df) == "timestamp"

    # Frame.parquet has column "t"
    fdf = data_loaders.load_dataframe(sample_files / "frame.parquet")
    # "t" isn't in the heuristic list, but it's already datetime — should fall back
    assert data_loaders.detect_timestamp_col(fdf) == "t"


def test_coerce_timestamp_from_iso(sample_files):
    df = data_loaders.load_dataframe(sample_files / "candles.csv")
    out = data_loaders.coerce_timestamp(df, "timestamp")
    assert pd.api.types.is_datetime64_any_dtype(out["timestamp"])


def test_coerce_timestamp_from_unix(tmp_path):
    df = pd.DataFrame({"ts": [1715846400, 1715846700], "v": [1, 2]})
    out = data_loaders.coerce_timestamp(df, "ts")
    assert pd.api.types.is_datetime64_any_dtype(out["ts"])


def test_numeric_cols():
    df = pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5], "c": ["x", "y"]})
    assert set(data_loaders.numeric_cols(df)) == {"a", "b"}


def test_unsupported_file_raises(tmp_path):
    p = tmp_path / "foo.xml"
    p.write_text("<x/>")
    with pytest.raises(ValueError):
        data_loaders.load_dataframe(p)
