"""Tests for dashboard.lib.state — JSONL/JSON loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dashboard.lib import state


def test_heartbeat_reads(synthetic_state_dir):
    hb = state.heartbeat()
    assert hb is not None
    assert hb["status"] == "completed"
    assert hb["run_number"] == 5
    assert hb["pid"] == 12345


def test_heartbeat_age_returns_positive_seconds(synthetic_state_dir):
    age = state.heartbeat_age_seconds()
    assert age is not None
    assert age > 0  # synthetic timestamp is in the past


def test_heartbeat_missing_returns_none(synthetic_state_dir):
    (synthetic_state_dir["state_dir"] / "heartbeat.json").unlink()
    assert state.heartbeat() is None
    assert state.heartbeat_age_seconds() is None


def test_latest_decision(synthetic_state_dir):
    d = state.latest_decision()
    assert d is not None
    assert d["direction"] == "NO_TRADE"
    assert d["model_consensus"]["neutral_count"] == 69


def test_decision_history_full(synthetic_state_dir):
    recs = state.decision_history()
    assert len(recs) == 2
    assert recs[0]["direction"] == "LONG"


def test_decision_history_tail(synthetic_state_dir):
    recs = state.decision_history(tail=1)
    assert len(recs) == 1
    assert recs[0]["direction"] == "NO_TRADE"


def test_trades(synthetic_state_dir):
    t = state.trades()
    assert len(t) == 2
    assert t[0]["action"] == "ENTRY"
    assert t[1]["action"] == "REJECTED"


def test_portfolio_snapshots(synthetic_state_dir):
    snaps = state.portfolio_snapshots()
    assert len(snaps) == 2
    assert snaps[-1]["equity"] == 95.6


def test_latest_portfolio(synthetic_state_dir):
    p = state.latest_portfolio()
    assert p is not None
    assert p["equity"] == 95.6
    assert p["positions"][0]["market"] == "BTC-USD"


def test_retrain_state(synthetic_state_dir):
    rs = state.retrain_state()
    assert rs is not None
    assert rs["status"] == "deployed"


def test_llm_decision_history(synthetic_state_dir):
    hist = state.llm_decision_history()
    assert isinstance(hist, list)
    assert len(hist) == 2


def test_decisions_df(synthetic_state_dir):
    df = state.decisions_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "weighted_score" in df.columns
    assert "btc_price" in df.columns
    assert df["weighted_score"].iloc[0] == 0.5


def test_trades_df(synthetic_state_dir):
    df = state.trades_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_portfolio_df(synthetic_state_dir):
    df = state.portfolio_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "equity" in df.columns


def test_latest_log_path(synthetic_state_dir):
    p = state.latest_log_path()
    assert p is not None
    assert p.name == "pipeline_20260516_140207.log"


def test_pid_and_pause_helpers(synthetic_state_dir):
    assert state.pid_info() is None  # no .bot_pid.json yet
    assert state.is_paused() is False
    # Create a fake pause flag
    state.pause_flag_file().write_text("{}")
    assert state.is_paused() is True
    state.pause_flag_file().unlink()
    assert state.is_paused() is False


def test_jsonl_tail_handles_corrupt_lines(synthetic_state_dir):
    """Garbage lines should be silently skipped, not crash."""
    (synthetic_state_dir["state_dir"] / "trades.jsonl").write_text(
        '{"timestamp":"2026-05-16T10:00:00+00:00","action":"OK"}\n'
        "garbage line\n"
        '{"timestamp":"2026-05-16T10:01:00+00:00","action":"OK2"}\n'
    )
    recs = state.trades()
    assert len(recs) == 2


def test_jsonl_tail_returns_last_n(synthetic_state_dir):
    """Tail of N should return the last N parseable records."""
    p = synthetic_state_dir["state_dir"] / "trades.jsonl"
    lines = [json.dumps({"timestamp": f"2026-05-16T10:{i:02d}:00+00:00",
                          "action": f"ACT_{i}"}) for i in range(10)]
    p.write_text("\n".join(lines) + "\n")
    recs = state.trades(tail=3)
    assert len(recs) == 3
    assert recs[-1]["action"] == "ACT_9"
    assert recs[0]["action"] == "ACT_7"
