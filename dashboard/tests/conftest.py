"""Shared test fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def synthetic_state_dir(tmp_path, monkeypatch):
    """Build a fake state_data/ directory with realistic JSONL/JSON files
    and point the state module at it.
    """
    state_dir = tmp_path / "state_data"
    llm_dir = tmp_path / "llm_agent"
    log_dir = tmp_path / "logs"
    config_dir = tmp_path / "config"
    for d in (state_dir, llm_dir, log_dir, config_dir):
        d.mkdir(parents=True)

    # heartbeat
    (state_dir / "heartbeat.json").write_text(json.dumps({
        "timestamp": "2026-05-16T11:02:07+00:00",
        "run_number": 5,
        "status": "completed",
        "pid": 12345,
        "elapsed_s": 42.0,
        "next_run_at": "2026-05-16T11:07:07+00:00",
        "last_success": True,
    }))

    # trades.jsonl
    trades_lines = [
        {"timestamp": "2026-05-16T10:55:00+00:00", "action": "ENTRY",
         "direction": "LONG", "fill_price": 77000, "size": 0.01,
         "status": "FILLED", "mode": "paper"},
        {"timestamp": "2026-05-16T11:02:38+00:00", "action": "REJECTED",
         "direction": "NO_TRADE", "rejection_reason": "direction is NO_TRADE",
         "status": "REJECTED", "mode": "paper", "confidence": 0.48},
    ]
    (state_dir / "trades.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trades_lines) + "\n"
    )

    # portfolio.jsonl
    portfolio_lines = [
        {"timestamp": "2026-04-08T22:00:00+00:00", "equity": 100.0,
         "free_collateral": 95.0, "margin_pct": 5.0, "positions": []},
        {"timestamp": "2026-04-08T23:36:31+00:00", "equity": 95.6,
         "free_collateral": 93.2, "margin_pct": 2.5,
         "positions": [{"market": "BTC-USD", "side": "LONG",
                        "size": "0.0017", "entry_price": "71115",
                        "unrealized_pnl": "-0.04"}]},
    ]
    (state_dir / "portfolio.jsonl").write_text(
        "\n".join(json.dumps(p) for p in portfolio_lines) + "\n"
    )

    # decisions.jsonl
    decisions_lines = [
        {"timestamp": "2026-05-16T10:55:00+00:00", "direction": "LONG",
         "confidence": 0.72, "entry_price": 77000, "take_profit": 78000,
         "stop_loss": 76500, "duration_minutes": 120, "position_size_usd": 50,
         "rationale": "...",
         "market_conditions": {"btc_price": 77000, "funding_rate": 1e-5, "fng_value": 35},
         "model_consensus": {"bullish_count": 4, "bearish_count": 1,
                             "neutral_count": 64, "total": 69,
                             "weighted_score": 0.5, "avg_raw_score": 0.12}},
        {"timestamp": "2026-05-16T11:02:36+00:00", "direction": "NO_TRADE",
         "confidence": 0.48, "entry_price": 0, "take_profit": 0,
         "stop_loss": 0, "duration_minutes": 0, "position_size_usd": 0,
         "rationale": "ML models neutral",
         "market_conditions": {"btc_price": 77932, "funding_rate": -2e-6, "fng_value": 31},
         "model_consensus": {"bullish_count": 0, "bearish_count": 0,
                             "neutral_count": 69, "total": 69,
                             "weighted_score": 0.0, "avg_raw_score": -0.219}},
    ]
    (state_dir / "decisions.jsonl").write_text(
        "\n".join(json.dumps(d) for d in decisions_lines) + "\n"
    )

    # llm_agent/decision.json (latest)
    (llm_dir / "decision.json").write_text(json.dumps(decisions_lines[-1]))

    # llm_agent/decision_history.json (array form)
    (llm_dir / "decision_history.json").write_text(json.dumps(decisions_lines))

    # retrain_state.json
    (state_dir / "retrain_state.json").write_text(json.dumps({
        "status": "deployed",
        "last_retrain_started": "2026-04-07T16:13:03Z",
        "last_deployed": "2026-04-07T22:38:03Z",
        "pid": 464591,
        "failure_reason": None,
        "completed_at": "2026-04-07T22:36:22Z",
    }))

    # A pipeline log file
    log_file = log_dir / "pipeline_20260516_140207.log"
    log_file.write_text(
        "2026-05-16 14:02:07 [pipeline] INFO: Pipeline run started\n"
        "2026-05-16 14:02:08 [strategies.engine] INFO: Stage 1: ML signals\n"
        "2026-05-16 14:02:36 [pipeline] INFO: Pipeline run completed\n"
    )

    # Patch the module-level paths
    from dashboard.lib import state as state_mod
    monkeypatch.setattr(state_mod, "STATE_DIR", state_dir)
    monkeypatch.setattr(state_mod, "LLM_DIR", llm_dir)
    monkeypatch.setattr(state_mod, "LOG_DIR", log_dir)
    monkeypatch.setattr(state_mod, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(state_mod, "REPO_ROOT", tmp_path)

    return {
        "root": tmp_path,
        "state_dir": state_dir,
        "llm_dir": llm_dir,
        "log_dir": log_dir,
        "config_dir": config_dir,
    }
