"""Tests for pipeline_parser — log → stage progression extraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.lib.pipeline_parser import parse_latest_cycle


def _write_log(path: Path, lines: list[str]):
    path.write_text("\n".join(lines) + "\n")


def test_completed_cycle(tmp_path):
    log = tmp_path / "pipeline.log"
    _write_log(log, [
        "2026-05-16 14:00:00 [pipeline] INFO: Pipeline run started at 2026-05-16 14:00 UTC",
        "2026-05-16 14:00:00 [pipeline] INFO: Step 1/2: Downloading market context data (last 24h) [all tiers]...",
        "2026-05-16 14:00:30 [pipeline] INFO: Market context data refreshed [all tiers]",
        "2026-05-16 14:00:30 [pipeline] INFO: Step 2/2: Running reasoning agent...",
        "2026-05-16 14:00:31 [llm_agent] INFO: Stage 0/7: Orphan cleanup + position monitoring + trailing stops...",
        "2026-05-16 14:00:32 [llm_agent] INFO: Stage 1/7: Generating ML signals...",
        "2026-05-16 14:00:35 [llm_agent] INFO: Stage 1.5/7: Generating conventional strategy signals...",
        "2026-05-16 14:00:36 [llm_agent] INFO: Stage 2/7: Building market context...",
        "2026-05-16 14:00:37 [llm_agent] INFO: Stage 3/7: Reading portfolio state...",
        "2026-05-16 14:00:38 [llm_agent] INFO: Stage 4/7: Resolving pending decisions...",
        "2026-05-16 14:00:39 [llm_agent] INFO: Stage 5/7: Loading decision history...",
        "2026-05-16 14:00:40 [llm_agent] INFO: Stage 6/7: Calling Grok for decision...",
        "2026-05-16 14:00:55 [llm_agent] INFO: Stage 7/7: Executing trade...",
        "2026-05-16 14:00:56 [pipeline] INFO: Reasoning agent completed successfully",
        "2026-05-16 14:00:56 [pipeline] INFO: Pipeline run completed",
    ])
    result = parse_latest_cycle(log)
    assert result["cycle_started"] is not None
    assert result["cycle_finished"] is not None
    assert result["paused"] is False
    stages_by_name = {s["name"]: s for s in result["stages"]}
    assert stages_by_name["Stage 1 — ML signals"]["status"] == "done"
    assert stages_by_name["Stage 1 — ML signals"]["duration_s"] is not None
    assert stages_by_name["Stage 7 — Execute trade"]["status"] == "done"
    # Stage 6 to Stage 7 = 15 seconds, Stage 7 to cycle_finished = 1 second
    assert 0 < stages_by_name["Stage 7 — Execute trade"]["duration_s"] <= 5


def test_in_progress_cycle(tmp_path):
    """Cycle with no terminator — last started stage should be 'running'."""
    log = tmp_path / "pipeline.log"
    _write_log(log, [
        "2026-05-16 14:00:00 [pipeline] INFO: Pipeline run started",
        "2026-05-16 14:00:31 [llm_agent] INFO: Stage 0/7: Orphan cleanup",
        "2026-05-16 14:00:32 [llm_agent] INFO: Stage 1/7: Generating ML signals...",
        "2026-05-16 14:00:33 [llm_agent] INFO: Stage 1.5/7: Generating conventional strategy signals...",
        # cycle truncated here
    ])
    result = parse_latest_cycle(log)
    assert result["cycle_finished"] is None
    stages_by_name = {s["name"]: s for s in result["stages"]}
    assert stages_by_name["Stage 1.5 — Strategy signals"]["status"] == "running"
    assert stages_by_name["Stage 7 — Execute trade"]["status"] == "pending"


def test_paused_cycle(tmp_path):
    log = tmp_path / "pipeline.log"
    _write_log(log, [
        "2026-05-16 14:00:00 [pipeline] INFO: Pipeline run started",
        "2026-05-16 14:00:31 [llm_agent] INFO: Stage 0/7: Orphan cleanup",
        "2026-05-16 14:00:32 [llm_agent] WARNING: PAUSE FLAG active (state_data/.pause_flag.json) — skipping Stages 1-7",
        "2026-05-16 14:00:32 [pipeline] INFO: Reasoning agent completed successfully",
        "2026-05-16 14:00:32 [pipeline] INFO: Pipeline run completed",
    ])
    result = parse_latest_cycle(log)
    assert result["paused"] is True


def test_no_cycle_markers(tmp_path):
    log = tmp_path / "pipeline.log"
    _write_log(log, [
        "2026-05-16 14:00:00 [foo] INFO: Some unrelated line",
    ])
    result = parse_latest_cycle(log)
    assert all(s["status"] in ("pending", "paused") for s in result["stages"])
