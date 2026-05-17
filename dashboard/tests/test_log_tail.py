"""Tests for log_tail helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboard.lib.log_tail import tail_lines, parse_log_line, filter_lines


@pytest.fixture
def sample_log(tmp_path):
    p = tmp_path / "pipeline_20260516.log"
    lines = [
        "2026-05-16 11:00:01 [pipeline] INFO: Pipeline run started",
        "2026-05-16 11:00:02 [pipeline] DEBUG: Loading config",
        "2026-05-16 11:00:03 [strategies.engine] INFO: Stage 1: ML signals",
        "2026-05-16 11:00:04 [llm_agent.reasoning_agent] WARNING: Grok retry 1",
        "2026-05-16 11:00:05 [execution.dydx_executor] ERROR: Order rejected",
        "Traceback (most recent call last):",
        "  File \"/foo.py\", line 1, in <module>",
        "2026-05-16 11:00:06 [pipeline] INFO: Pipeline run completed",
    ]
    p.write_text("\n".join(lines) + "\n")
    return p


def test_tail_all(sample_log):
    lines = tail_lines(sample_log, n=100)
    assert len(lines) == 8


def test_tail_last_3(sample_log):
    lines = tail_lines(sample_log, n=3)
    assert len(lines) == 3
    assert "completed" in lines[-1]


def test_tail_missing_file(tmp_path):
    p = tmp_path / "nope.log"
    assert tail_lines(p) == []


def test_parse_log_line():
    parsed = parse_log_line("2026-05-16 11:00:01 [pipeline] INFO: Hello world")
    assert parsed is not None
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "pipeline"
    assert parsed["msg"] == "Hello world"


def test_parse_log_line_unparseable():
    assert parse_log_line("Traceback (most recent call last):") is None


def test_filter_by_level(sample_log):
    lines = tail_lines(sample_log, n=100)
    filtered = filter_lines(lines, level="WARNING")
    assert len(filtered) == 2  # WARNING + ERROR
    assert all("WARNING" in l or "ERROR" in l for l in filtered)


def test_filter_by_regex(sample_log):
    lines = tail_lines(sample_log, n=100)
    filtered = filter_lines(lines, regex=r"Stage \d")
    assert len(filtered) == 1
    assert "Stage 1" in filtered[0]


def test_filter_combined(sample_log):
    lines = tail_lines(sample_log, n=100)
    filtered = filter_lines(lines, level="ERROR", regex="Order")
    assert len(filtered) == 1
    assert "Order rejected" in filtered[0]
