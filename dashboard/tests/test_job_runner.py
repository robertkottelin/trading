"""Tests for job_runner — subprocess lifecycle + stdout capture."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from dashboard.lib.job_runner import spawn_job


def test_short_job_succeeds(tmp_path):
    log = tmp_path / "job.log"
    cmd = [sys.executable, "-c",
           "import sys; print('hello'); print('world'); sys.exit(0)"]
    h = spawn_job("test", cmd, cwd=tmp_path, log_path=log)
    # Wait briefly for completion
    for _ in range(50):
        if not h.is_running():
            break
        time.sleep(0.05)
    assert not h.is_running()
    assert h.status() == "succeeded"
    assert h.returncode == 0
    # Capture should include both prints
    time.sleep(0.1)  # let reader flush
    tail = h.tail()
    assert any("hello" in l for l in tail)
    assert any("world" in l for l in tail)
    # Log file should also contain output
    text = log.read_text()
    assert "hello" in text and "world" in text


def test_failing_job_reports_failure(tmp_path):
    log = tmp_path / "job.log"
    cmd = [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]
    h = spawn_job("test", cmd, cwd=tmp_path, log_path=log)
    for _ in range(50):
        if not h.is_running():
            break
        time.sleep(0.05)
    assert "failed (3)" in h.status()
    assert h.returncode == 3


def test_kill_long_running_job(tmp_path):
    log = tmp_path / "job.log"
    cmd = [sys.executable, "-c", "import time; time.sleep(60)"]
    h = spawn_job("test", cmd, cwd=tmp_path, log_path=log)
    time.sleep(0.2)
    assert h.is_running()
    h.kill()
    time.sleep(0.3)
    assert not h.is_running()


def test_tail_returns_last_n(tmp_path):
    log = tmp_path / "job.log"
    cmd = [sys.executable, "-c",
           "import sys\n"
           "for i in range(20):\n"
           "    print(f'line_{i}')\n"
           "    sys.stdout.flush()\n"]
    h = spawn_job("test", cmd, cwd=tmp_path, log_path=log)
    for _ in range(60):
        if not h.is_running():
            break
        time.sleep(0.05)
    time.sleep(0.2)
    tail3 = h.tail(n=3)
    assert len(tail3) == 3
    assert tail3[-1] == "line_19"
