"""Tests for bot_supervisor — uses a real subprocess (`sleep`) as a stand-in
for the trading bot to verify start/stop/pause/resume.

The supervisor checks that the running process's argv contains
'run_pipeline.py'. We bypass that by passing a custom command_override that
includes 'run_pipeline.py' as the script *argument* to a long-running command
that just sleeps. We use `python -c "import time; ...; time.sleep(60)"` and
inject `run_pipeline.py` as an argv element so the cmdline-match heuristic
matches.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import psutil
import pytest

from dashboard.lib import bot_supervisor, state


@pytest.fixture
def isolated_pid(tmp_path, monkeypatch):
    """Point supervisor at a tmp directory for the PID/pause files."""
    state_dir = tmp_path / "state_data"
    state_dir.mkdir()
    monkeypatch.setattr(state, "STATE_DIR", state_dir)
    monkeypatch.setattr(bot_supervisor, "PID_FILE", state_dir / ".bot_pid.json")
    monkeypatch.setattr(bot_supervisor, "PAUSE_FILE", state_dir / ".pause_flag.json")
    monkeypatch.setattr(bot_supervisor, "REPO_ROOT", tmp_path)
    # Create logs dir under tmp
    (tmp_path / "logs").mkdir()
    yield state_dir
    # Best-effort cleanup if a test leaks a process
    info = bot_supervisor._read_pid_file()
    if info and info.get("pid") and psutil.pid_exists(info["pid"]):
        try:
            os.kill(info["pid"], 9)
        except OSError:
            pass


def _dummy_bot_cmd():
    """Return a long-running command whose argv contains 'run_pipeline.py'."""
    return [sys.executable, "-c",
            "import sys, time; sys.stderr.write('dummy bot up\\n'); time.sleep(120)",
            "run_pipeline.py"]  # extra arg so cmdline heuristic matches


def test_status_when_no_pid_file(isolated_pid):
    s = bot_supervisor.status()
    assert s["state"] == "stopped"
    assert s["pid"] is None


def test_start_then_status_then_stop(isolated_pid):
    payload = bot_supervisor.start_bot(command_override=_dummy_bot_cmd())
    pid = payload["pid"]
    assert pid > 0
    # Give it a beat to actually start
    time.sleep(0.2)
    s = bot_supervisor.status()
    assert s["state"] == "running"
    assert s["pid"] == pid
    assert s["uptime_s"] is not None and s["uptime_s"] >= 0

    result = bot_supervisor.stop_bot(grace_seconds=2.0)
    assert result["signalled"] is True
    assert result["pid"] == pid
    # Process should be gone (psutil sometimes shows zombies briefly)
    time.sleep(0.3)
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    # PID file cleared
    s2 = bot_supervisor.status()
    assert s2["state"] == "stopped"


def test_start_twice_raises(isolated_pid):
    bot_supervisor.start_bot(command_override=_dummy_bot_cmd())
    try:
        with pytest.raises(RuntimeError, match="already running"):
            bot_supervisor.start_bot(command_override=_dummy_bot_cmd())
    finally:
        bot_supervisor.stop_bot(grace_seconds=2.0)


def test_stale_pid_file_self_heals(isolated_pid):
    """A PID file pointing at a non-existent process should be cleaned up."""
    # Write a PID file with a definitely-dead PID
    fake = {"pid": 9999999, "command": ["x", "run_pipeline.py"],
            "started_at": "2026-01-01T00:00:00+00:00"}
    bot_supervisor._write_pid_file(fake)
    s = bot_supervisor.status()
    assert s["state"] == "stopped"
    assert not bot_supervisor.PID_FILE.exists()


def test_pid_reuse_protection(isolated_pid):
    """If a PID gets reused by an unrelated process, supervisor must NOT kill it."""
    # Use our own PID (Python interpreter) as a 'foreign' process
    foreign_pid = os.getpid()
    fake = {"pid": foreign_pid, "command": ["x", "run_pipeline.py"],
            "started_at": "2026-01-01T00:00:00+00:00"}
    bot_supervisor._write_pid_file(fake)
    # status() should detect that this isn't our bot (the running cmdline is
    # pytest, not run_pipeline.py).
    s = bot_supervisor.status()
    assert s["state"] == "stopped"
    # The PID file should have been cleared, and the foreign process must still be alive.
    assert psutil.pid_exists(foreign_pid)


def test_pause_resume(isolated_pid):
    assert bot_supervisor.is_paused() is False
    bot_supervisor.pause()
    assert bot_supervisor.is_paused() is True
    payload = json.loads(bot_supervisor.PAUSE_FILE.read_text())
    assert "paused_at" in payload
    bot_supervisor.resume()
    assert bot_supervisor.is_paused() is False
    # Idempotent resume
    bot_supervisor.resume()
    assert bot_supervisor.is_paused() is False


def test_build_command_paper_testnet():
    cmd = bot_supervisor.build_command(mode="paper", network="testnet",
                                       interval=180, skip_signals=True)
    assert "--paper" in cmd
    assert "--testnet" in cmd
    assert "--skip-signals" in cmd
    assert "--interval" in cmd
    assert "180" in cmd


def test_build_command_live_mainnet():
    cmd = bot_supervisor.build_command(mode="live", network="mainnet",
                                       interval=300, verbose=True)
    assert "--live" in cmd
    assert "--no-testnet" in cmd
    assert "--verbose" in cmd


def test_stop_when_not_running_is_no_op(isolated_pid):
    result = bot_supervisor.stop_bot()
    assert result["signalled"] is False
    assert result["killed"] is False
