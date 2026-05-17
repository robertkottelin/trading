"""Bot lifecycle supervisor — start / stop / pause / resume / status.

The bot is `run_pipeline.py --loop`. Lifecycle state lives in
`state_data/.bot_pid.json`:

    {
      "pid": 12345,
      "command": ["python", "run_pipeline.py", "--loop", "--interval", "300", ...],
      "started_at": "2026-05-17T08:00:00+00:00",
      "mode": "paper",
      "network": "testnet",
      "log_path": "logs/dashboard_bot_20260517_080000.log"
    }

Pause is signalled via `state_data/.pause_flag.json` (read by reasoning_agent
at the top of each cycle).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from dashboard.lib import state

REPO_ROOT = state.REPO_ROOT
PID_FILE = state.bot_pid_file()
PAUSE_FILE = state.pause_flag_file()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_pid_file() -> dict[str, Any] | None:
    if not PID_FILE.exists():
        return None
    try:
        with open(PID_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_pid_file(payload: dict[str, Any]) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PID_FILE, "w") as f:
        json.dump(payload, f, indent=2)


def _clear_pid_file() -> None:
    try:
        PID_FILE.unlink()
    except FileNotFoundError:
        pass


def _pid_is_our_bot(pid: int, cmd_hint: list[str] | None = None) -> bool:
    """Return True only if PID is alive AND looks like our bot.

    Guards against PID reuse: a different process happening to be assigned
    the same PID after our bot died must not be killed.
    """
    if not psutil.pid_exists(pid):
        return False
    try:
        proc = psutil.Process(pid)
        cmdline = proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    joined = " ".join(cmdline)
    if "run_pipeline.py" not in joined:
        return False
    if cmd_hint:
        # If we have the original cmdline saved, require at least the script
        # name to still appear — strong evidence it is our process.
        return any("run_pipeline.py" in c for c in cmdline)
    return True


def status() -> dict[str, Any]:
    """Return a dict describing current bot status.

    Keys:
      - state: "running" | "stopped"
      - pid: int | None
      - command: list[str] | None
      - started_at: ISO timestamp | None
      - uptime_s: float | None
      - mode: "paper" | "live" | None
      - network: "testnet" | "mainnet" | None
      - log_path: str | None
      - paused: bool
      - heartbeat: latest heartbeat dict or None
    """
    info = _read_pid_file()
    paused = PAUSE_FILE.exists()
    hb = state.heartbeat()

    if not info:
        return {"state": "stopped", "pid": None, "command": None,
                "started_at": None, "uptime_s": None, "mode": None,
                "network": None, "log_path": None, "paused": paused,
                "heartbeat": hb}

    pid = info.get("pid")
    cmd = info.get("command")
    if pid is None or not _pid_is_our_bot(pid, cmd):
        # Stale — clean it up so the UI shows "stopped".
        _clear_pid_file()
        return {"state": "stopped", "pid": None, "command": None,
                "started_at": None, "uptime_s": None, "mode": None,
                "network": None, "log_path": None, "paused": paused,
                "heartbeat": hb}

    started = info.get("started_at")
    uptime = None
    if started:
        try:
            ts = datetime.fromisoformat(started)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            uptime = (datetime.now(timezone.utc) - ts).total_seconds()
        except ValueError:
            uptime = None

    return {
        "state": "running",
        "pid": pid,
        "command": cmd,
        "started_at": started,
        "uptime_s": uptime,
        "mode": info.get("mode"),
        "network": info.get("network"),
        "log_path": info.get("log_path"),
        "paused": paused,
        "heartbeat": hb,
    }


def build_command(*, mode: str = "paper", network: str = "testnet",
                  interval: int = 300, skip_signals: bool = False,
                  skip_web_search: bool = False, no_execute: bool = False,
                  verbose: bool = False, full: bool = False) -> list[str]:
    """Construct the argv for run_pipeline.py given UI flags."""
    cmd = [sys.executable, "run_pipeline.py", "--loop", "--interval", str(int(interval))]
    cmd.append("--live" if mode == "live" else "--paper")
    cmd.append("--testnet" if network == "testnet" else "--no-testnet")
    if skip_signals:
        cmd.append("--skip-signals")
    if skip_web_search:
        cmd.append("--skip-web-search")
    if no_execute:
        cmd.append("--no-execute")
    if verbose:
        cmd.append("--verbose")
    if full:
        cmd.append("--full")
    return cmd


def start_bot(*, mode: str = "paper", network: str = "testnet",
              interval: int = 300, skip_signals: bool = False,
              skip_web_search: bool = False, no_execute: bool = False,
              verbose: bool = False, full: bool = False,
              command_override: list[str] | None = None) -> dict[str, Any]:
    """Start the bot as a background subprocess.

    Raises RuntimeError if the bot is already running.
    """
    existing = status()
    if existing["state"] == "running":
        raise RuntimeError(f"Bot already running with PID {existing['pid']}")

    cmd = command_override or build_command(
        mode=mode, network=network, interval=interval,
        skip_signals=skip_signals, skip_web_search=skip_web_search,
        no_execute=no_execute, verbose=verbose, full=full,
    )

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dashboard_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    log_fh = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group → group-signal safely
        )
    except Exception:
        log_fh.close()
        raise

    payload = {
        "pid": proc.pid,
        "command": cmd,
        "started_at": _now_iso(),
        "mode": mode,
        "network": network,
        "interval": int(interval),
        "log_path": str(log_path),
    }
    _write_pid_file(payload)
    return payload


def stop_bot(grace_seconds: float = 10.0) -> dict[str, Any]:
    """Send SIGTERM to the bot's process group; SIGKILL after grace period.

    Always clears the PID file at the end (even if the process was already gone).
    """
    info = _read_pid_file()
    result = {"signalled": False, "killed": False, "pid": None}
    if not info:
        return result

    pid = info.get("pid")
    cmd = info.get("command")
    result["pid"] = pid

    if pid is None or not _pid_is_our_bot(pid, cmd):
        _clear_pid_file()
        return result

    # Try to signal the process group first; fall back to the PID.
    try:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        result["signalled"] = True

        deadline = time.time() + grace_seconds
        while time.time() < deadline:
            if not psutil.pid_exists(pid):
                break
            time.sleep(0.2)

        if psutil.pid_exists(pid):
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            result["killed"] = True
            # Brief wait for the kernel to reap
            for _ in range(20):
                if not psutil.pid_exists(pid):
                    break
                time.sleep(0.1)
    finally:
        _clear_pid_file()

    return result


def restart_bot(**kwargs) -> dict[str, Any]:
    """Stop the bot (if running), then start with the given flags."""
    stop_bot()
    time.sleep(0.5)
    return start_bot(**kwargs)


def pause() -> None:
    """Create the pause flag — bot honours this at the top of its next cycle."""
    PAUSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PAUSE_FILE.write_text(json.dumps({
        "paused_at": _now_iso(),
        "reason": "dashboard pause",
    }))


def resume() -> None:
    """Remove the pause flag."""
    try:
        PAUSE_FILE.unlink()
    except FileNotFoundError:
        pass


def is_paused() -> bool:
    return PAUSE_FILE.exists()
