"""Background job runner — spawn subprocesses (backtest, retrain) and capture
their output line-by-line into an in-memory ring buffer.

Jobs survive Streamlit reruns thanks to st.session_state holding the JobHandle
references between reruns. The Popen lives on the file descriptor; reading
stdout happens in a daemon thread.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


class JobHandle:
    """Lifecycle wrapper around a single subprocess.

    Captures stdout/stderr to (a) an on-disk log file and (b) an in-memory
    ring buffer for quick UI display.
    """

    def __init__(self, name: str, cmd: list[str], cwd: Path,
                 log_path: Path, buffer_lines: int = 1000):
        self.name = name
        self.cmd = cmd
        self.cwd = cwd
        self.log_path = log_path
        self.buffer: deque[str] = deque(maxlen=buffer_lines)
        self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.finished_at: str | None = None
        self.returncode: int | None = None
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(self.log_path, "ab")
        self._proc = subprocess.Popen(
            self.cmd, cwd=str(self.cwd),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True, text=True, bufsize=1,
        )

        def _pump():
            try:
                assert self._proc and self._proc.stdout
                for line in iter(self._proc.stdout.readline, ""):
                    if not line:
                        break
                    self.buffer.append(line.rstrip("\n"))
                    try:
                        log_fh.write(line.encode("utf-8", errors="replace"))
                        log_fh.flush()
                    except OSError:
                        pass
            finally:
                try:
                    log_fh.close()
                except OSError:
                    pass
                rc = self._proc.wait() if self._proc else -1
                self.returncode = rc
                self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        self._reader = threading.Thread(target=_pump, daemon=True)
        self._reader.start()

    def is_running(self) -> bool:
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def status(self) -> str:
        if self._proc is None:
            return "not_started"
        rc = self._proc.poll()
        if rc is None:
            return "running"
        return "succeeded" if rc == 0 else f"failed ({rc})"

    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def kill(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            try:
                pgid = os.getpgid(self._proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(self._proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            # Wait up to 5 s, then SIGKILL
            for _ in range(50):
                if self._proc.poll() is not None:
                    return
                time.sleep(0.1)
            try:
                pgid = os.getpgid(self._proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(self._proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        finally:
            try:
                self._proc.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                pass

    def tail(self, n: int = 200) -> list[str]:
        if n >= len(self.buffer):
            return list(self.buffer)
        return list(self.buffer)[-n:]


def spawn_job(name: str, cmd: list[str], cwd: Path, log_path: Path,
              buffer_lines: int = 1000) -> JobHandle:
    handle = JobHandle(name=name, cmd=cmd, cwd=cwd, log_path=log_path,
                       buffer_lines=buffer_lines)
    handle.start()
    return handle


def find_other_running(name_pattern: str) -> list[dict[str, Any]]:
    """Scan running processes whose cmdline contains *name_pattern*.
    Used to detect external retrain/backtest jobs not started by this dashboard.
    """
    matches = []
    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if name_pattern in cmd:
            matches.append({"pid": p.info["pid"], "cmd": cmd,
                             "started_at": datetime.fromtimestamp(
                                 p.info["create_time"], tz=timezone.utc
                             ).isoformat(timespec="seconds")})
    return matches
