"""Retrain manager — orchestrates 2-day ML model retraining and strategy tuning.

Called by run_pipeline.py on --full startup and every 48h during the loop.
Retraining runs as a background subprocess so the pipeline keeps trading.
Deployment is atomic: staging dirs are renamed to live only after full chain succeeds.

State file: state_data/retrain_state.json
Log file:   logs/retrain.log
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

STATE_PATH = Path("state_data/retrain_state.json")
RETRAIN_SCRIPT = Path("retraining/run_retrain_chain.sh")
RETRAIN_LOG = Path("logs/retrain.log")

RETRAIN_INTERVAL_H = 48  # hours between automatic retrains

# Staging → live directory/file mappings
STAGING_MAP = [
    ("models/v23_staging",                       "models/v23"),
    ("models/bearish_staging",                    "models/bearish"),
    ("config/strategy_params_staging.yaml",       "config/strategy_params.yaml"),
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> dict:
    """Load retrain state, returning defaults if missing or corrupt."""
    defaults = {
        "status": "idle",
        "last_retrain_started": None,
        "last_deployed": None,
        "pid": None,
        "failure_reason": None,
    }
    try:
        if STATE_PATH.exists():
            with open(STATE_PATH) as f:
                data = json.load(f)
            defaults.update(data)
    except Exception:
        pass
    return defaults


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def should_retrain() -> bool:
    """Return True if 48+ hours have elapsed since last successful deploy."""
    state = load_state()
    # Don't trigger if one is already running
    if state["status"] == "running":
        pid = state.get("pid")
        if pid and _pid_alive(pid):
            return False
        # PID gone but status still running → previous run crashed; allow retry
    # Don't trigger if a chain just completed and is awaiting deployment —
    # check_and_deploy() will handle it this cycle; triggering here would
    # overwrite the ready_to_deploy state and lose the trained models.
    if state["status"] == "ready_to_deploy":
        return False
    last = state.get("last_deployed") or state.get("last_retrain_started")
    if last is None:
        return True
    try:
        last_dt = datetime.strptime(last, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        elapsed_h = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
        return elapsed_h >= RETRAIN_INTERVAL_H
    except Exception:
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        # kill(0) succeeds for zombie processes too — check proc status
        status_path = Path(f"/proc/{pid}/status")
        if status_path.exists():
            txt = status_path.read_text()
            for line in txt.splitlines():
                if line.startswith("State:"):
                    return "Z" not in line  # Z = zombie
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def run_retrain_chain(background: bool = True):
    """Launch the full retrain chain.

    background=True: fire-and-forget subprocess, pipeline keeps running.
    background=False: block until complete (used with --full flag).
    """
    state = load_state()
    state["status"] = "running"
    state["last_retrain_started"] = _now_iso()
    state["failure_reason"] = None
    save_state(state)

    RETRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    script = str(RETRAIN_SCRIPT.resolve())
    os.chmod(script, 0o755)

    log.info("Launching retrain chain (background=%s) ...", background)
    with open(RETRAIN_LOG, "a") as logf:
        proc = subprocess.Popen(
            ["bash", script],
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(Path(".").resolve()),
            start_new_session=True,
        )

    state["pid"] = proc.pid
    save_state(state)
    log.info("Retrain chain started (pid=%d). Logs: %s", proc.pid, RETRAIN_LOG)

    if not background:
        log.info("Waiting for retrain chain to complete (blocking)...")
        proc.wait()
        rc = proc.returncode
        if rc != 0:
            log.error("Retrain chain failed (exit=%d). Check %s", rc, RETRAIN_LOG)
        else:
            log.info("Retrain chain finished successfully.")


def check_and_deploy(strategy_engine=None) -> bool:
    """Check if staging is ready and deploy atomically.

    Called every pipeline cycle. If staging is ready, renames dirs atomically
    and reloads strategy engine. Returns True if deployment occurred.
    """
    state = load_state()
    if state.get("status") != "ready_to_deploy":
        return False

    log.info("Staging ready — deploying new models and strategy params...")

    try:
        _atomic_deploy()
    except Exception as e:
        log.error("Deployment failed: %s", e)
        state["status"] = "failed"
        state["failure_reason"] = str(e)
        save_state(state)
        return False

    state["status"] = "deployed"
    state["last_deployed"] = _now_iso()
    save_state(state)

    log.info("Deployment complete. New models and params are live.")

    if strategy_engine is not None:
        try:
            strategy_engine.reload_params()
            log.info("Strategy engine reloaded with new parameters.")
        except Exception as e:
            log.warning("Strategy engine reload failed: %s", e)

    return True


def _atomic_deploy():
    """Rename staging dirs/files to live paths atomically (same filesystem)."""
    for staging, live in STAGING_MAP:
        staging_path = Path(staging)
        live_path = Path(live)

        if not staging_path.exists():
            log.debug("Staging path missing, skipping: %s", staging)
            continue

        # Move live to backup, then promote staging
        backup = Path(str(live_path) + "_old")
        if live_path.exists():
            if backup.exists():
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
            live_path.rename(backup)

        staging_path.rename(live_path)
        log.info("  Deployed: %s → %s", staging, live)

        # Clean up backup
        if backup.exists():
            if backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
