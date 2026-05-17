"""Parse the latest pipeline log to figure out which stage we're in,
which have finished, and how long each took.

The reasoning_agent emits log lines like:
  "Stage 1/7: Generating ML signals..."
  "Stage 1.5/7: Generating conventional strategy signals..."
  ...
  "Stage 7/7: Executing trade..."

and run_pipeline.py emits markers like:
  "Pipeline run started at ..."
  "Step 1/2: Downloading market context data ..."
  "Step 2/2: Running reasoning agent..."
  "Pipeline run completed"  or  "Pipeline run FAILED"
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashboard.lib.log_tail import tail_lines, parse_log_line

# Ordered stage names — matches reasoning_agent.py emission strings.
STAGE_DEFS = [
    ("Market data refresh", r"Step 1/2: Downloading market context data"),
    ("Stage 0 — Position monitoring", r"Stage 0/7: Orphan cleanup"),
    ("Stage 1 — ML signals", r"Stage 1/7: Generating ML signals"),
    ("Stage 1.5 — Strategy signals", r"Stage 1\.5/7: Generating conventional strategy signals"),
    ("Stage 2 — Market context", r"Stage 2/7: Building market context"),
    ("Stage 3 — Portfolio state", r"Stage 3/7: Reading portfolio state"),
    ("Stage 4 — Resolve pending", r"Stage 4/7: Resolving pending decisions"),
    ("Stage 5 — Decision history", r"Stage 5/7: Loading decision history"),
    ("Stage 6 — Call Grok", r"Stage 6/7: Calling Grok"),
    ("Stage 7 — Execute trade", r"Stage 7/7: Executing trade"),
]
PAUSE_MARKER = r"PAUSE FLAG active"
CYCLE_START_RE = re.compile(r"Pipeline run started")
CYCLE_DONE_RE = re.compile(r"Pipeline run completed|Pipeline run FAILED")


def _parse_ts(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def parse_latest_cycle(log_path: Path, n_lines: int = 2000) -> dict[str, Any]:
    """Inspect the tail of a log file and return the most recent cycle's status.

    Returns:
        {
          "cycle_started": datetime|None,
          "cycle_finished": datetime|None,
          "stages": list of {"name", "status", "started_at", "duration_s"}
          "paused": bool,
          "raw_tail": list of lines used,
        }
    """
    lines = tail_lines(log_path, n=n_lines)
    # Find the last cycle start in the buffer
    last_start_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if CYCLE_START_RE.search(lines[i]):
            last_start_idx = i
            break

    if last_start_idx is None:
        # Couldn't find a cycle marker — show all stages as pending
        return {
            "cycle_started": None,
            "cycle_finished": None,
            "stages": [{"name": n, "status": "pending", "started_at": None,
                        "duration_s": None} for n, _ in STAGE_DEFS],
            "paused": any(PAUSE_MARKER in ln for ln in lines[-200:]),
            "raw_tail": lines[-200:],
        }

    cycle_lines = lines[last_start_idx:]
    cycle_started = None
    cycle_finished = None
    parsed_first = parse_log_line(cycle_lines[0])
    if parsed_first:
        cycle_started = _parse_ts(parsed_first["ts"])

    paused = any(PAUSE_MARKER in ln for ln in cycle_lines)

    # Find a cycle terminator if present
    for ln in cycle_lines:
        if CYCLE_DONE_RE.search(ln):
            p = parse_log_line(ln)
            if p:
                cycle_finished = _parse_ts(p["ts"])
            break

    # For each stage, find the first matching line in this cycle
    stage_hits: list[dict[str, Any]] = []
    for name, pattern in STAGE_DEFS:
        regex = re.compile(pattern)
        hit_idx = None
        hit_ts = None
        for i, ln in enumerate(cycle_lines):
            if regex.search(ln):
                p = parse_log_line(ln)
                if p:
                    hit_ts = _parse_ts(p["ts"])
                hit_idx = i
                break
        stage_hits.append({"name": name, "idx": hit_idx, "ts": hit_ts})

    # Compute durations: each completed stage runs until the next stage begins
    # (or until the cycle terminator if it's the last started one).
    started_stages = [s for s in stage_hits if s["idx"] is not None]
    out_stages: list[dict[str, Any]] = []
    for s in stage_hits:
        entry: dict[str, Any] = {"name": s["name"], "started_at": s["ts"],
                                  "duration_s": None}
        if s["idx"] is None:
            entry["status"] = "pending"
        else:
            # Find the next stage hit after this one
            later = [o for o in started_stages
                     if o["idx"] is not None and o["idx"] > s["idx"]]
            next_hit = min(later, key=lambda o: o["idx"]) if later else None
            if next_hit is not None and s["ts"] and next_hit["ts"]:
                entry["duration_s"] = (next_hit["ts"] - s["ts"]).total_seconds()
                entry["status"] = "done"
            elif cycle_finished and s["ts"]:
                entry["duration_s"] = (cycle_finished - s["ts"]).total_seconds()
                entry["status"] = "done"
            else:
                entry["status"] = "running"
        out_stages.append(entry)

    # If pause flag was active and we're past Stage 0, mark subsequent stages
    # as 'paused' rather than 'pending'.
    if paused:
        seen_pause = False
        for entry in out_stages:
            if entry["status"] == "pending":
                if seen_pause or "Stage 0" in entry["name"] or "Market data" in entry["name"]:
                    pass
            if "Stage 0" in entry["name"] or "Market data" in entry["name"]:
                seen_pause = True
                continue
            if seen_pause and entry["status"] == "pending":
                entry["status"] = "paused"

    return {
        "cycle_started": cycle_started,
        "cycle_finished": cycle_finished,
        "stages": out_stages,
        "paused": paused,
        "raw_tail": cycle_lines[-200:],
    }
