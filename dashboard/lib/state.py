"""State file loaders — read JSONL/JSON artifacts from state_data/ and llm_agent/.

All loaders return plain Python dicts/DataFrames so the rest of the dashboard
can stay framework-agnostic. Streamlit caching is applied at the page level
(via @st.cache_data wrappers) rather than here, so these helpers also work in
tests and ad-hoc scripts.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Resolve the trading repo root from this file's location:
# dashboard/lib/state.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPO_ROOT / "state_data"
LLM_DIR = REPO_ROOT / "llm_agent"
LOG_DIR = REPO_ROOT / "logs"
CONFIG_DIR = REPO_ROOT / "config"
MODELS_DIR = REPO_ROOT / "models"
MARKET_DATA_DIR = REPO_ROOT / "market_context_data"
PROCESSED_DATA_DIR = REPO_ROOT / "processed_data"
RAW_DATA_DIR = REPO_ROOT / "raw_data"


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON file, return None if missing or unparseable."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _read_jsonl(path: Path, tail: int | None = None) -> list[dict[str, Any]]:
    """Read a JSONL file. If *tail* is set, read only the last N lines.

    Uses a reverse-scan for efficient tailing of large append-only files.
    """
    if not path.exists():
        return []
    if tail is None:
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    # Tail mode: chunked reverse scan, keep growing the line buffer until we
    # have at least `tail` non-empty parseable records (with headroom for any
    # blank/garbage lines that get filtered out).
    target = tail
    chunk_size = 8192
    parsed: list[dict[str, Any]] = []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        buf = b""
        while end > 0:
            read_size = min(chunk_size, end)
            end -= read_size
            f.seek(end)
            buf = f.read(read_size) + buf
            # Collect candidate lines from the current buffer
            parts = buf.split(b"\n")
            # The first chunk may be a partial line — keep it in buf until we
            # read more (or hit start of file).
            if end > 0:
                buf = parts[0]
                candidate_lines = parts[1:]
            else:
                buf = b""
                candidate_lines = parts
            # Parse from the end backwards, accumulating until we have target records
            text_candidates = [ln.decode("utf-8", errors="replace").strip()
                               for ln in candidate_lines]
            text_candidates = [ln for ln in text_candidates if ln]
            new_parsed: list[dict[str, Any]] = []
            for ln in reversed(text_candidates):
                try:
                    new_parsed.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
                if len(new_parsed) + len(parsed) >= target:
                    break
            parsed = list(reversed(new_parsed)) + parsed
            if len(parsed) >= target:
                break
    return parsed[-target:]


def heartbeat() -> dict[str, Any] | None:
    """Return latest heartbeat dict or None if not yet written."""
    return _read_json(STATE_DIR / "heartbeat.json")


def heartbeat_age_seconds() -> float | None:
    """Seconds since the last heartbeat timestamp, or None if missing."""
    hb = heartbeat()
    if not hb or "timestamp" not in hb:
        return None
    try:
        ts = datetime.fromisoformat(hb["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except (ValueError, TypeError):
        return None


def latest_decision() -> dict[str, Any] | None:
    """Return contents of llm_agent/decision.json (single dict)."""
    return _read_json(LLM_DIR / "decision.json")


def decision_history(tail: int | None = None) -> list[dict[str, Any]]:
    """Return decisions.jsonl from state_data/, optionally last N records."""
    return _read_jsonl(STATE_DIR / "decisions.jsonl", tail=tail)


def trades(tail: int | None = None) -> list[dict[str, Any]]:
    """Return trades.jsonl records (executed/rejected/etc)."""
    return _read_jsonl(STATE_DIR / "trades.jsonl", tail=tail)


def portfolio_snapshots(tail: int | None = None) -> list[dict[str, Any]]:
    """Return portfolio.jsonl snapshots."""
    return _read_jsonl(STATE_DIR / "portfolio.jsonl", tail=tail)


def latest_portfolio() -> dict[str, Any] | None:
    """Last portfolio snapshot or None."""
    snaps = portfolio_snapshots(tail=1)
    return snaps[0] if snaps else None


def retrain_state() -> dict[str, Any] | None:
    return _read_json(STATE_DIR / "retrain_state.json")


def grok_failures() -> dict[str, Any] | None:
    return _read_json(STATE_DIR / "grok_failures.json")


def llm_decision_history() -> list[dict[str, Any]]:
    """Read llm_agent/decision_history.json (array of historical decisions
    with resolved outcomes: TP_HIT / SL_HIT / EXPIRED / PENDING)."""
    data = _read_json(LLM_DIR / "decision_history.json")
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "history" in data:
        return list(data["history"])
    return []


def portfolio_df(tail: int | None = 500) -> pd.DataFrame:
    """Return portfolio snapshots as a DataFrame indexed by timestamp."""
    snaps = portfolio_snapshots(tail=tail)
    if not snaps:
        return pd.DataFrame(columns=["timestamp", "equity", "free_collateral", "margin_pct"])
    df = pd.DataFrame(snaps)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def decisions_df(tail: int | None = 500) -> pd.DataFrame:
    """Flatten decisions.jsonl into a tabular DataFrame (one row per decision)."""
    recs = decision_history(tail=tail)
    if not recs:
        return pd.DataFrame()
    rows = []
    for r in recs:
        row = {
            "timestamp": r.get("timestamp"),
            "direction": r.get("direction"),
            "confidence": r.get("confidence"),
            "entry_price": r.get("entry_price"),
            "take_profit": r.get("take_profit"),
            "stop_loss": r.get("stop_loss"),
            "duration_minutes": r.get("duration_minutes"),
            "position_size_usd": r.get("position_size_usd"),
            "rationale": r.get("rationale", "")[:200],
        }
        mc = r.get("model_consensus") or {}
        row.update({
            "bullish_count": mc.get("bullish_count"),
            "bearish_count": mc.get("bearish_count"),
            "neutral_count": mc.get("neutral_count"),
            "weighted_score": mc.get("weighted_score"),
            "avg_raw_score": mc.get("avg_raw_score"),
        })
        cond = r.get("market_conditions") or {}
        row.update({
            "btc_price": cond.get("btc_price"),
            "funding_rate": cond.get("funding_rate"),
            "fng_value": cond.get("fng_value"),
        })
        rows.append(row)
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def trades_df(tail: int | None = 500) -> pd.DataFrame:
    """Flatten trades.jsonl into a DataFrame."""
    recs = trades(tail=tail)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def latest_log_path() -> Path | None:
    """Return the most recently modified pipeline_*.log file, or pipeline_live.log
    if it exists (preferred for loop mode)."""
    live = LOG_DIR / "pipeline_live.log"
    candidates: list[Path] = []
    if live.exists():
        candidates.append(live)
    if LOG_DIR.exists():
        candidates.extend(p for p in LOG_DIR.glob("pipeline_*.log") if p != live)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def list_log_files() -> list[Path]:
    """All pipeline_*.log + retrain*.log files sorted newest first."""
    if not LOG_DIR.exists():
        return []
    logs = list(LOG_DIR.glob("pipeline_*.log")) + list(LOG_DIR.glob("retrain*.log"))
    return sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True)


def bot_pid_file() -> Path:
    return STATE_DIR / ".bot_pid.json"


def pause_flag_file() -> Path:
    return STATE_DIR / ".pause_flag.json"


def is_paused() -> bool:
    return pause_flag_file().exists()


def pid_info() -> dict[str, Any] | None:
    """Return the .bot_pid.json contents (PID, command, started_at) if present."""
    return _read_json(bot_pid_file())


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe to display via st.dataframe / Arrow serialization.

    Mixed-type object columns (e.g. trades.jsonl where 'size' is sometimes a
    string and sometimes a float) trigger pyarrow conversion errors. Coerce
    such columns to plain strings.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            # Stringify any column that has more than one non-null Python type
            types = {type(v).__name__ for v in out[col].dropna().head(50)}
            if len(types) > 1:
                out[col] = out[col].astype("string")
    return out
