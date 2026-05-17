"""Efficient log file tailing — read last N lines from large log files
without loading the whole file into memory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Loose regex for the pipeline log format:
#   2026-05-16 11:02:07 [pipeline] INFO: message
LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s"
    r"\[(?P<logger>[^\]]+)\]\s"
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL):\s"
    r"(?P<msg>.*)$"
)


def tail_lines(path: Path, n: int = 200) -> list[str]:
    """Return the last N lines of a file as a list of strings (no trailing
    newlines). Uses a chunked reverse scan to avoid loading the whole file.
    """
    if not path.exists() or n <= 0:
        return []
    chunk_size = 16384
    collected: list[bytes] = []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        buf = b""
        while end > 0 and len(collected) <= n:
            read_size = min(chunk_size, end)
            end -= read_size
            f.seek(end)
            buf = f.read(read_size) + buf
            parts = buf.split(b"\n")
            if end > 0:
                buf = parts[0]
                collected = parts[1:] + collected
            else:
                collected = parts + collected
                break
    text = [ln.decode("utf-8", errors="replace") for ln in collected if ln]
    return text[-n:]


def parse_log_line(line: str) -> dict[str, str] | None:
    """Parse a pipeline log line into structured fields; return None if it
    doesn't match the expected format (e.g. a traceback line)."""
    m = LOG_LINE_RE.match(line)
    if not m:
        return None
    return m.groupdict()


def filter_lines(lines: list[str], level: str | None = None,
                 regex: str | None = None) -> list[str]:
    """Filter a list of log lines.

    - level: only keep lines at or above this level
    - regex: case-insensitive substring/regex match
    """
    level_order = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    min_level = level_order.get(level.upper(), 0) if level else 0

    pat = re.compile(regex, re.IGNORECASE) if regex else None
    out = []
    for ln in lines:
        if level:
            parsed = parse_log_line(ln)
            # If we can't parse it, keep it only when no filter is requested
            if parsed is None:
                if min_level == 0:
                    pass  # keep
                else:
                    continue
            else:
                if level_order.get(parsed["level"], 0) < min_level:
                    continue
        if pat and not pat.search(ln):
            continue
        out.append(ln)
    return out


def colorize_level(line: str) -> str:
    """Wrap log level token with simple HTML span colors for st.markdown."""
    parsed = parse_log_line(line)
    if not parsed:
        return line
    colors = {
        "DEBUG": "#888",
        "INFO": "#1f77b4",
        "WARNING": "#e6a700",
        "ERROR": "#d62728",
        "CRITICAL": "#a31515",
    }
    color = colors.get(parsed["level"], "#000")
    return (
        f'<span style="color:#666">{parsed["ts"]}</span> '
        f'<span style="color:#7f7f7f">[{parsed["logger"]}]</span> '
        f'<b style="color:{color}">{parsed["level"]}</b>: '
        f'{parsed["msg"]}'
    )
