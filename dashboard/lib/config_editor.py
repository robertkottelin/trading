"""Safe YAML config editor — read, validate, diff, and write with backup."""

from __future__ import annotations

import difflib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dashboard.lib import state


BACKUP_DIR = state.CONFIG_DIR / ".backups"


def list_yaml_configs() -> list[Path]:
    """All *.yaml / *.yml files in config/, sorted by name."""
    if not state.CONFIG_DIR.exists():
        return []
    files = list(state.CONFIG_DIR.glob("*.yaml")) + list(state.CONFIG_DIR.glob("*.yml"))
    return sorted(files, key=lambda p: p.name)


def read_text(path: Path) -> str:
    return path.read_text()


def validate_yaml(text: str) -> tuple[bool, str | None, Any]:
    """Try to parse YAML.

    Returns (ok, error_message, parsed_value).
    """
    try:
        parsed = yaml.safe_load(text)
        return True, None, parsed
    except yaml.YAMLError as e:
        return False, str(e), None


def diff_text(old: str, new: str, path: str = "") -> str:
    """Unified diff between two strings (CR-friendly)."""
    diff = difflib.unified_diff(
        old.splitlines(keepends=False),
        new.splitlines(keepends=False),
        fromfile=f"{path} (current)",
        tofile=f"{path} (proposed)",
        lineterm="",
    )
    return "\n".join(diff)


def backup(path: Path) -> Path:
    """Copy *path* to config/.backups/<name>.<UTC timestamp>.bak; return backup path."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dst = BACKUP_DIR / f"{path.name}.{ts}.bak"
    shutil.copy2(path, dst)
    return dst


def save(path: Path, text: str) -> Path:
    """Validate and write text to *path*; back up the existing file first.

    Raises ValueError if the YAML is invalid (the file on disk is not touched).
    """
    ok, err, _ = validate_yaml(text)
    if not ok:
        raise ValueError(f"YAML is invalid — refusing to write. Error: {err}")
    backup_path = backup(path) if path.exists() else None
    path.write_text(text)
    return backup_path
