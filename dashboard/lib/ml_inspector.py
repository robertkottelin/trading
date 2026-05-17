"""Inspect ML model artifacts — production configs, weights, metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.lib import state


def find_production_configs() -> list[Path]:
    """Return paths to all production_config_*.json files under models/."""
    if not state.MODELS_DIR.exists():
        return []
    return sorted(state.MODELS_DIR.rglob("production_config_*.json"))


def load_config(path: Path) -> dict[str, Any]:
    """Read a production_config_*.json file."""
    with open(path) as f:
        return json.load(f)


def summarize_config(cfg: dict[str, Any], path: Path) -> dict[str, Any]:
    """Return a compact summary suitable for the UI."""
    weights = cfg.get("model_weights") or {}
    models_list = cfg.get("models")  # bearish format uses a "models" list
    metrics = cfg.get("metrics") or {}
    return {
        "path": str(path),
        "version": cfg.get("version"),
        "generated": cfg.get("generated"),
        "config_name": cfg.get("config_name"),
        "num_weighted_models": len(weights),
        "num_listed_models": len(models_list) if models_list else 0,
        "dd_limit": cfg.get("dd_limit"),
        "cooldown": cfg.get("cooldown"),
        "max_concurrent": cfg.get("max_concurrent"),
        "position_scale": cfg.get("position_scale"),
        "fee_rt": cfg.get("fee_rt"),
        "metrics": metrics,
        "weights": weights,
        "models_list": models_list or [],
    }


def weights_dataframe(weights: dict[str, float]) -> pd.DataFrame:
    """Turn the model_weights dict into a sortable DataFrame, parsing the name."""
    rows = []
    for name, w in weights.items():
        # name format like "fav_12_0002_p45all" or "up_36_0003_p45t10"
        parts = name.split("_")
        direction = parts[0] if parts else ""
        horizon = parts[1] if len(parts) > 1 else ""
        threshold = parts[2] if len(parts) > 2 else ""
        prob = parts[3] if len(parts) > 3 else ""
        rows.append({"name": name, "direction": direction, "horizon": horizon,
                     "threshold": threshold, "prob": prob, "weight": w})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    return df


def model_pkl_files(models_dir: Path) -> list[Path]:
    """All .pkl files for a given model directory (e.g. models/v23/)."""
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("*.pkl"))


def firing_history_df(decisions_df: pd.DataFrame) -> pd.DataFrame:
    """Slice firing-relevant columns from a decisions DataFrame for plotting."""
    cols = [c for c in ("timestamp", "bullish_count", "bearish_count",
                         "neutral_count", "weighted_score", "avg_raw_score")
            if c in decisions_df.columns]
    return decisions_df[cols].copy()
