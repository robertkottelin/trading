"""Feature importance analysis across trained LightGBM models.

Loads all production LightGBM models from a model directory, extracts
feature importances (gain + split), aggregates across models, and
identifies low-importance features as candidates for removal.

Usage:
    python -m model_training.analyze_feature_importance
    python -m model_training.analyze_feature_importance --model-dir models/v2_all
    python -m model_training.analyze_feature_importance --top 50
    python -m model_training.analyze_feature_importance --save
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_models(model_dir: str) -> list[dict]:
    """Load all LightGBM .pkl models and their configs from a model directory."""
    model_dir = Path(model_dir)

    # Find production config
    config_files = list(model_dir.glob("production_config*.json"))
    if not config_files:
        print(f"ERROR: No production_config*.json in {model_dir}")
        sys.exit(1)

    with open(config_files[0]) as f:
        config = json.load(f)

    models = []
    for entry in config.get("models", []):
        lgb_file = entry.get("lgb_file")
        if not lgb_file:
            continue
        lgb_path = model_dir / lgb_file
        if not lgb_path.exists():
            print(f"  WARNING: {lgb_path} not found, skipping")
            continue
        with open(lgb_path, "rb") as mf:
            model = pickle.load(mf)

        models.append({
            "name": entry["name"],
            "model": model,
            "features": entry.get("features", []),
            "target": entry.get("target", ""),
        })

    return models


def extract_importances(models: list[dict]) -> pd.DataFrame:
    """Extract and aggregate feature importances across all models."""
    gain_counts = defaultdict(list)
    split_counts = defaultdict(list)

    for info in models:
        model = info["model"]
        features = info["features"]

        # LightGBM Booster
        if hasattr(model, "feature_importance"):
            gain = model.feature_importance(importance_type="gain")
            split = model.feature_importance(importance_type="split")
            names = model.feature_name()
        elif hasattr(model, "booster_"):
            # sklearn-wrapped LightGBM
            gain = model.booster_.feature_importance(importance_type="gain")
            split = model.booster_.feature_importance(importance_type="split")
            names = model.booster_.feature_name()
        else:
            print(f"  WARNING: {info['name']} — unknown model type, skipping")
            continue

        for i, name in enumerate(names):
            if i < len(gain):
                gain_counts[name].append(gain[i])
            if i < len(split):
                split_counts[name].append(split[i])

    # Build summary DataFrame
    all_features = sorted(set(gain_counts.keys()) | set(split_counts.keys()))
    rows = []
    for feat in all_features:
        gains = gain_counts.get(feat, [0])
        splits = split_counts.get(feat, [0])
        rows.append({
            "feature": feat,
            "mean_gain": np.mean(gains),
            "median_gain": np.median(gains),
            "std_gain": np.std(gains),
            "max_gain": np.max(gains),
            "mean_splits": np.mean(splits),
            "models_using": len(gains),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("mean_gain", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["cumulative_gain_pct"] = (
        df["mean_gain"].cumsum() / df["mean_gain"].sum() * 100
    )
    return df


def print_report(df: pd.DataFrame, n_models: int, top_n: int = 50):
    """Print a human-readable feature importance report."""
    total = len(df)
    total_gain = df["mean_gain"].sum()

    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*80}")
    print(f"  Models analyzed: {n_models}")
    print(f"  Total features:  {total}")
    print(f"  Total gain:      {total_gain:,.0f}")

    # Top features
    print(f"\n--- TOP {top_n} FEATURES (by mean gain) ---")
    print(f"{'Rank':>4}  {'Feature':<45} {'Mean Gain':>10} {'Cum%':>6} {'Models':>6}")
    print("-" * 80)
    for _, row in df.head(top_n).iterrows():
        print(
            f"{int(row['rank']):>4}  {row['feature']:<45} "
            f"{row['mean_gain']:>10.1f} {row['cumulative_gain_pct']:>5.1f}% "
            f"{int(row['models_using']):>6}"
        )

    # Coverage analysis
    for pct in [80, 90, 95, 99]:
        n = (df["cumulative_gain_pct"] <= pct).sum() + 1
        print(f"\n  Features needed for {pct}% of total gain: {n}/{total}")

    # Bottom features (candidates for removal)
    bottom = df.tail(min(30, total))
    print(f"\n--- BOTTOM {len(bottom)} FEATURES (removal candidates) ---")
    print(f"{'Rank':>4}  {'Feature':<45} {'Mean Gain':>10} {'Models':>6}")
    print("-" * 80)
    for _, row in bottom.iterrows():
        print(
            f"{int(row['rank']):>4}  {row['feature']:<45} "
            f"{row['mean_gain']:>10.1f} {int(row['models_using']):>6}"
        )

    # Zero-importance features
    zero = df[df["mean_gain"] == 0]
    if len(zero) > 0:
        print(f"\n  ZERO-IMPORTANCE features: {len(zero)}")
        for _, row in zero.iterrows():
            print(f"    - {row['feature']}")

    # Feature category breakdown
    print(f"\n--- IMPORTANCE BY FEATURE CATEGORY ---")
    categories = {}
    for _, row in df.iterrows():
        feat = row["feature"]
        # Extract category prefix (e.g., "ta_", "sent_", "bybit_", etc.)
        parts = feat.split("_")
        if len(parts) >= 2:
            cat = parts[0]
        else:
            cat = "other"
        if cat not in categories:
            categories[cat] = {"gain": 0, "count": 0}
        categories[cat]["gain"] += row["mean_gain"]
        categories[cat]["count"] += 1

    cat_df = pd.DataFrame([
        {"category": k, "total_gain": v["gain"], "n_features": v["count"],
         "avg_gain": v["gain"] / v["count"] if v["count"] > 0 else 0}
        for k, v in categories.items()
    ]).sort_values("total_gain", ascending=False)

    print(f"{'Category':<15} {'Total Gain':>12} {'# Features':>10} {'Avg Gain':>10} {'Gain%':>7}")
    print("-" * 60)
    for _, row in cat_df.iterrows():
        pct = row["total_gain"] / total_gain * 100 if total_gain > 0 else 0
        print(
            f"{row['category']:<15} {row['total_gain']:>12.0f} "
            f"{int(row['n_features']):>10} {row['avg_gain']:>10.1f} {pct:>6.1f}%"
        )

    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze feature importance across trained LightGBM models"
    )
    parser.add_argument("--model-dir", default="models/v2_all",
                        help="Directory containing trained models (default: models/v2_all)")
    parser.add_argument("--top", type=int, default=50,
                        help="Number of top features to display (default: 50)")
    parser.add_argument("--save", action="store_true",
                        help="Save full importance table to CSV")
    args = parser.parse_args()

    print(f"Loading models from {args.model_dir}...")
    models = load_models(args.model_dir)
    if not models:
        print("ERROR: No models loaded")
        sys.exit(1)
    print(f"  Loaded {len(models)} models")

    print("Extracting feature importances...")
    df = extract_importances(models)

    print_report(df, len(models), top_n=args.top)

    if args.save:
        out_path = os.path.join(args.model_dir, "feature_importance.csv")
        df.to_csv(out_path, index=False)
        print(f"\nFull table saved to: {out_path}")


if __name__ == "__main__":
    main()
