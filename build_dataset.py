"""Build ML Training Dataset — dYdX backbone + all data sources.

Creates a single parquet file with:
- 5-minute grid based on dYdX candles (~240K rows, Nov 2023 → present)
- ~309 TA features from dYdX OHLCV (native features)
- ~309 TA features from Binance futures OHLCV (bnc_ prefix)
- ~150 supplementary features from 12 feature modules
- ~97 ML targets computed from dYdX close
- No look-ahead bias

Usage:
    python build_dataset.py                 # Build full dataset
    python build_dataset.py --skip-binance  # Skip Binance TA (faster, ~half features)
    python build_dataset.py --dry-run       # Show what would be built without saving

Output: processed_data/btc_training_dataset.parquet
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import pandas as pd

# Add project root to path for features package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.ta_core import compute_ta_features, compute_targets
from features.cross_exchange import build_cross_exchange_features
from features.funding import build_funding_features
from features.open_interest import build_open_interest_features
from features.positioning import build_positioning_features
from features.volatility_implied import build_implied_vol_features
from features.macro import build_macro_features
from features.sentiment import build_sentiment_features
from features.onchain import build_onchain_features
from features.defi import build_defi_features
from features.coinalyze import build_coinalyze_features
from features.dydx_trades import build_dydx_trades_features
from features.liquidations import build_liquidation_features

RAW_DIR = "raw_data"
OUTPUT_DIR = "processed_data"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
OUTPUT_PARQUET = os.path.join(OUTPUT_DIR, "btc_training_dataset.parquet")

log = logging.getLogger("build_dataset")


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(LOG_DIR, "build_dataset.log"),
                             mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)


def load_dydx_grid() -> pd.DataFrame:
    """Load dYdX candles as the master 5-minute grid."""
    path = os.path.join(RAW_DIR, "dydx_candles_5m.csv")
    df = pd.read_csv(path)

    # Convert timestamp to open_time_ms if needed
    if "open_time_ms" not in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        df["open_time_ms"] = ((ts - epoch) // pd.Timedelta(milliseconds=1)).astype(np.int64)

    # Sort and deduplicate
    df = df.sort_values("open_time_ms").drop_duplicates(subset="open_time_ms", keep="last")
    df = df.reset_index(drop=True)

    log.info("  dYdX grid: %s rows", f"{len(df):,}")
    start = pd.Timestamp(df["open_time_ms"].iloc[0], unit="ms", tz="UTC")
    end = pd.Timestamp(df["open_time_ms"].iloc[-1], unit="ms", tz="UTC")
    log.info("  Range: %s → %s", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return df


def load_binance_futures() -> pd.DataFrame:
    """Load Binance futures klines, filtered to dYdX date range."""
    path = os.path.join(RAW_DIR, "binance_futures_klines_5m.csv")
    df = pd.read_csv(path)

    # Ensure open_time_ms exists
    if "open_time_ms" not in df.columns and "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        df["open_time_ms"] = ((ts - epoch) // pd.Timedelta(milliseconds=1)).astype(np.int64)

    df = df.sort_values("open_time_ms").drop_duplicates(subset="open_time_ms", keep="last")
    df = df.reset_index(drop=True)
    log.info("  Binance futures: %s rows", f"{len(df):,}")
    return df


def build_dydx_ta_features(dydx: pd.DataFrame) -> pd.DataFrame:
    """Compute ~309 TA features from dYdX OHLCV."""
    log.info("  Computing dYdX TA features...")
    t0 = time.time()

    # Prepare dYdX DataFrame with standard OHLCV columns
    ohlcv = dydx[["open_time_ms", "open", "high", "low", "close", "volume"]].copy()

    # dYdX has usd_volume and trades — add if available
    if "trades" in dydx.columns:
        ohlcv["trades"] = dydx["trades"].values

    features = compute_ta_features(ohlcv, prefix="")
    log.info("  dYdX TA: %d features in %.1fs",
             len(features.columns) - 1, time.time() - t0)
    return features


def build_binance_ta_features(bnc: pd.DataFrame, dydx_start_ms: int,
                               dydx_end_ms: int) -> pd.DataFrame:
    """Compute ~309 TA features from Binance futures OHLCV with bnc_ prefix.

    Includes warmup period before dYdX start to avoid NaN at beginning.
    """
    log.info("  Computing Binance TA features...")
    t0 = time.time()

    # Include 2000 candles (~7 days) before dYdX start for warmup
    warmup_ms = 2000 * 5 * 60 * 1000
    start_ms = dydx_start_ms - warmup_ms
    bnc_window = bnc[bnc["open_time_ms"] >= start_ms].copy()
    log.info("  Binance window: %s rows (with warmup)", f"{len(bnc_window):,}")

    # Standard OHLCV columns
    ohlcv = bnc_window[["open_time_ms", "open", "high", "low", "close", "volume"]].copy()
    for col in ["trades", "taker_buy_volume", "quote_volume"]:
        if col in bnc_window.columns:
            ohlcv[col] = bnc_window[col].values

    features = compute_ta_features(ohlcv, prefix="bnc_")

    # Trim to dYdX date range (remove warmup period)
    features = features[features["open_time_ms"] >= dydx_start_ms].reset_index(drop=True)

    log.info("  Binance TA: %d features in %.1fs",
             len(features.columns) - 1, time.time() - t0)
    return features


def build_supplementary_features(grid: pd.DataFrame,
                                  spot_close: pd.DataFrame,
                                  base_df: pd.DataFrame) -> pd.DataFrame:
    """Build all supplementary features from 12 feature modules."""
    builders = [
        ("Cross-exchange", lambda: build_cross_exchange_features(grid, spot_close)),
        ("Funding", lambda: build_funding_features(grid)),
        ("Open Interest", lambda: build_open_interest_features(grid)),
        ("Positioning", lambda: build_positioning_features(grid)),
        ("Implied Vol", lambda: build_implied_vol_features(grid, base_df)),
        ("Macro", lambda: build_macro_features(grid, spot_close)),
        ("Sentiment", lambda: build_sentiment_features(grid)),
        ("On-chain", lambda: build_onchain_features(grid)),
        ("DeFi", lambda: build_defi_features(grid)),
        ("Coinalyze", lambda: build_coinalyze_features(grid)),
        ("dYdX Trades", lambda: build_dydx_trades_features(grid, spot_close)),
        ("Liquidations", lambda: build_liquidation_features(grid)),
    ]

    result = grid[["open_time_ms"]].copy()
    for name, builder in builders:
        t0 = time.time()
        try:
            feat_df = builder()
            new_cols = [c for c in feat_df.columns if c != "open_time_ms"]
            if new_cols:
                result = result.merge(feat_df, on="open_time_ms", how="left")
            elapsed = time.time() - t0
            log.info("  %-20s +%3d features  (%.1fs)", name, len(new_cols), elapsed)
        except FileNotFoundError as e:
            log.warning("  %-20s SKIPPED (data not downloaded yet: %s)",
                        name, os.path.basename(str(e)))
        except Exception as e:
            log.error("  %-20s FAILED: %s", name, e)
            import traceback
            traceback.print_exc()

    return result


def validate_dataset(df: pd.DataFrame, feature_cols: list,
                     target_cols: list) -> None:
    """Validate the final dataset: inf check, NaN coverage, sanity checks."""
    log.info("")
    log.info("=" * 70)
    log.info("VALIDATION REPORT")
    log.info("=" * 70)

    # Check for infinities
    inf_counts = {}
    for c in feature_cols + target_cols:
        n_inf = np.isinf(df[c].dropna()).sum()
        if n_inf > 0:
            inf_counts[c] = n_inf
    if inf_counts:
        log.warning("  %d columns have inf values — replacing with NaN", len(inf_counts))
        for c in inf_counts:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    else:
        log.info("  No inf values found.")

    # NaN coverage by prefix
    log.info("")
    log.info("  Coverage summary by prefix:")
    prefixes = {}
    for c in feature_cols:
        parts = c.split("_")
        # Group by meaningful prefix
        if c.startswith("bnc_"):
            prefix = "bnc"
        elif c.startswith("oc_") or c.startswith("net_"):
            prefix = "onchain"
        elif c.startswith("sent_"):
            prefix = "sentiment"
        elif c.startswith("defi_"):
            prefix = "defi"
        elif c.startswith("macro_"):
            prefix = "macro"
        elif c.startswith("cz_"):
            prefix = "coinalyze"
        elif c.startswith("funding_"):
            prefix = "funding"
        elif c.startswith("cot_"):
            prefix = "positioning"
        elif c.startswith("dvol_"):
            prefix = "volatility"
        elif c.startswith("dydx_"):
            prefix = "dydx_cross"
        elif c.startswith("bybit_"):
            prefix = "bybit"
        elif c.startswith("okx_"):
            prefix = "okx"
        elif c.startswith("oi_"):
            prefix = "open_interest"
        else:
            prefix = "dydx_ta"

        if prefix not in prefixes:
            prefixes[prefix] = []
        pct = 100.0 * df[c].notna().sum() / len(df)
        prefixes[prefix].append(pct)

    for prefix in sorted(prefixes):
        vals = prefixes[prefix]
        log.info("    %-20s avg=%5.1f%%  min=%5.1f%%  count=%d",
                 prefix, np.mean(vals), np.min(vals), len(vals))

    # Target coverage
    log.info("")
    log.info("  Target columns: %d", len(target_cols))
    target_coverage = [100.0 * df[c].notna().sum() / len(df) for c in target_cols]
    log.info("    Coverage: avg=%.1f%%  min=%.1f%%",
             np.mean(target_coverage), np.min(target_coverage))

    # Sanity check: dYdX close should correlate with Binance close
    if "bnc_close" in df.columns and "close" in df.columns:
        mask = df["bnc_close"].notna() & df["close"].notna()
        if mask.sum() > 100:
            corr = df.loc[mask, "close"].corr(df.loc[mask, "bnc_close"])
            log.info("  dYdX vs Binance close correlation: %.6f", corr)

    log.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Build ML training dataset")
    parser.add_argument("--skip-binance", action="store_true",
                        help="Skip Binance TA features (faster build)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be built without saving")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging()

    log.info("=" * 70)
    log.info("  BUILD ML TRAINING DATASET")
    log.info("  dYdX backbone + all data sources")
    log.info("=" * 70)
    total_t0 = time.time()

    # ---- Step 1: Load dYdX as master grid ----
    log.info("\n[1/6] Loading dYdX master grid...")
    dydx = load_dydx_grid()
    dydx_start_ms = int(dydx["open_time_ms"].iloc[0])
    dydx_end_ms = int(dydx["open_time_ms"].iloc[-1])

    # ---- Step 2: Compute dYdX TA features ----
    log.info("\n[2/6] Computing dYdX TA features (~309 features)...")
    dydx_ta = build_dydx_ta_features(dydx)

    # Start building the master DataFrame
    # Keep raw OHLCV + metadata from dYdX for cross-exchange features
    df = dydx[["open_time_ms", "open", "high", "low", "close", "volume"]].copy()
    if "timestamp" in dydx.columns:
        df["timestamp"] = dydx["timestamp"].values

    # Merge dYdX TA features
    df = df.merge(dydx_ta, on="open_time_ms", how="left")
    log.info("  After dYdX TA: %d columns", len(df.columns))

    # ---- Step 3: Compute Binance TA features ----
    if not args.skip_binance:
        log.info("\n[3/6] Computing Binance TA features (~309 features)...")
        bnc_path = os.path.join(RAW_DIR, "binance_futures_klines_5m.csv")
        if os.path.exists(bnc_path):
            bnc = load_binance_futures()
            bnc_ta = build_binance_ta_features(bnc, dydx_start_ms, dydx_end_ms)

            # Also keep raw Binance close for cross-venue features
            bnc_close_raw = bnc[["open_time_ms", "close"]].copy()
            bnc_close_raw = bnc_close_raw.rename(columns={"close": "bnc_close"})
            bnc_in_range = bnc_close_raw[
                (bnc_close_raw["open_time_ms"] >= dydx_start_ms) &
                (bnc_close_raw["open_time_ms"] <= dydx_end_ms)
            ]
            df = df.merge(bnc_in_range, on="open_time_ms", how="left")

            # Merge Binance TA features
            df = df.merge(bnc_ta, on="open_time_ms", how="left")
            log.info("  After Binance TA: %d columns", len(df.columns))
        else:
            log.warning("  Binance futures file not found — skipping")
    else:
        log.info("\n[3/6] Skipping Binance TA features (--skip-binance)")

    # ---- Step 4: Compute supplementary features ----
    log.info("\n[4/6] Computing supplementary features...")
    grid = df[["open_time_ms"]].copy()

    # spot_close: use Binance spot close for cross-exchange features
    spot_path = os.path.join(RAW_DIR, "binance_spot_klines_5m.csv")
    if os.path.exists(spot_path):
        spot = pd.read_csv(spot_path)
        spot_close = spot[["open_time_ms", "close"]].copy()
    else:
        # Fallback: use dYdX close as spot proxy
        log.warning("  No Binance spot data — using dYdX close as proxy")
        spot_close = df[["open_time_ms", "close"]].copy()

    # base_df for implied vol (needs garch_vol_fast, realized_vol_288)
    base_df = df.copy()

    supp = build_supplementary_features(grid, spot_close, base_df)
    supp_cols = [c for c in supp.columns if c != "open_time_ms"]
    df = df.merge(supp, on="open_time_ms", how="left")
    log.info("  After supplementary: %d columns (+%d new)", len(df.columns), len(supp_cols))

    # ---- Step 5: Compute targets ----
    log.info("\n[5/6] Computing ML targets (~97 targets)...")
    targets = compute_targets(df)
    target_cols = [c for c in targets.columns if c != "open_time_ms"]
    df = df.merge(targets, on="open_time_ms", how="left")
    log.info("  After targets: %d columns (+%d targets)", len(df.columns), len(target_cols))

    # ---- Step 6: Validate and save ----
    log.info("\n[6/6] Validating and saving...")

    # Identify feature vs target vs metadata columns
    meta_cols = {"open_time_ms", "timestamp", "open", "high", "low", "close",
                 "volume", "bnc_close"}
    target_col_names = sorted(c for c in df.columns if c.startswith("target_"))
    feature_col_names = sorted(c for c in df.columns
                                if c not in meta_cols and c not in target_col_names)

    # Drop dead features: constant columns and >90% NaN
    dead_features = []
    for c in feature_col_names:
        nan_pct = df[c].isna().mean()
        nunique = df[c].nunique()
        if nunique <= 1 or nan_pct > 0.90:
            dead_features.append(c)
    if dead_features:
        log.info("  Dropping %d dead features (constant or >90%% NaN):", len(dead_features))
        for c in dead_features:
            log.info("    %s", c)
        df = df.drop(columns=dead_features)
        feature_col_names = [c for c in feature_col_names if c not in dead_features]

    log.info("")
    log.info("  Dataset shape: %s rows x %d columns", f"{len(df):,}", len(df.columns))
    log.info("  Features: %d", len(feature_col_names))
    log.info("  Targets: %d", len(target_col_names))
    log.info("  Metadata: %d", len(meta_cols & set(df.columns)))

    # Validate
    validate_dataset(df, feature_col_names, target_col_names)

    if not args.dry_run:
        # Cast feature columns to float32 for space efficiency
        for c in feature_col_names:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)

        log.info("\nSaving to %s...", OUTPUT_PARQUET)
        df.to_parquet(OUTPUT_PARQUET, index=False)
        file_size_gb = os.path.getsize(OUTPUT_PARQUET) / (1024 ** 3)
        log.info("  Saved: %s rows x %d cols, %.2f GB",
                 f"{len(df):,}", len(df.columns), file_size_gb)
    else:
        log.info("\n  [DRY RUN] Would save %s rows x %d cols to %s",
                 f"{len(df):,}", len(df.columns), OUTPUT_PARQUET)

    elapsed = time.time() - total_t0
    log.info("\nTotal time: %.1fs", elapsed)
    log.info("=" * 70)

    return df


if __name__ == "__main__":
    main()
