"""
Bitcoin ML Feature Engineering Pipeline
========================================
Downloads 2 BTC data sources (spot klines, futures klines) and builds an
ML-ready feature table with ~100+ features per 5-minute candle.

Pipeline: Download -> Feature Engineering -> Output

Data sources (all free, no API keys):
  1. Spot OHLCV klines (Binance) — full history from 2017-08-17
  2. Futures OHLCV klines (Binance) — full history from 2019-09-08

Requirements:
  pip install requests pandas ta

Usage:
  python btc_indicators.py                # Download + build features
  python btc_indicators.py --download-only # Download raw CSVs only

Output files:
  btc_data/raw_spot_klines.csv       — raw spot OHLCV (~892K rows)
  btc_data/raw_futures_klines.csv    — raw futures OHLCV (~677K rows)
  btc_data/btc_features_5m.csv       — ML-ready feature table (~892K rows x 100+ cols)
  btc_data/btc_indicators.log        — full debug log
"""

import requests
import pandas as pd
import numpy as np
import ta
import time
import os
import logging
import traceback
import argparse
from datetime import datetime, timezone


# --- Configuration -----------------------------------------------------------

OUTPUT_DIR = "btc_data"
LOG_FILE = os.path.join(OUTPUT_DIR, "btc_indicators.log")
BINANCE_SPOT_TICKER = "BTCUSDT"
BINANCE_FUTURES_TICKER = "BTCUSDT"

# 5m timeframe
TIMEFRAME = "5m"
TIMEFRAME_MS = 5 * 60 * 1000  # 5 minutes in milliseconds

# Earliest data availability (approximate Binance listing dates)
SPOT_START_MS = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)
FUTURES_START_MS = int(datetime(2019, 9, 8, tzinfo=timezone.utc).timestamp() * 1000)

# Pagination limits (max per request)
KLINE_LIMIT = 1000
REQUEST_DELAY = 0.3  # seconds between paginated requests

# Retry settings for transient HTTP errors
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, doubled each retry


# --- Logging Setup -----------------------------------------------------------

log = logging.getLogger("btc_indicators")


def setup_logging():
    """Configure logging to both console and a log file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.setLevel(logging.DEBUG)

    # File handler — verbose (DEBUG)
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler — concise (INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    log.addHandler(fh)
    log.addHandler(ch)


# --- Helpers -----------------------------------------------------------------

def save_csv(df, filename, description):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    log.info("  Saved %s -> %s (%s rows)", description, path, f"{len(df):,}")


def timestamp_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def ms_to_str(ms):
    """Convert millisecond timestamp to readable string."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _http_get(url, params, timeout=30):
    """
    Perform a GET request with retries on transient errors.
    Returns the Response object, or raises on persistent failure.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.debug("HTTP GET %s  params=%s  (attempt %d/%d)", url, params, attempt, MAX_RETRIES)
            resp = requests.get(url, params=params, timeout=timeout)
            log.debug("HTTP %d  %d bytes  %s", resp.status_code, len(resp.content), url)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", RETRY_BACKOFF * attempt))
                log.warning("Rate-limited (429) on %s — sleeping %ds", url, retry_after)
                time.sleep(retry_after)
                continue

            if resp.status_code >= 500:
                log.warning("Server error %d on %s — retrying in %ds", resp.status_code, url, RETRY_BACKOFF * attempt)
                time.sleep(RETRY_BACKOFF * attempt)
                continue

            resp.raise_for_status()
            return resp

        except requests.exceptions.ConnectionError as e:
            last_exc = e
            log.warning("Connection error on %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, e)
            time.sleep(RETRY_BACKOFF * attempt)
        except requests.exceptions.Timeout as e:
            last_exc = e
            log.warning("Timeout on %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, e)
            time.sleep(RETRY_BACKOFF * attempt)
        except requests.exceptions.HTTPError:
            raise  # 4xx client errors — don't retry

    raise requests.exceptions.ConnectionError(
        f"Failed after {MAX_RETRIES} retries on {url}: {last_exc}"
    )


def validate_dataframe(df, name, expected_cols=None):
    """Log data-quality diagnostics for a DataFrame."""
    if df.empty:
        log.warning("VALIDATION %s: DataFrame is EMPTY", name)
        return

    log.info("  [check] %s: %s rows, %s cols", name, f"{len(df):,}", len(df.columns))

    if "timestamp" in df.columns:
        log.info("  [check] %s: date range %s .. %s", name, df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    nulls = df.isnull().sum()
    cols_with_nulls = nulls[nulls > 0]
    if not cols_with_nulls.empty:
        log.warning("  [check] %s: columns with nulls: %s", name, cols_with_nulls.to_dict())

    if expected_cols:
        missing = set(expected_cols) - set(df.columns)
        if missing:
            log.error("  [check] %s: MISSING expected columns: %s", name, missing)


# --- Paginated Download Helpers ----------------------------------------------

def paginate_klines(url, symbol, interval, start_ms, end_ms=None,
                    limit=KLINE_LIMIT, delay=REQUEST_DELAY):
    """
    Download ALL klines from a Binance klines endpoint by paginating forward
    from start_ms to end_ms (default: now).
    """
    if end_ms is None:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    log.debug("paginate_klines: %s %s %s  range %s -> %s", url, symbol, interval, ms_to_str(start_ms), ms_to_str(end_ms))

    all_klines = []
    current = start_ms
    request_count = 0

    while current < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current,
            "endTime": end_ms,
            "limit": limit,
        }

        try:
            resp = _http_get(url, params)
            batch = resp.json()
        except Exception as e:
            log.error("paginate_klines FAILED at offset %s after %d requests (%s rows fetched): %s",
                      ms_to_str(current), request_count, f"{len(all_klines):,}", e)
            log.debug(traceback.format_exc())
            break

        request_count += 1

        if not batch:
            log.debug("paginate_klines: empty response at %s — end of data", ms_to_str(current))
            break

        all_klines.extend(batch)
        current = batch[-1][0] + TIMEFRAME_MS

        if len(batch) < limit:
            break

        if len(all_klines) % 10000 < limit:
            log.info("    ... %s candles (up to %s)", f"{len(all_klines):,}", ms_to_str(batch[-1][0]))

        time.sleep(delay)

    log.info("    Total: %s candles  (%d requests)", f"{len(all_klines):,}", request_count)
    return all_klines


# --- Kline Cache -------------------------------------------------------------
# Avoids re-downloading klines if called multiple times.

_kline_cache = {}


def get_cached_klines(url, symbol, interval, start_ms, label=""):
    """Download klines with caching."""
    key = (url, symbol, interval, start_ms)
    if key in _kline_cache:
        log.info("  Using cached %s klines (%s candles)", label, f"{len(_kline_cache[key]):,}")
        return _kline_cache[key]
    klines = paginate_klines(url, symbol, interval, start_ms)
    _kline_cache[key] = klines
    return klines


# --- Download Functions ------------------------------------------------------

def download_spot_klines():
    """Download all spot BTCUSDT 5m klines. Returns raw list of klines."""
    log.info("\n[1/2] Spot Klines (Binance) — full history")
    log.info("  Downloading all %s spot klines...", TIMEFRAME)

    klines = get_cached_klines(
        url="https://api.binance.com/api/v3/klines",
        symbol=BINANCE_SPOT_TICKER,
        interval=TIMEFRAME,
        start_ms=SPOT_START_MS,
        label="spot",
    )

    if not klines:
        log.error("  No spot klines returned")
        return []

    # Save raw CSV
    rows = []
    for k in klines:
        rows.append({
            "open_time_ms": k[0],
            "timestamp": ms_to_str(k[0]) + " UTC",
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "quote_volume": float(k[7]),
            "trades": int(k[8]),
            "taker_buy_volume": float(k[9]),
            "taker_buy_quote_volume": float(k[10]),
        })

    df = pd.DataFrame(rows)
    save_csv(df, "raw_spot_klines.csv", "raw spot klines")
    validate_dataframe(df, "raw_spot_klines", expected_cols=["timestamp", "close", "volume"])

    return klines


def download_futures_klines():
    """Download all futures BTCUSDT 5m klines. Returns raw list of klines."""
    log.info("\n[2/2] Futures Klines (Binance) — full history")
    log.info("  Downloading all %s futures klines...", TIMEFRAME)

    klines = get_cached_klines(
        url="https://fapi.binance.com/fapi/v1/klines",
        symbol=BINANCE_FUTURES_TICKER,
        interval=TIMEFRAME,
        start_ms=FUTURES_START_MS,
        label="futures",
    )

    if not klines:
        log.error("  No futures klines returned")
        return []

    # Save raw CSV
    rows = []
    for k in klines:
        rows.append({
            "open_time_ms": k[0],
            "timestamp": ms_to_str(k[0]) + " UTC",
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "quote_volume": float(k[7]),
            "trades": int(k[8]),
            "taker_buy_volume": float(k[9]),
            "taker_buy_quote_volume": float(k[10]),
        })

    df = pd.DataFrame(rows)
    save_csv(df, "raw_futures_klines.csv", "raw futures klines")
    validate_dataframe(df, "raw_futures_klines", expected_cols=["timestamp", "close", "volume"])

    return klines


# --- Feature Engineering -----------------------------------------------------

def build_features(spot_klines, futures_klines):
    """
    Build ML-ready feature table from raw data.

    Steps:
      1. Build spot DataFrame with OHLCV + taker volumes
      2. Add ~80 TA indicators via ta library
      3. Add CVD features (delta, cumulative delta, delta SMA)
      4. Add custom price features (returns, distances, time)
      5. Merge futures data (basis, volume ratio)
      6. Add ML target columns (future returns/direction)

    Returns: DataFrame with ~100+ columns per 5m candle.
    """
    log.info("\nBuilding ML feature table...")

    # --- Step 1: Build spot DataFrame ---
    log.info("  Step 1: Building spot DataFrame...")
    rows = []
    for k in spot_klines:
        total_vol = float(k[5])
        taker_buy = float(k[9])
        taker_sell = total_vol - taker_buy
        rows.append({
            "open_time_ms": k[0],
            "timestamp": ms_to_str(k[0]) + " UTC",
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": total_vol,
            "quote_volume": float(k[7]),
            "trades": int(k[8]),
            "taker_buy_vol": taker_buy,
            "taker_sell_vol": taker_sell,
        })

    df = pd.DataFrame(rows)
    log.info("    Spot DataFrame: %s rows", f"{len(df):,}")

    # --- Step 2: Add TA indicators ---
    log.info("  Step 2: Adding TA indicators (~80 features)...")
    cols_before = len(df.columns)
    df = ta.add_all_ta_features(
        df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=False,
    )
    cols_added = len(df.columns) - cols_before
    log.info("    Added %d TA indicator columns", cols_added)

    # --- Step 3: CVD features ---
    log.info("  Step 3: Adding CVD features...")
    df["delta_btc"] = df["taker_buy_vol"] - df["taker_sell_vol"]
    df["cumulative_delta_btc"] = df["delta_btc"].cumsum()
    df["delta_sma_14"] = df["delta_btc"].rolling(window=14, min_periods=1).mean()

    # --- Step 4: Custom price features ---
    log.info("  Step 4: Adding custom price features...")

    # Returns at various horizons
    df["return_1"] = df["close"].pct_change(1)
    df["return_6"] = df["close"].pct_change(6)      # 30min
    df["return_12"] = df["close"].pct_change(12)     # 1h
    df["return_288"] = df["close"].pct_change(288)   # 24h

    # Log return
    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

    # Distance from moving averages
    sma50 = df["close"].rolling(window=50, min_periods=1).mean()
    sma200 = df["close"].rolling(window=200, min_periods=1).mean()
    df["price_sma50_dist"] = (df["close"] - sma50) / sma50
    df["price_sma200_dist"] = (df["close"] - sma200) / sma200

    # Candle range
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]

    # Time features
    dt_series = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    df["hour_of_day"] = dt_series.dt.hour
    df["day_of_week"] = dt_series.dt.dayofweek

    # --- Step 5: Merge futures data ---
    log.info("  Step 5: Merging futures data...")
    if futures_klines:
        futures_rows = []
        for k in futures_klines:
            futures_rows.append({
                "open_time_ms": k[0],
                "futures_close": float(k[4]),
                "futures_volume_btc": float(k[5]),
                "futures_trades": int(k[8]),
            })
        df_futures = pd.DataFrame(futures_rows)

        df = df.merge(df_futures, on="open_time_ms", how="left")

        # Derived futures features
        df["basis"] = df["futures_close"] - df["close"]
        df["basis_pct"] = (df["basis"] / df["close"]) * 100
        df["futures_to_spot_vol_ratio"] = df["futures_volume_btc"] / df["volume"]
        # Handle division by zero
        df["futures_to_spot_vol_ratio"] = df["futures_to_spot_vol_ratio"].replace(
            [np.inf, -np.inf], np.nan
        )

        log.info("    Futures columns merged (NaN before %s)", ms_to_str(FUTURES_START_MS))
    else:
        log.warning("    No futures data — skipping futures features")
        for col in ["futures_close", "basis", "basis_pct",
                     "futures_volume_btc", "futures_to_spot_vol_ratio", "futures_trades"]:
            df[col] = np.nan

    # --- Step 6: ML target columns ---
    log.info("  Step 6: Adding ML target columns...")

    close = df["close"]

    # --- Direction targets (binary: 1=up, 0=down) at multiple horizons ---
    for n, label in [(1, "5m"), (6, "30m"), (12, "1h"), (36, "3h"), (288, "24h")]:
        fwd_return = close.pct_change(n).shift(-n)
        df[f"target_return_{n}"] = fwd_return
        direction = (fwd_return > 0).astype(float)
        direction[fwd_return.isna()] = np.nan
        df[f"target_direction_{n}"] = direction

    # --- Trend targets (is price sustaining a move?) ---
    # Fraction of next N closes above current close (0.0 = all below, 1.0 = all above)
    for n, label in [(12, "1h"), (36, "3h"), (288, "24h")]:
        future_above = pd.DataFrame({
            f"s{i}": (close.shift(-i) > close).astype(float) for i in range(1, n + 1)
        })
        trend_strength = future_above.mean(axis=1)
        trend_strength[close.shift(-n).isna()] = np.nan
        df[f"target_trend_strength_{n}"] = trend_strength
        # Binary: sustained trend = >60% of candles in one direction
        trend_up = (trend_strength > 0.6).astype(float)
        trend_up[trend_strength.isna()] = np.nan
        df[f"target_trend_up_{n}"] = trend_up
        trend_down = (trend_strength < 0.4).astype(float)
        trend_down[trend_strength.isna()] = np.nan
        df[f"target_trend_down_{n}"] = trend_down

    # --- Volatility targets ---
    # Forward realized volatility (std of returns over next N candles)
    returns_1 = close.pct_change(1)
    for n, label in [(12, "1h"), (36, "3h"), (288, "24h")]:
        fwd_vol = returns_1.shift(-n).rolling(window=n).std().shift(n - 1).shift(-n)
        # Simpler: compute directly from future window
        fwd_returns = pd.DataFrame({
            f"r{i}": close.pct_change(1).shift(-i) for i in range(1, n + 1)
        })
        fwd_vol = fwd_returns.std(axis=1)
        fwd_vol[close.shift(-n).isna()] = np.nan
        df[f"target_volatility_{n}"] = fwd_vol

    # Forward max drawdown and max runup (worst dip / best rally in next N candles)
    for n, label in [(12, "1h"), (36, "3h")]:
        future_closes = pd.DataFrame({
            f"c{i}": close.shift(-i) for i in range(1, n + 1)
        })
        future_max = future_closes.max(axis=1)
        future_min = future_closes.min(axis=1)
        max_runup = (future_max - close) / close
        max_drawdown = (close - future_min) / close
        max_runup[close.shift(-n).isna()] = np.nan
        max_drawdown[close.shift(-n).isna()] = np.nan
        df[f"target_max_runup_{n}"] = max_runup
        df[f"target_max_drawdown_{n}"] = max_drawdown

    # Binary big move target: does price move >0.5% in next 12 candles (1h)?
    df["target_big_move_12"] = ((df["target_max_runup_12"] > 0.005) |
                                 (df["target_max_drawdown_12"] > 0.005)).astype(float)
    df.loc[df["target_max_runup_12"].isna(), "target_big_move_12"] = np.nan

    target_cols = [c for c in df.columns if c.startswith("target_")]
    log.info("    Added %d target columns: %s", len(target_cols), ", ".join(target_cols))
    log.info("  Feature table complete: %s rows x %d columns", f"{len(df):,}", len(df.columns))

    return df


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download BTC data and build ML feature table"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download raw CSVs, skip feature engineering",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Rebuild features from existing raw CSVs (skip download)",
    )
    args = parser.parse_args()

    setup_logging()

    log.info("=" * 60)
    log.info("  BTC ML Feature Engineering Pipeline")
    log.info("  Timeframe: %s", TIMEFRAME)
    log.info("  Time: %s", timestamp_now())
    if args.download_only:
        log.info("  Mode: download-only (no feature engineering)")
    if args.features_only:
        log.info("  Mode: features-only (rebuild from existing raw CSVs)")
    log.info("  Log file: %s", LOG_FILE)
    log.info("=" * 60)

    results = {}
    total_start = time.time()

    spot_klines = []
    futures_klines = []

    if args.features_only:
        # --- Load from existing raw CSVs ---
        log.info("\n--- Loading from existing raw CSVs ---")
        t0 = time.time()
        try:
            df_spot = pd.read_csv(os.path.join(OUTPUT_DIR, "raw_spot_klines.csv"))
            log.info("  Loaded raw_spot_klines.csv: %s rows", f"{len(df_spot):,}")
            # Convert back to kline list format
            for _, row in df_spot.iterrows():
                spot_klines.append([
                    int(row["open_time_ms"]), str(row["open"]), str(row["high"]),
                    str(row["low"]), str(row["close"]), str(row["volume"]),
                    0, str(row["quote_volume"]), int(row["trades"]),
                    str(row["taker_buy_volume"]), str(row["taker_buy_quote_volume"]), 0,
                ])
            results["load_spot"] = {"status": "ok", "elapsed": time.time() - t0}
        except Exception as e:
            results["load_spot"] = {"status": "error", "error": str(e), "elapsed": time.time() - t0}
            log.error("Failed to load raw_spot_klines.csv: %s", e)

        t0 = time.time()
        try:
            df_fut = pd.read_csv(os.path.join(OUTPUT_DIR, "raw_futures_klines.csv"))
            log.info("  Loaded raw_futures_klines.csv: %s rows", f"{len(df_fut):,}")
            for _, row in df_fut.iterrows():
                futures_klines.append([
                    int(row["open_time_ms"]), str(row["open"]), str(row["high"]),
                    str(row["low"]), str(row["close"]), str(row["volume"]),
                    0, str(row["quote_volume"]), int(row["trades"]),
                    str(row["taker_buy_volume"]), str(row["taker_buy_quote_volume"]), 0,
                ])
            results["load_futures"] = {"status": "ok", "elapsed": time.time() - t0}
        except Exception as e:
            results["load_futures"] = {"status": "error", "error": str(e), "elapsed": time.time() - t0}
            log.error("Failed to load raw_futures_klines.csv: %s", e)
    else:
        # --- Phase 1: Download ---
        log.info("\n--- Phase 1: Download ---")

        # Download spot klines
        t0 = time.time()
        try:
            spot_klines = download_spot_klines()
            results["spot_klines"] = {"status": "ok", "elapsed": time.time() - t0}
        except Exception as e:
            results["spot_klines"] = {"status": "error", "error": str(e), "elapsed": time.time() - t0}
            log.error("[1/2] FAILED: %s", e)
            log.debug(traceback.format_exc())

        # Download futures klines
        t0 = time.time()
        try:
            futures_klines = download_futures_klines()
            results["futures_klines"] = {"status": "ok", "elapsed": time.time() - t0}
        except Exception as e:
            results["futures_klines"] = {"status": "error", "error": str(e), "elapsed": time.time() - t0}
            log.error("[2/2] FAILED: %s", e)
            log.debug(traceback.format_exc())

    # --- Phase 2: Feature Engineering ---
    if not args.download_only:
        log.info("\n--- Phase 2: Feature Engineering ---")

        if not spot_klines:
            log.error("Cannot build features — no spot kline data")
        else:
            t0 = time.time()
            try:
                df = build_features(spot_klines, futures_klines)
                save_csv(df, "btc_features_5m.csv", "ML feature table")
                validate_dataframe(
                    df, "btc_features_5m",
                    expected_cols=[
                        "timestamp", "close", "volume",
                        "delta_btc", "cumulative_delta_btc",
                        "return_1", "log_return_1",
                        "target_return_1", "target_direction_1",
                    ],
                )
                results["features"] = {"status": "ok", "elapsed": time.time() - t0}
            except Exception as e:
                results["features"] = {"status": "error", "error": str(e), "elapsed": time.time() - t0}
                log.error("Feature engineering FAILED: %s", e)
                log.debug(traceback.format_exc())
    else:
        log.info("\n--- Phase 2: Skipped (--download-only) ---")

    # --- Run Report ---
    total_elapsed = time.time() - total_start

    log.info("")
    log.info("=" * 60)
    log.info("  RUN REPORT")
    log.info("=" * 60)

    any_error = False
    for name, r in results.items():
        elapsed = f"{r['elapsed']:.1f}s"
        if r["status"] == "ok":
            log.info("  [OK]    %-20s (%s)", name, elapsed)
        else:
            any_error = True
            log.error("  [FAIL]  %-20s (%s)  -- %s", name, elapsed, r["error"])

    log.info("")
    log.info("  Total time: %.1fs", total_elapsed)
    log.info("  Output directory: ./%s/", OUTPUT_DIR)

    csv_files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith('.csv'))
    if csv_files:
        log.info("  CSV files:")
        for f in csv_files:
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            if size >= 1_000_000:
                log.info("    %s  (%.1f MB)", f, size / 1_000_000)
            else:
                log.info("    %s  (%.1f KB)", f, size / 1_000)

    log.info("")
    log.info("  Full log: %s", LOG_FILE)

    if any_error:
        log.info("  Some steps failed — check log for details.")
    else:
        log.info("  All steps completed successfully.")

    log.info("=" * 60)


if __name__ == "__main__":
    main()
