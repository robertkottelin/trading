"""
Bitcoin ML Feature Engineering Pipeline — v3
=============================================
Downloads BTC data (spot + futures klines) and builds an ML-ready feature table.

Key improvements over v2:
  - Advanced volatility estimators: Parkinson, Garman-Klass, EWMA, Yang-Zhang
  - Intraday volatility patterns (hour-of-day seasonality)
  - Historical threshold hit rates (forward-looking removed, backward proxy)
  - ATR features, conditional volatility, mean-reversion intensity
  - Risk-reward targets: "will max_runup > X * max_drawdown?"

Usage:
  python btc_indicators.py                  # Download + build features
  python btc_indicators.py --download-only  # Download raw CSVs only
  python btc_indicators.py --features-only  # Rebuild from cached CSVs
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
TIMEFRAME = "5m"
TIMEFRAME_MS = 5 * 60 * 1000
SPOT_START_MS = int(datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)
FUTURES_START_MS = int(datetime(2019, 9, 8, tzinfo=timezone.utc).timestamp() * 1000)
KLINE_LIMIT = 1000
REQUEST_DELAY = 0.3
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

log = logging.getLogger("btc_indicators")


def setup_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-7s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(ch)


def save_csv(df, filename, description):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    log.info("  Saved %s -> %s (%s rows)", description, path, f"{len(df):,}")


def ms_to_str(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _http_get(url, params, timeout=30):
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", RETRY_BACKOFF * attempt))
                log.warning("Rate-limited (429) — sleeping %ds", retry_after)
                time.sleep(retry_after)
                continue
            if resp.status_code >= 500:
                time.sleep(RETRY_BACKOFF * attempt)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError as e:
            last_exc = e
            time.sleep(RETRY_BACKOFF * attempt)
        except requests.exceptions.Timeout as e:
            last_exc = e
            time.sleep(RETRY_BACKOFF * attempt)
        except requests.exceptions.HTTPError:
            raise
    raise requests.exceptions.ConnectionError(f"Failed after {MAX_RETRIES} retries: {last_exc}")


def paginate_klines(url, symbol, interval, start_ms, end_ms=None, limit=KLINE_LIMIT, delay=REQUEST_DELAY):
    if end_ms is None:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    all_klines = []
    current = start_ms
    request_count = 0
    while current < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": current, "endTime": end_ms, "limit": limit}
        try:
            resp = _http_get(url, params)
            batch = resp.json()
        except Exception as e:
            log.error("paginate_klines FAILED at %s after %d requests: %s", ms_to_str(current), request_count, e)
            break
        request_count += 1
        if not batch:
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


_kline_cache = {}


def get_cached_klines(url, symbol, interval, start_ms, label=""):
    key = (url, symbol, interval, start_ms)
    if key in _kline_cache:
        log.info("  Using cached %s klines (%s candles)", label, f"{len(_kline_cache[key]):,}")
        return _kline_cache[key]
    klines = paginate_klines(url, symbol, interval, start_ms)
    _kline_cache[key] = klines
    return klines


def download_spot_klines():
    log.info("\n[1/2] Spot Klines — full history")
    klines = get_cached_klines(
        url="https://api.binance.com/api/v3/klines",
        symbol=BINANCE_SPOT_TICKER, interval=TIMEFRAME,
        start_ms=SPOT_START_MS, label="spot",
    )
    if not klines:
        log.error("  No spot klines returned")
        return []
    rows = []
    for k in klines:
        rows.append({
            "open_time_ms": k[0], "timestamp": ms_to_str(k[0]) + " UTC",
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
            "close": float(k[4]), "volume": float(k[5]), "quote_volume": float(k[7]),
            "trades": int(k[8]), "taker_buy_volume": float(k[9]),
            "taker_buy_quote_volume": float(k[10]),
        })
    df = pd.DataFrame(rows)
    save_csv(df, "raw_spot_klines.csv", "raw spot klines")
    return klines


def download_futures_klines():
    log.info("\n[2/2] Futures Klines — full history")
    klines = get_cached_klines(
        url="https://fapi.binance.com/fapi/v1/klines",
        symbol=BINANCE_FUTURES_TICKER, interval=TIMEFRAME,
        start_ms=FUTURES_START_MS, label="futures",
    )
    if not klines:
        log.error("  No futures klines returned")
        return []
    rows = []
    for k in klines:
        rows.append({
            "open_time_ms": k[0], "timestamp": ms_to_str(k[0]) + " UTC",
            "open": float(k[1]), "high": float(k[2]), "low": float(k[3]),
            "close": float(k[4]), "volume": float(k[5]), "quote_volume": float(k[7]),
            "trades": int(k[8]), "taker_buy_volume": float(k[9]),
            "taker_buy_quote_volume": float(k[10]),
        })
    df = pd.DataFrame(rows)
    save_csv(df, "raw_futures_klines.csv", "raw futures klines")
    return klines


# --- Feature Engineering (v2) ------------------------------------------------

def build_features_from_csv():
    """
    Build features directly from CSV files using vectorized pandas.
    Much faster than the old klines-list approach.
    """
    log.info("\nBuilding ML feature table (v2 — vectorized)...")

    # --- Step 1: Load spot data ---
    log.info("  Step 1: Loading spot data...")
    t0 = time.time()
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "raw_spot_klines.csv"))
    log.info("    Loaded %s rows in %.1fs", f"{len(df):,}", time.time() - t0)

    # Compute taker buy/sell volumes for CVD
    df["taker_buy_vol"] = df["taker_buy_volume"]
    df["taker_sell_vol"] = df["volume"] - df["taker_buy_volume"]

    # --- Step 2: TA indicators ---
    log.info("  Step 2: Adding TA indicators (~80 features)...")
    t0 = time.time()
    cols_before = len(df.columns)
    try:
        df = ta.add_all_ta_features(
            df, open="open", high="high", low="low", close="close",
            volume="volume", fillna=False,
        )
        log.info("    Added %d TA columns in %.1fs", len(df.columns) - cols_before, time.time() - t0)
    except Exception as e:
        log.error("    TA library failed: %s", e)
        log.warning("    Continuing without TA indicators...")

    # --- Step 3: CVD features ---
    log.info("  Step 3: Adding CVD features...")
    df["delta_btc"] = df["taker_buy_vol"] - df["taker_sell_vol"]
    df["cumulative_delta_btc"] = df["delta_btc"].cumsum()
    df["delta_sma_14"] = df["delta_btc"].rolling(14, min_periods=1).mean()

    # --- Step 4: Custom price features ---
    log.info("  Step 4: Adding custom price features...")
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Returns at multiple horizons
    for n in [1, 3, 6, 12, 24, 48, 96, 288]:
        df[f"return_{n}"] = close.pct_change(n)

    # Log returns
    df["log_return_1"] = np.log(close / close.shift(1))
    df["log_return_6"] = np.log(close / close.shift(6))

    # SMA distances
    for w in [20, 50, 100, 200]:
        sma = close.rolling(w, min_periods=1).mean()
        df[f"price_sma{w}_dist"] = (close - sma) / sma

    # EMA distances
    for w in [12, 26, 50]:
        ema = close.ewm(span=w, adjust=False).mean()
        df[f"price_ema{w}_dist"] = (close - ema) / ema

    # Candle features
    df["high_low_range"] = (high - low) / close
    df["body_ratio"] = abs(close - df["open"]) / (high - low + 1e-10)
    df["upper_wick"] = (high - np.maximum(close, df["open"])) / (high - low + 1e-10)
    df["lower_wick"] = (np.minimum(close, df["open"]) - low) / (high - low + 1e-10)

    # Time features (cyclical encoding)
    dt_series = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    hour = dt_series.dt.hour
    dow = dt_series.dt.dayofweek
    df["hour_of_day"] = hour
    df["day_of_week"] = dow
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # --- Step 5: Microstructure features (NEW) ---
    log.info("  Step 5: Adding microstructure features...")

    # Order flow imbalance at multiple windows
    for w in [6, 12, 24, 48, 96]:
        buy_sum = df["taker_buy_vol"].rolling(w, min_periods=1).sum()
        sell_sum = df["taker_sell_vol"].rolling(w, min_periods=1).sum()
        df[f"ofi_{w}"] = (buy_sum - sell_sum) / (buy_sum + sell_sum + 1e-10)

    # Volume profile
    vol_sma20 = volume.rolling(20, min_periods=1).mean()
    vol_sma100 = volume.rolling(100, min_periods=1).mean()
    df["volume_ratio_20"] = volume / (vol_sma20 + 1e-10)
    df["volume_ratio_100"] = volume / (vol_sma100 + 1e-10)
    df["volume_trend"] = vol_sma20 / (vol_sma100 + 1e-10)

    # Trade intensity
    trades = df["trades"]
    trades_sma20 = trades.rolling(20, min_periods=1).mean()
    df["trade_intensity"] = trades / (trades_sma20 + 1e-10)

    # Quote volume / volume = avg trade price proxy
    df["avg_trade_size"] = df["quote_volume"] / (trades + 1e-10)
    avg_ts_sma = df["avg_trade_size"].rolling(20, min_periods=1).mean()
    df["avg_trade_size_ratio"] = df["avg_trade_size"] / (avg_ts_sma + 1e-10)

    # Realized volatility at multiple windows
    log_ret = df["log_return_1"]
    for w in [6, 12, 24, 48, 96, 288]:
        df[f"realized_vol_{w}"] = log_ret.rolling(w, min_periods=max(2, w // 2)).std() * np.sqrt(288)  # annualized

    # Volatility of volatility (vol clustering)
    df["vol_of_vol_24"] = df["realized_vol_24"].rolling(24, min_periods=6).std()

    # Price acceleration (second derivative)
    df["price_accel_1"] = df["return_1"] - df["return_1"].shift(1)
    df["price_accel_6"] = df["return_6"] - df["return_6"].shift(6)

    # High/low relative to recent range
    for w in [24, 96, 288]:
        roll_high = high.rolling(w, min_periods=1).max()
        roll_low = low.rolling(w, min_periods=1).min()
        df[f"range_position_{w}"] = (close - roll_low) / (roll_high - roll_low + 1e-10)

    # Consecutive up/down candles
    is_up = (close > df["open"]).astype(float)
    df["consecutive_up"] = is_up.groupby((is_up != is_up.shift()).cumsum()).cumsum()
    is_down = (close < df["open"]).astype(float)
    df["consecutive_down"] = is_down.groupby((is_down != is_down.shift()).cumsum()).cumsum()

    # VWAP deviation (rolling)
    for w in [12, 48, 96]:
        cum_vol = volume.rolling(w, min_periods=1).sum()
        cum_pv = (close * volume).rolling(w, min_periods=1).sum()
        vwap = cum_pv / (cum_vol + 1e-10)
        df[f"vwap_dev_{w}"] = (close - vwap) / (vwap + 1e-10)

    # --- Step 5b: Advanced volatility features (v3 NEW) ---
    log.info("  Step 5b: Adding advanced volatility features...")

    # Parkinson volatility (uses high-low range — 5x more efficient than close-to-close)
    hl_log = np.log(high / (low + 1e-10))
    for w in [12, 24, 48, 96, 288]:
        park_var = (hl_log ** 2).rolling(w, min_periods=max(2, w // 2)).mean() / (4 * np.log(2))
        df[f"parkinson_vol_{w}"] = np.sqrt(park_var) * np.sqrt(288)  # annualized

    # Garman-Klass volatility (uses OHLC — even more efficient)
    open_price = df["open"]
    gk_term1 = 0.5 * (np.log(high / (low + 1e-10))) ** 2
    gk_term2 = -(2 * np.log(2) - 1) * (np.log(close / (open_price + 1e-10))) ** 2
    for w in [12, 24, 48, 96, 288]:
        gk_var = (gk_term1 + gk_term2).rolling(w, min_periods=max(2, w // 2)).mean()
        gk_var = gk_var.clip(lower=0)  # ensure non-negative
        df[f"garman_klass_vol_{w}"] = np.sqrt(gk_var) * np.sqrt(288)

    # EWMA volatility (exponentially weighted — adapts faster to regime changes)
    for span in [12, 24, 48, 96]:
        ewma_var = (log_ret ** 2).ewm(span=span, adjust=False).mean()
        df[f"ewma_vol_{span}"] = np.sqrt(ewma_var) * np.sqrt(288)

    # Intraday volatility pattern (typical vol at this hour — captures seasonality)
    hourly_vol = log_ret.abs().groupby(hour).transform("mean")
    df["intraday_vol_pattern"] = log_ret.abs() / (hourly_vol + 1e-10)
    # Expanding window mean of vol per hour (no future leak)
    df["hour_vol_rank"] = log_ret.abs().rolling(288 * 7, min_periods=288).rank(pct=True)

    # Vol-of-vol at multiple scales
    for w in [48, 96]:
        df[f"vol_of_vol_{w}"] = df["realized_vol_24"].rolling(w, min_periods=12).std()

    # Volatility term structure (short vs long vol ratio)
    df["vol_term_12_96"] = df["realized_vol_12"] / (df["realized_vol_96"] + 1e-10)
    df["vol_term_24_288"] = df["realized_vol_24"] / (df["realized_vol_288"] + 1e-10)
    df["vol_term_48_288"] = df["realized_vol_48"] / (df["realized_vol_288"] + 1e-10)

    # Parkinson vs realized vol ratio (measures intraday range vs close-to-close)
    df["park_vs_rv_24"] = df["parkinson_vol_24"] / (df["realized_vol_24"] + 1e-10)
    df["park_vs_rv_96"] = df["parkinson_vol_96"] / (df["realized_vol_96"] + 1e-10)

    # ATR (Average True Range) at multiple windows
    true_range = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
    )
    df["true_range"] = true_range
    for w in [12, 24, 48, 96]:
        df[f"atr_{w}"] = true_range.rolling(w, min_periods=max(2, w // 2)).mean()
        df[f"atr_ratio_{w}"] = true_range / (df[f"atr_{w}"] + 1e-10)

    # Historical threshold hit rates (what fraction of recent candles had big moves?)
    abs_ret_1 = df["return_1"].abs()
    for thresh_pct in [0.002, 0.003, 0.005]:
        big_move_mask = (abs_ret_1 > thresh_pct).astype(float)
        t_label = str(thresh_pct).replace(".", "")
        for w in [48, 96, 288]:
            df[f"hit_rate_{t_label}_{w}"] = big_move_mask.rolling(w, min_periods=12).mean()

    # Conditional volatility (vol conditional on direction)
    up_returns = log_ret.where(log_ret > 0, 0)
    dn_returns = log_ret.where(log_ret < 0, 0)
    for w in [24, 96]:
        df[f"upside_vol_{w}"] = np.sqrt((up_returns ** 2).rolling(w, min_periods=12).mean()) * np.sqrt(288)
        df[f"downside_vol_{w}"] = np.sqrt((dn_returns ** 2).rolling(w, min_periods=12).mean()) * np.sqrt(288)
        df[f"vol_skew_{w}"] = df[f"upside_vol_{w}"] / (df[f"downside_vol_{w}"] + 1e-10)

    # Mean reversion intensity: autocorrelation of returns at lag 1
    ret_series = df["return_1"]
    ret_lag1 = ret_series.shift(1)
    for w in [24, 48, 96]:
        cov = (ret_series * ret_lag1).rolling(w, min_periods=12).mean() - \
              ret_series.rolling(w, min_periods=12).mean() * ret_lag1.rolling(w, min_periods=12).mean()
        var = ret_series.rolling(w, min_periods=12).var()
        df[f"ret_autocorr_{w}"] = cov / (var + 1e-10)

    # Max drawdown features (trailing)
    for w in [48, 96, 288]:
        roll_max = close.rolling(w, min_periods=1).max()
        df[f"trail_dd_{w}"] = (close - roll_max) / (roll_max + 1e-10)
        roll_min = close.rolling(w, min_periods=1).min()
        df[f"trail_runup_{w}"] = (close - roll_min) / (roll_min + 1e-10)

    # Hurst exponent proxy (compare vol at different scales)
    df["hurst_proxy"] = np.log(df["realized_vol_48"] + 1e-10) / (np.log(df["realized_vol_12"] + 1e-10) + 1e-10)

    # --- Step 6: Regime features (NEW) ---
    log.info("  Step 6: Adding regime features...")

    # Volatility regime (percentile rank of current vol in trailing window)
    vol_24 = df["realized_vol_24"]
    df["vol_regime_rank"] = vol_24.rolling(288, min_periods=48).rank(pct=True)

    # Trend strength regime (rolling ADX-like: abs(return_48) / realized_vol_48)
    df["trend_strength_48"] = abs(df["return_48"]) / (df["realized_vol_48"] + 1e-10)
    df["trend_strength_96"] = abs(df["return_96"]) / (df["realized_vol_96"] + 1e-10)

    # Mean reversion signal: z-score of price vs rolling mean
    for w in [48, 96, 288]:
        roll_mean = close.rolling(w, min_periods=1).mean()
        roll_std = close.rolling(w, min_periods=max(2, w // 2)).std()
        df[f"zscore_{w}"] = (close - roll_mean) / (roll_std + 1e-10)

    # Hurst-like exponent proxy (ratio of realized vol at different scales)
    df["vol_scaling_ratio"] = df["realized_vol_48"] / (df["realized_vol_12"] + 1e-10)

    # --- Step 7: Cross-timeframe features (NEW) ---
    log.info("  Step 7: Adding cross-timeframe features...")

    # 15m aggregated (3 candles)
    for col, agg in [("close", "last"), ("high", "max"), ("low", "min"), ("volume", "sum")]:
        df[f"tf15m_{col}_{agg}"] = df[col].rolling(3, min_periods=1).agg(agg)

    # 1h aggregated (12 candles)
    df["tf1h_return"] = close.pct_change(12)
    df["tf1h_range"] = (high.rolling(12, min_periods=1).max() - low.rolling(12, min_periods=1).min()) / close
    df["tf1h_volume"] = volume.rolling(12, min_periods=1).sum()

    # 4h aggregated (48 candles)
    df["tf4h_return"] = close.pct_change(48)
    df["tf4h_range"] = (high.rolling(48, min_periods=1).max() - low.rolling(48, min_periods=1).min()) / close
    df["tf4h_volume"] = volume.rolling(48, min_periods=1).sum()

    # Cross-TF momentum alignment
    df["mtf_alignment"] = np.sign(df["return_6"]) + np.sign(df["return_12"]) + np.sign(df["return_48"])

    # --- Step 8: Interaction + robustness features ---
    log.info("  Step 8: Adding interaction + robustness features...")
    if "momentum_rsi" in df.columns:
        df["rsi_x_volume"] = df["momentum_rsi"] * df["volume_ratio_20"]
        df["rsi_x_vol_regime"] = df["momentum_rsi"] * df["vol_regime_rank"]
    if "trend_adx" in df.columns:
        df["adx_x_return6"] = df["trend_adx"] * abs(df["return_6"])

    # Momentum divergence: price making new highs but RSI not (bearish divergence signal)
    if "momentum_rsi" in df.columns:
        price_rank_48 = close.rolling(48, min_periods=1).rank(pct=True)
        rsi_rank_48 = df["momentum_rsi"].rolling(48, min_periods=1).rank(pct=True)
        df["divergence_48"] = price_rank_48 - rsi_rank_48  # positive = bearish divergence

    # Volume-price correlation (does volume confirm price moves?)
    df["vol_price_corr_24"] = close.pct_change().rolling(24, min_periods=12).corr(volume.pct_change())
    df["vol_price_corr_96"] = close.pct_change().rolling(96, min_periods=48).corr(volume.pct_change())

    # Regime transition detection: vol regime change speed
    if "vol_regime_rank" in df.columns:
        df["vol_regime_change"] = df["vol_regime_rank"].diff(6)
        df["vol_regime_accel"] = df["vol_regime_change"].diff(6)

    # Mean reversion strength: return relative to recent vol
    for w in [12, 48]:
        df[f"normalized_return_{w}"] = df[f"return_{w}"] / (df[f"realized_vol_{w}"] + 1e-10)

    # Spread-based features: high-low range relative to ATR
    atr_48 = df["high_low_range"].rolling(48, min_periods=1).mean()
    df["range_vs_atr"] = df["high_low_range"] / (atr_48 + 1e-10)

    # OFI trend: is order flow consistently improving? (using diff as fast proxy)
    df["ofi_trend_48"] = df["ofi_12"].rolling(48, min_periods=12).mean() - df["ofi_12"].rolling(48, min_periods=12).mean().shift(24)

    # Cross-asset momentum proxy: bitcoin dominance-like feature
    # Using volume trend as proxy for market interest
    df["volume_momentum_24"] = volume.rolling(24, min_periods=1).mean().pct_change(24)
    df["volume_momentum_96"] = volume.rolling(96, min_periods=1).mean().pct_change(96)

    # --- Step 9: Merge futures data (VECTORIZED) ---
    log.info("  Step 9: Merging futures data...")
    t0 = time.time()
    futures_path = os.path.join(OUTPUT_DIR, "raw_futures_klines.csv")
    if os.path.exists(futures_path):
        df_fut = pd.read_csv(futures_path)
        df_futures = df_fut[["open_time_ms", "close", "volume", "trades"]].rename(columns={
            "close": "futures_close", "volume": "futures_volume_btc", "trades": "futures_trades"
        })
        df = df.merge(df_futures, on="open_time_ms", how="left")
        df["basis"] = df["futures_close"] - df["close"]
        df["basis_pct"] = (df["basis"] / df["close"]) * 100
        df["futures_to_spot_vol_ratio"] = df["futures_volume_btc"] / (df["volume"] + 1e-10)
        df["futures_to_spot_vol_ratio"] = df["futures_to_spot_vol_ratio"].replace([np.inf, -np.inf], np.nan)

        # Basis momentum (NEW)
        df["basis_pct_change_6"] = df["basis_pct"].diff(6)
        df["basis_pct_change_24"] = df["basis_pct"].diff(24)
        # Basis z-score
        basis_mean = df["basis_pct"].rolling(288, min_periods=48).mean()
        basis_std = df["basis_pct"].rolling(288, min_periods=48).std()
        df["basis_zscore"] = (df["basis_pct"] - basis_mean) / (basis_std + 1e-10)

        log.info("    Futures merged in %.1fs", time.time() - t0)
    else:
        log.warning("    No futures data file found — skipping")
        for col in ["futures_close", "basis", "basis_pct", "futures_volume_btc",
                     "futures_to_spot_vol_ratio", "futures_trades",
                     "basis_pct_change_6", "basis_pct_change_24", "basis_zscore"]:
            df[col] = np.nan

    # --- Step 10: ML target columns ---
    log.info("  Step 10: Adding ML target columns...")
    close = df["close"]

    # Direction targets
    for n, label in [(1, "5m"), (6, "30m"), (12, "1h"), (24, "2h"), (36, "3h"), (48, "4h"), (96, "8h"), (288, "24h")]:
        fwd_return = close.shift(-n) / close - 1
        df[f"target_return_{n}"] = fwd_return
        direction = (fwd_return > 0).astype(float)
        direction[fwd_return.isna()] = np.nan
        df[f"target_direction_{n}"] = direction

    # Threshold return targets (NEW — predict moves > X%)
    for n in [6, 12, 24, 36, 48]:
        fwd_ret = df[f"target_return_{n}"]
        for thresh in [0.002, 0.003, 0.005, 0.01]:
            t_label = str(thresh).replace(".", "")
            up = (fwd_ret > thresh).astype(float)
            up[fwd_ret.isna()] = np.nan
            df[f"target_up_{n}_{t_label}"] = up
            down = (fwd_ret < -thresh).astype(float)
            down[fwd_ret.isna()] = np.nan
            df[f"target_down_{n}_{t_label}"] = down
            big_move = ((fwd_ret > thresh) | (fwd_ret < -thresh)).astype(float)
            big_move[fwd_ret.isna()] = np.nan
            df[f"target_bigmove_{n}_{t_label}"] = big_move

    # Trend targets
    for n in [12, 36, 288]:
        future_above = pd.DataFrame({
            f"s{i}": (close.shift(-i) > close).astype(float) for i in range(1, n + 1)
        })
        trend_strength = future_above.mean(axis=1)
        trend_strength[close.shift(-n).isna()] = np.nan
        df[f"target_trend_strength_{n}"] = trend_strength
        trend_up = (trend_strength > 0.6).astype(float)
        trend_up[trend_strength.isna()] = np.nan
        df[f"target_trend_up_{n}"] = trend_up
        trend_down = (trend_strength < 0.4).astype(float)
        trend_down[trend_strength.isna()] = np.nan
        df[f"target_trend_down_{n}"] = trend_down

    # Volatility targets
    for n in [12, 36]:
        future_closes = pd.DataFrame({f"c{i}": close.shift(-i) for i in range(1, n + 1)})
        future_max = future_closes.max(axis=1)
        future_min = future_closes.min(axis=1)
        max_runup = (future_max - close) / close
        max_drawdown = (close - future_min) / close
        mask = close.shift(-n).isna()
        max_runup[mask] = np.nan
        max_drawdown[mask] = np.nan
        df[f"target_max_runup_{n}"] = max_runup
        df[f"target_max_drawdown_{n}"] = max_drawdown

    # Big move target
    if "target_max_runup_12" in df.columns:
        df["target_big_move_12"] = ((df["target_max_runup_12"] > 0.005) |
                                     (df["target_max_drawdown_12"] > 0.005)).astype(float)
        df.loc[df["target_max_runup_12"].isna(), "target_big_move_12"] = np.nan

    # Risk-reward target: max_runup / (max_drawdown + fee)
    if "target_max_runup_12" in df.columns:
        df["target_risk_reward_12"] = df["target_max_runup_12"] / (df["target_max_drawdown_12"] + 0.001)
        df.loc[df["target_max_runup_12"].isna(), "target_risk_reward_12"] = np.nan

    # Asymmetric threshold targets: "will price hit +X% before -X/2?"
    # Vectorized: check cumulative max/min return at each future step
    log.info("  Step 10b: Adding asymmetric risk-reward targets...")
    for n in [12, 36]:
        close_arr = close.values
        mask = close.shift(-n).isna()
        for up_thresh in [0.002, 0.003, 0.005]:
            down_thresh = up_thresh / 2
            t_label = str(up_thresh).replace(".", "")
            col_name = f"target_favorable_{n}_{t_label}"
            # For each future step, compute return
            favorable = np.full(len(close_arr), np.nan)
            for step in range(1, n + 1):
                shifted = close.shift(-step).values
                ret = (shifted - close_arr) / (close_arr + 1e-10)
                if step == 1:
                    cum_max_ret = ret.copy()
                    cum_min_ret = ret.copy()
                    # Track first hit: up or down
                    hit_up = ret >= up_thresh
                    hit_down = ret <= -down_thresh
                    decided = hit_up | hit_down
                    favorable[hit_up & ~np.isnan(ret)] = 1.0
                    favorable[hit_down & ~hit_up & ~np.isnan(ret)] = 0.0
                else:
                    new_up = (ret >= up_thresh) & ~decided & ~np.isnan(ret)
                    new_down = (ret <= -down_thresh) & ~decided & ~np.isnan(ret)
                    favorable[new_up] = 1.0
                    favorable[new_down] = 0.0
                    decided = decided | new_up | new_down
            # Anything not decided and not NaN → price stayed in range → 0 (no up hit)
            favorable[~decided & ~mask.values] = 0.0
            df[col_name] = favorable

    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if not c.startswith("target_") and c not in ("open_time_ms", "timestamp")]
    log.info("    %d target columns, %d feature columns", len(target_cols), len(feature_cols))
    log.info("  Feature table: %s rows x %d columns", f"{len(df):,}", len(df.columns))

    return df


# --- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download BTC data and build ML feature table")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()

    setup_logging()
    log.info("=" * 60)
    log.info("  BTC ML Feature Engineering Pipeline v2")
    log.info("  Time: %s", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    log.info("=" * 60)

    total_start = time.time()

    if not args.features_only:
        log.info("\n--- Phase 1: Download ---")
        t0 = time.time()
        try:
            download_spot_klines()
            log.info("  Spot download: %.1fs", time.time() - t0)
        except Exception as e:
            log.error("  Spot download FAILED: %s", e)

        t0 = time.time()
        try:
            download_futures_klines()
            log.info("  Futures download: %.1fs", time.time() - t0)
        except Exception as e:
            log.error("  Futures download FAILED: %s", e)

    if not args.download_only:
        log.info("\n--- Phase 2: Feature Engineering ---")
        t0 = time.time()
        try:
            df = build_features_from_csv()
            # Save as both parquet (fast) and CSV (compatible)
            parquet_path = os.path.join(OUTPUT_DIR, "btc_features_5m.parquet")
            df.to_parquet(parquet_path, index=False)
            log.info("  Saved parquet -> %s (%.1f MB)", parquet_path, os.path.getsize(parquet_path) / 1e6)

            csv_path = os.path.join(OUTPUT_DIR, "btc_features_5m.csv")
            df.to_csv(csv_path, index=False)
            log.info("  Saved CSV -> %s (%.1f MB)", csv_path, os.path.getsize(csv_path) / 1e6)

            log.info("  Feature engineering: %.1fs", time.time() - t0)
        except Exception as e:
            log.error("  Feature engineering FAILED: %s", e)
            log.error("  %s", traceback.format_exc())

    log.info("\n  Total time: %.1fs", time.time() - total_start)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
