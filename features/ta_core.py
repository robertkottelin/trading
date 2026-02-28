"""Core TA feature computation — reusable across any OHLCV source.

Extracted from btc_indicators.py Steps 2-8c. Accepts a DataFrame with
standard OHLCV columns and returns ~309 technical features.

Usage:
    from features.ta_core import compute_ta_features
    features_df = compute_ta_features(ohlcv_df)  # no prefix
    bnc_features = compute_ta_features(bnc_df, prefix="bnc_")
"""

import numpy as np
import pandas as pd
import ta
import logging

log = logging.getLogger(__name__)


def compute_ta_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Compute ~309 TA features from an OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open_time_ms, open, high, low, close, volume.
            Optional: taker_buy_volume, quote_volume, trades.
        prefix: Prefix for all output column names (e.g. "bnc_").

    Returns:
        DataFrame with open_time_ms + prefixed feature columns. Same row count as input.
    """
    # Work on a copy
    d = df.copy()

    # Ensure required columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in d.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute taker buy/sell volumes for CVD (if available)
    _has_real_taker_data = "taker_buy_volume" in d.columns
    if _has_real_taker_data:
        d["taker_buy_vol"] = d["taker_buy_volume"]
        d["taker_sell_vol"] = d["volume"] - d["taker_buy_volume"]
    else:
        # Placeholder for TA library (not used in CVD/OFI features)
        d["taker_buy_vol"] = d["volume"] * 0.5
        d["taker_sell_vol"] = d["volume"] * 0.5

    if "trades" not in d.columns:
        d["trades"] = 1  # placeholder
    if "quote_volume" not in d.columns:
        d["quote_volume"] = d["close"] * d["volume"]

    close = d["close"]
    high = d["high"]
    low = d["low"]
    volume = d["volume"]
    open_price = d["open"]

    # --- Step 2: TA indicators (~80 features) ---
    cols_before = set(d.columns)
    try:
        d = ta.add_all_ta_features(
            d, open="open", high="high", low="low", close="close",
            volume="volume", fillna=False,
        )
        # Re-bind close/high/low/volume after ta mutation
        close = d["close"]
        high = d["high"]
        low = d["low"]
        volume = d["volume"]
        open_price = d["open"]
        log.info("  [TA] Added %d TA columns", len(d.columns) - len(cols_before))
        # Sanitize TA library outputs: replace inf and clip extreme values
        ta_cols = set(d.columns) - cols_before
        for c in ta_cols:
            if d[c].dtype in [np.float32, np.float64]:
                d[c] = d[c].replace([np.inf, -np.inf], np.nan)
    except Exception as e:
        log.warning("  [TA] TA library failed: %s — continuing without", e)

    # --- Step 3: CVD features (only with real taker data) ---
    if _has_real_taker_data:
        d["delta_btc"] = d["taker_buy_vol"] - d["taker_sell_vol"]
        d["cumulative_delta_btc"] = d["delta_btc"].cumsum()
        d["delta_sma_14"] = d["delta_btc"].rolling(14, min_periods=1).mean()

    # --- Step 4: Custom price features ---
    for n in [1, 3, 6, 12, 24, 48, 96, 288]:
        d[f"return_{n}"] = close.pct_change(n)

    d["log_return_1"] = np.log(close / close.shift(1))
    d["log_return_6"] = np.log(close / close.shift(6))

    for w in [20, 50, 100, 200]:
        sma = close.rolling(w, min_periods=1).mean()
        d[f"price_sma{w}_dist"] = (close - sma) / sma

    for w in [12, 26, 50]:
        ema = close.ewm(span=w, adjust=False).mean()
        d[f"price_ema{w}_dist"] = (close - ema) / ema

    d["high_low_range"] = (high - low) / close
    d["body_ratio"] = abs(close - open_price) / (high - low + 1e-10)
    d["upper_wick"] = (high - np.maximum(close, open_price)) / (high - low + 1e-10)
    d["lower_wick"] = (np.minimum(close, open_price) - low) / (high - low + 1e-10)

    dt_series = pd.to_datetime(d["open_time_ms"], unit="ms", utc=True)
    hour = dt_series.dt.hour
    dow = dt_series.dt.dayofweek
    d["hour_of_day"] = hour
    d["day_of_week"] = dow
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    d["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    d["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # --- Step 5: Microstructure features ---
    if _has_real_taker_data:
        for w in [6, 12, 24, 48, 96]:
            buy_sum = d["taker_buy_vol"].rolling(w, min_periods=1).sum()
            sell_sum = d["taker_sell_vol"].rolling(w, min_periods=1).sum()
            d[f"ofi_{w}"] = (buy_sum - sell_sum) / (buy_sum + sell_sum + 1e-10)

    vol_sma20 = volume.rolling(20, min_periods=1).mean()
    vol_sma100 = volume.rolling(100, min_periods=1).mean()
    d["volume_ratio_20"] = volume / (vol_sma20 + 1e-10)
    d["volume_ratio_100"] = volume / (vol_sma100 + 1e-10)
    d["volume_trend"] = vol_sma20 / (vol_sma100 + 1e-10)

    trades = d["trades"]
    trades_sma20 = trades.rolling(20, min_periods=1).mean()
    d["trade_intensity"] = trades / (trades_sma20 + 1e-10)

    d["avg_trade_size"] = d["quote_volume"] / (trades + 1e-10)
    avg_ts_sma = d["avg_trade_size"].rolling(20, min_periods=1).mean()
    d["avg_trade_size_ratio"] = d["avg_trade_size"] / (avg_ts_sma + 1e-10)

    log_ret = d["log_return_1"]
    for w in [6, 12, 24, 48, 96, 288]:
        d[f"realized_vol_{w}"] = log_ret.rolling(w, min_periods=max(2, w // 2)).std() * np.sqrt(288)

    d["vol_of_vol_24"] = d["realized_vol_24"].rolling(24, min_periods=6).std()

    d["price_accel_1"] = d["return_1"] - d["return_1"].shift(1)
    d["price_accel_6"] = d["return_6"] - d["return_6"].shift(6)

    for w in [24, 96, 288]:
        roll_high = high.rolling(w, min_periods=1).max()
        roll_low = low.rolling(w, min_periods=1).min()
        d[f"range_position_{w}"] = (close - roll_low) / (roll_high - roll_low + 1e-10)

    is_up = (close > open_price).astype(float)
    d["consecutive_up"] = is_up.groupby((is_up != is_up.shift()).cumsum()).cumsum()
    is_down = (close < open_price).astype(float)
    d["consecutive_down"] = is_down.groupby((is_down != is_down.shift()).cumsum()).cumsum()

    for w in [12, 48, 96]:
        cum_vol = volume.rolling(w, min_periods=1).sum()
        cum_pv = (close * volume).rolling(w, min_periods=1).sum()
        vwap = cum_pv / (cum_vol + 1e-10)
        vwap_safe = vwap.where(vwap > 0, np.nan)
        d[f"vwap_dev_{w}"] = ((close - vwap) / vwap_safe).clip(-1, 1)

    # --- Step 5b: Advanced volatility features ---
    hl_log = np.log(high / (low + 1e-10))
    for w in [12, 24, 48, 96, 288]:
        park_var = (hl_log ** 2).rolling(w, min_periods=max(2, w // 2)).mean() / (4 * np.log(2))
        d[f"parkinson_vol_{w}"] = np.sqrt(park_var) * np.sqrt(288)

    gk_term1 = 0.5 * (np.log(high / (low + 1e-10))) ** 2
    gk_term2 = -(2 * np.log(2) - 1) * (np.log(close / (open_price + 1e-10))) ** 2
    for w in [12, 24, 48, 96, 288]:
        gk_var = (gk_term1 + gk_term2).rolling(w, min_periods=max(2, w // 2)).mean()
        gk_var = gk_var.clip(lower=0)
        d[f"garman_klass_vol_{w}"] = np.sqrt(gk_var) * np.sqrt(288)

    for span in [12, 24, 48, 96]:
        ewma_var = (log_ret ** 2).ewm(span=span, adjust=False).mean()
        d[f"ewma_vol_{span}"] = np.sqrt(ewma_var) * np.sqrt(288)

    abs_log_ret = log_ret.abs()
    hourly_vol_expanding = abs_log_ret.groupby(hour).transform(
        lambda x: x.expanding(min_periods=288).mean()
    )
    d["intraday_vol_pattern"] = abs_log_ret / (hourly_vol_expanding + 1e-10)
    d["hour_vol_rank"] = abs_log_ret.rolling(288 * 7, min_periods=288).rank(pct=True)

    for w in [48, 96]:
        d[f"vol_of_vol_{w}"] = d["realized_vol_24"].rolling(w, min_periods=12).std()

    d["vol_term_12_96"] = d["realized_vol_12"] / (d["realized_vol_96"] + 1e-10)
    d["vol_term_24_288"] = d["realized_vol_24"] / (d["realized_vol_288"] + 1e-10)
    d["vol_term_48_288"] = d["realized_vol_48"] / (d["realized_vol_288"] + 1e-10)

    d["park_vs_rv_24"] = d["parkinson_vol_24"] / (d["realized_vol_24"] + 1e-10)
    d["park_vs_rv_96"] = d["parkinson_vol_96"] / (d["realized_vol_96"] + 1e-10)

    true_range = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
    )
    d["true_range"] = true_range
    for w in [12, 24, 48, 96]:
        d[f"atr_{w}"] = true_range.rolling(w, min_periods=max(2, w // 2)).mean()
        d[f"atr_ratio_{w}"] = true_range / (d[f"atr_{w}"] + 1e-10)

    abs_ret_1 = d["return_1"].abs()
    for thresh_pct in [0.002, 0.003, 0.005]:
        big_move_mask = (abs_ret_1 > thresh_pct).astype(float)
        t_label = str(thresh_pct).replace(".", "")
        for w in [48, 96, 288]:
            d[f"hit_rate_{t_label}_{w}"] = big_move_mask.rolling(w, min_periods=12).mean()

    up_returns = log_ret.where(log_ret > 0, 0)
    dn_returns = log_ret.where(log_ret < 0, 0)
    for w in [24, 96]:
        d[f"upside_vol_{w}"] = np.sqrt((up_returns ** 2).rolling(w, min_periods=12).mean()) * np.sqrt(288)
        d[f"downside_vol_{w}"] = np.sqrt((dn_returns ** 2).rolling(w, min_periods=12).mean()) * np.sqrt(288)
        d[f"vol_skew_{w}"] = d[f"upside_vol_{w}"] / (d[f"downside_vol_{w}"] + 1e-10)

    ret_series = d["return_1"]
    ret_lag1 = ret_series.shift(1)
    for w in [24, 48, 96]:
        cov = (ret_series * ret_lag1).rolling(w, min_periods=12).mean() - \
              ret_series.rolling(w, min_periods=12).mean() * ret_lag1.rolling(w, min_periods=12).mean()
        var = ret_series.rolling(w, min_periods=12).var()
        d[f"ret_autocorr_{w}"] = cov / (var + 1e-10)

    for w in [48, 96, 288]:
        roll_max = close.rolling(w, min_periods=1).max()
        d[f"trail_dd_{w}"] = (close - roll_max) / (roll_max + 1e-10)
        roll_min = close.rolling(w, min_periods=1).min()
        d[f"trail_runup_{w}"] = (close - roll_min) / (roll_min + 1e-10)

    d["hurst_proxy"] = np.log(d["realized_vol_48"] + 1e-10) / (np.log(d["realized_vol_12"] + 1e-10) + 1e-10)

    # --- Step 6: Regime features ---
    vol_24 = d["realized_vol_24"]
    d["vol_regime_rank"] = vol_24.rolling(288, min_periods=48).rank(pct=True)

    d["trend_strength_48"] = abs(d["return_48"]) / (d["realized_vol_48"] + 1e-10)
    d["trend_strength_96"] = abs(d["return_96"]) / (d["realized_vol_96"] + 1e-10)

    for w in [48, 96, 288]:
        roll_mean = close.rolling(w, min_periods=1).mean()
        roll_std = close.rolling(w, min_periods=max(2, w // 2)).std()
        d[f"zscore_{w}"] = (close - roll_mean) / (roll_std + 1e-10)

    d["vol_scaling_ratio"] = d["realized_vol_48"] / (d["realized_vol_12"] + 1e-10)

    # --- Step 7: Cross-timeframe features ---
    for col, agg in [("close", "last"), ("high", "max"), ("low", "min"), ("volume", "sum")]:
        d[f"tf15m_{col}_{agg}"] = d[col].rolling(3, min_periods=1).agg(agg)

    d["tf1h_return"] = close.pct_change(12)
    d["tf1h_range"] = (high.rolling(12, min_periods=1).max() - low.rolling(12, min_periods=1).min()) / close
    d["tf1h_volume"] = volume.rolling(12, min_periods=1).sum()

    d["tf4h_return"] = close.pct_change(48)
    d["tf4h_range"] = (high.rolling(48, min_periods=1).max() - low.rolling(48, min_periods=1).min()) / close
    d["tf4h_volume"] = volume.rolling(48, min_periods=1).sum()

    d["mtf_alignment"] = np.sign(d["return_6"]) + np.sign(d["return_12"]) + np.sign(d["return_48"])

    # --- Step 8: Interaction + robustness features ---
    if "momentum_rsi" in d.columns:
        d["rsi_x_volume"] = d["momentum_rsi"] * d["volume_ratio_20"]
        d["rsi_x_vol_regime"] = d["momentum_rsi"] * d["vol_regime_rank"]
    if "trend_adx" in d.columns:
        d["adx_x_return6"] = d["trend_adx"] * abs(d["return_6"])

    if "momentum_rsi" in d.columns:
        price_rank_48 = close.rolling(48, min_periods=1).rank(pct=True)
        rsi_rank_48 = d["momentum_rsi"].rolling(48, min_periods=1).rank(pct=True)
        d["divergence_48"] = price_rank_48 - rsi_rank_48

    d["vol_price_corr_24"] = close.pct_change().rolling(24, min_periods=12).corr(volume.pct_change())
    d["vol_price_corr_96"] = close.pct_change().rolling(96, min_periods=48).corr(volume.pct_change())

    if "vol_regime_rank" in d.columns:
        d["vol_regime_change"] = d["vol_regime_rank"].diff(6)
        d["vol_regime_accel"] = d["vol_regime_change"].diff(6)

    for w in [12, 48]:
        d[f"normalized_return_{w}"] = d[f"return_{w}"] / (d[f"realized_vol_{w}"] + 1e-10)

    atr_48 = d["high_low_range"].rolling(48, min_periods=1).mean()
    d["range_vs_atr"] = d["high_low_range"] / (atr_48 + 1e-10)

    if "ofi_12" in d.columns:
        d["ofi_trend_48"] = d["ofi_12"].rolling(48, min_periods=12).mean() - d["ofi_12"].rolling(48, min_periods=12).mean().shift(24)

    d["volume_momentum_24"] = volume.rolling(24, min_periods=1).mean().pct_change(24)
    d["volume_momentum_96"] = volume.rolling(96, min_periods=1).mean().pct_change(96)

    # --- Step 8b: Higher-order statistics ---
    for w in [24, 48, 96, 288]:
        d[f"return_skew_{w}"] = ret_series.rolling(w, min_periods=max(12, w // 4)).skew()

    for w in [24, 48, 96, 288]:
        d[f"return_kurtosis_{w}"] = ret_series.rolling(w, min_periods=max(12, w // 4)).kurt()

    vol_ret = volume.pct_change().replace([np.inf, -np.inf], np.nan)
    for w in [24, 96]:
        d[f"volume_skew_{w}"] = vol_ret.rolling(w, min_periods=12).skew()

    def rolling_entropy(series, window, n_bins=10):
        result = np.full(len(series), np.nan)
        vals = series.values
        for i in range(window, len(vals)):
            chunk = vals[i - window:i]
            chunk = chunk[~np.isnan(chunk)]
            if len(chunk) < window // 2:
                continue
            hist, _ = np.histogram(chunk, bins=n_bins)
            probs = hist / hist.sum()
            probs = probs[probs > 0]
            result[i] = -np.sum(probs * np.log2(probs))
        return result

    d["return_entropy_48"] = rolling_entropy(ret_series, 48)
    d["return_entropy_288"] = rolling_entropy(ret_series, 288)

    for w in [24, 96]:
        up_vol = (volume * (ret_series > 0).astype(float)).rolling(w, min_periods=12).sum()
        dn_vol = (volume * (ret_series < 0).astype(float)).rolling(w, min_periods=12).sum()
        d[f"vol_return_asymmetry_{w}"] = (up_vol - dn_vol) / (up_vol + dn_vol + 1e-10)

    close_arr = close.values
    for w in [24, 48, 96]:
        x = np.arange(w, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()
        result = np.full(len(close_arr), np.nan)
        for i in range(w, len(close_arr)):
            y = close_arr[i - w:i]
            if np.any(np.isnan(y)):
                continue
            y_mean = y.mean()
            ss_tot = ((y - y_mean) ** 2).sum()
            if ss_tot < 1e-20:
                result[i] = 1.0
                continue
            beta = ((x - x_mean) * (y - y_mean)).sum() / x_var
            y_pred = y_mean + beta * (x - x_mean)
            ss_res = ((y - y_pred) ** 2).sum()
            result[i] = 1.0 - ss_res / ss_tot
        d[f"momentum_quality_{w}"] = result

    d["session_asian"] = ((hour >= 0) & (hour < 8)).astype(float)
    d["session_european"] = ((hour >= 7) & (hour < 16)).astype(float)
    d["session_us"] = ((hour >= 13) & (hour < 22)).astype(float)
    d["session_overlap_eu_us"] = ((hour >= 13) & (hour < 16)).astype(float)

    for w in [12, 24, 48]:
        net_move = abs(close - close.shift(w))
        total_path = abs(close.diff()).rolling(w, min_periods=1).sum()
        d[f"efficiency_ratio_{w}"] = net_move / (total_path + 1e-10)

    higher_low = (low > low.shift(1)).astype(float)
    lower_high = (high < high.shift(1)).astype(float)
    for w in [6, 12, 24]:
        d[f"higher_lows_{w}"] = higher_low.rolling(w, min_periods=1).sum() / w
        d[f"lower_highs_{w}"] = lower_high.rolling(w, min_periods=1).sum() / w

    for w in [24, 96]:
        vol_mean = volume.rolling(w, min_periods=12).mean()
        vol_std = volume.rolling(w, min_periods=12).std()
        d[f"volume_zscore_{w}"] = (volume - vol_mean) / (vol_std + 1e-10)

    for w in [12, 24, 48]:
        d[f"return_range_{w}"] = ret_series.rolling(w, min_periods=6).max() - ret_series.rolling(w, min_periods=6).min()

    # --- Step 8c: GARCH-inspired volatility features ---
    eps2 = (log_ret ** 2).values
    neg_indicator = (log_ret.values < 0).astype(float)
    for speed_label, omega, alpha, gamma, beta in [
        ("fast", 1e-6, 0.15, 0.10, 0.75),
        ("slow", 1e-6, 0.05, 0.05, 0.90),
    ]:
        sigma2 = np.full(len(eps2), np.nan)
        init_var = np.nanmean(eps2[:100]) if len(eps2) > 100 else 1e-8
        sigma2[0] = init_var
        for i in range(1, len(eps2)):
            if np.isnan(eps2[i - 1]):
                sigma2[i] = sigma2[i - 1] if not np.isnan(sigma2[i - 1]) else init_var
            else:
                sigma2[i] = (omega + alpha * eps2[i - 1]
                            + gamma * eps2[i - 1] * neg_indicator[i - 1]
                            + beta * sigma2[i - 1])
        garch_vol = np.sqrt(sigma2) * np.sqrt(288)
        d[f"garch_vol_{speed_label}"] = garch_vol
        d[f"vol_surprise_{speed_label}"] = eps2 / (sigma2 + 1e-20)

    d["garch_vol_ratio"] = d["garch_vol_fast"] / (d["garch_vol_slow"] + 1e-10)
    d["garch_vs_rv_24"] = d["garch_vol_fast"] / (d["realized_vol_24"] + 1e-10)
    d["garch_vs_rv_96"] = d["garch_vol_slow"] / (d["realized_vol_96"] + 1e-10)

    garch_vol_fast_s = pd.Series(d["garch_vol_fast"], index=ret_series.index)
    for w in [24, 96]:
        d[f"leverage_effect_{w}"] = ret_series.rolling(w, min_periods=12).corr(
            garch_vol_fast_s.shift(1)
        )

    for w in [12, 24, 48]:
        vw_ret = (ret_series * volume).rolling(w, min_periods=6).sum() / (volume.rolling(w, min_periods=6).sum() + 1e-10)
        d[f"vw_return_{w}"] = vw_ret

    for lag in [2, 3, 6, 12]:
        ret_lagged = ret_series.shift(lag)
        for w in [48, 96]:
            cov = (ret_series * ret_lagged).rolling(w, min_periods=12).mean() - \
                  ret_series.rolling(w, min_periods=12).mean() * ret_lagged.rolling(w, min_periods=12).mean()
            var_r = ret_series.rolling(w, min_periods=12).var()
            d[f"ret_autocorr_lag{lag}_{w}"] = cov / (var_r + 1e-10)

    hourly_mean_vol = d.groupby(hour)["realized_vol_6"].transform(
        lambda x: x.expanding(min_periods=288).mean()
    )
    d["session_vol_anomaly"] = d["realized_vol_6"] / (hourly_mean_vol + 1e-10)

    # --- Build output ---
    # Drop raw OHLCV and intermediate columns, keep only features
    drop_cols = {"open", "high", "low", "close", "volume", "taker_buy_volume",
                 "taker_buy_vol", "taker_sell_vol", "quote_volume", "trades",
                 "timestamp", "open_time_ms", "taker_buy_quote_volume",
                 "usd_volume", "starting_oi", "orderbook_mid_open", "orderbook_mid_close"}
    feature_cols = sorted(c for c in d.columns if c not in drop_cols)

    result = d[["open_time_ms"]].copy()
    for c in feature_cols:
        out_name = f"{prefix}{c}" if prefix else c
        result[out_name] = d[c].values

    log.info("  [TA] %d features computed%s", len(feature_cols),
             f" (prefix={prefix})" if prefix else "")
    return result


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 97 ML target columns from a DataFrame with 'close' column.

    Args:
        df: DataFrame with open_time_ms and close columns.

    Returns:
        DataFrame with open_time_ms + target columns.
    """
    result = df[["open_time_ms"]].copy()
    close = df["close"]

    # Direction + return targets
    for n, label in [(1, "5m"), (6, "30m"), (12, "1h"), (24, "2h"),
                     (36, "3h"), (48, "4h"), (96, "8h"), (288, "24h")]:
        fwd_return = close.shift(-n) / close - 1
        result[f"target_return_{n}"] = fwd_return
        direction = (fwd_return > 0).astype(float)
        direction[fwd_return.isna()] = np.nan
        result[f"target_direction_{n}"] = direction

    # Threshold targets
    for n in [6, 12, 24, 36, 48]:
        fwd_ret = result[f"target_return_{n}"]
        for thresh in [0.002, 0.003, 0.005, 0.01]:
            t_label = str(thresh).replace(".", "")
            up = (fwd_ret > thresh).astype(float)
            up[fwd_ret.isna()] = np.nan
            result[f"target_up_{n}_{t_label}"] = up
            down = (fwd_ret < -thresh).astype(float)
            down[fwd_ret.isna()] = np.nan
            result[f"target_down_{n}_{t_label}"] = down
            big_move = ((fwd_ret > thresh) | (fwd_ret < -thresh)).astype(float)
            big_move[fwd_ret.isna()] = np.nan
            result[f"target_bigmove_{n}_{t_label}"] = big_move

    # Trend targets
    for n in [12, 36, 288]:
        future_above = pd.DataFrame({
            f"s{i}": (close.shift(-i) > close).astype(float) for i in range(1, n + 1)
        })
        trend_strength = future_above.mean(axis=1)
        trend_strength[close.shift(-n).isna()] = np.nan
        result[f"target_trend_strength_{n}"] = trend_strength
        trend_up = (trend_strength > 0.6).astype(float)
        trend_up[trend_strength.isna()] = np.nan
        result[f"target_trend_up_{n}"] = trend_up
        trend_down = (trend_strength < 0.4).astype(float)
        trend_down[trend_strength.isna()] = np.nan
        result[f"target_trend_down_{n}"] = trend_down

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
        result[f"target_max_runup_{n}"] = max_runup
        result[f"target_max_drawdown_{n}"] = max_drawdown

    # Big move target
    if "target_max_runup_12" in result.columns:
        result["target_big_move_12"] = ((result["target_max_runup_12"] > 0.005) |
                                        (result["target_max_drawdown_12"] > 0.005)).astype(float)
        result.loc[result["target_max_runup_12"].isna(), "target_big_move_12"] = np.nan

    # Risk-reward target
    if "target_max_runup_12" in result.columns:
        result["target_risk_reward_12"] = result["target_max_runup_12"] / (result["target_max_drawdown_12"] + 0.001)
        result.loc[result["target_max_runup_12"].isna(), "target_risk_reward_12"] = np.nan

    # Asymmetric risk-reward targets
    for n in [12, 36]:
        close_arr = close.values
        mask = close.shift(-n).isna()
        for up_thresh in [0.002, 0.003, 0.005]:
            down_thresh = up_thresh / 2
            t_label = str(up_thresh).replace(".", "")
            col_name = f"target_favorable_{n}_{t_label}"
            favorable = np.full(len(close_arr), np.nan)
            for step in range(1, n + 1):
                shifted = close.shift(-step).values
                ret = (shifted - close_arr) / (close_arr + 1e-10)
                if step == 1:
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
            favorable[~decided & ~mask.values] = 0.0
            result[col_name] = favorable

    target_cols = [c for c in result.columns if c.startswith("target_")]
    log.info("  [Targets] %d target columns computed", len(target_cols))
    return result
