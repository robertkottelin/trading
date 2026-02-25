"""Coinalyze features: cross-exchange derivatives aggregates (~20 features).

Sources (daily):
  coinalyze_oi_daily.csv, coinalyze_funding_daily.csv,
  coinalyze_liquidations_daily.csv, coinalyze_long_short_ratio_daily.csv

Sources (5-min, recent/growing):
  coinalyze_oi_aggregated.csv, coinalyze_funding_rates.csv,
  coinalyze_liquidations.csv, coinalyze_long_short_ratio.csv

Daily resolution, lag_days=0 (crypto derivatives data has no market-close delay).
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_daily, align_5m, align_ffill, rolling_zscore

LAG_DAYS = 0


def build_coinalyze_features(grid: pd.DataFrame) -> pd.DataFrame:
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    # --- Open Interest (daily) ---
    try:
        oi = load_csv("coinalyze_oi_daily.csv")
        oi_cols = [c for c in ["open_interest_close"] if c in oi.columns]
        if oi_cols:
            oi_aligned = align_daily(oi, gms, "timestamp", oi_cols,
                                     "cz_", lag_days=LAG_DAYS)
            oi_close = oi_aligned["cz_open_interest_close"]
            result["cz_oi_daily"] = oi_close
            result["cz_oi_change_1d"] = oi_close.pct_change(288).astype(np.float32)
            result["cz_oi_change_7d"] = oi_close.pct_change(7 * 288).astype(np.float32)
            result["cz_oi_zscore_30d"] = rolling_zscore(oi_close, 30 * 288)
    except FileNotFoundError:
        pass

    # --- Funding Rate (daily) ---
    try:
        funding = load_csv("coinalyze_funding_daily.csv")
        funding_cols = [c for c in ["funding_rate_close"] if c in funding.columns]
        if funding_cols:
            f_aligned = align_daily(funding, gms, "timestamp", funding_cols,
                                    "cz_", lag_days=LAG_DAYS)
            fr = f_aligned["cz_funding_rate_close"]
            result["cz_funding_daily"] = fr
            result["cz_funding_zscore_30d"] = rolling_zscore(fr, 30 * 288)
    except FileNotFoundError:
        pass

    # --- Liquidations (daily) ---
    try:
        liq = load_csv("coinalyze_liquidations_daily.csv")
        liq_cols = [c for c in ["long_liquidations", "short_liquidations"]
                    if c in liq.columns]
        if liq_cols:
            l_aligned = align_daily(liq, gms, "timestamp", liq_cols,
                                    "cz_", lag_days=LAG_DAYS)

            long_liq = l_aligned["cz_long_liquidations"]
            short_liq = l_aligned["cz_short_liquidations"]
            result["cz_liq_long"] = long_liq
            result["cz_liq_short"] = short_liq

            # Long/short liquidation ratio
            total_liq = long_liq + short_liq
            result["cz_liq_ratio"] = (
                long_liq / total_liq.replace(0, np.nan)
            ).astype(np.float32)

            # Z-score of total liquidation volume
            result["cz_liq_total_zscore"] = rolling_zscore(total_liq, 30 * 288)
    except FileNotFoundError:
        pass

    # --- Long/Short Ratio (daily) ---
    try:
        ls = load_csv("coinalyze_long_short_ratio_daily.csv")
        ls_cols = [c for c in ["ls_ratio"] if c in ls.columns]
        if ls_cols:
            ls_aligned = align_daily(ls, gms, "timestamp", ls_cols,
                                     "cz_", lag_days=LAG_DAYS)
            ls_ratio = ls_aligned["cz_ls_ratio"]
            result["cz_ls_ratio"] = ls_ratio
            result["cz_ls_ratio_change_7d"] = ls_ratio.diff(7 * 288).astype(np.float32)
    except FileNotFoundError:
        pass

    # ===== 5-minute resolution sources (recent, growing over time) =====

    # --- OI aggregated (5m) ---
    try:
        oi5 = load_csv("coinalyze_oi_aggregated.csv")
        oi5_aligned = align_ffill(oi5, gms, "timestamp_ms",
                                  ["open_interest_close"], "cz5_")
        result["cz_oi_5m"] = oi5_aligned["cz5_open_interest_close"]
        result["cz_oi_5m_zscore"] = rolling_zscore(result["cz_oi_5m"], 288)
    except (FileNotFoundError, KeyError):
        pass

    # --- Funding rates (5m) ---
    try:
        fr5 = load_csv("coinalyze_funding_rates.csv")
        fr5_aligned = align_ffill(fr5, gms, "timestamp_ms",
                                  ["funding_rate_close"], "cz5_")
        result["cz_funding_5m"] = fr5_aligned["cz5_funding_rate_close"]
        result["cz_funding_5m_zscore"] = rolling_zscore(result["cz_funding_5m"], 288)
    except (FileNotFoundError, KeyError):
        pass

    # --- Liquidations (5m) ---
    try:
        liq5 = load_csv("coinalyze_liquidations.csv")
        liq5_aligned = align_ffill(liq5, gms, "timestamp_ms",
                                   ["long_liquidations", "short_liquidations"], "cz5_")
        result["cz_liq_long_5m"] = liq5_aligned["cz5_long_liquidations"]
        result["cz_liq_short_5m"] = liq5_aligned["cz5_short_liquidations"]
        total = result["cz_liq_long_5m"] + result["cz_liq_short_5m"]
        result["cz_liq_ratio_5m"] = (
            result["cz_liq_long_5m"] / total.replace(0, np.nan)
        ).astype(np.float32)
        result["cz_liq_total_5m_zscore"] = rolling_zscore(total, 288)
    except (FileNotFoundError, KeyError):
        pass

    # --- Long/Short ratio (5m) ---
    try:
        ls5 = load_csv("coinalyze_long_short_ratio.csv")
        ls5_aligned = align_ffill(ls5, gms, "timestamp_ms",
                                  ["ls_ratio"], "cz5_")
        result["cz_ls_ratio_5m"] = ls5_aligned["cz5_ls_ratio"]
        result["cz_ls_ratio_5m_zscore"] = rolling_zscore(result["cz_ls_ratio_5m"], 288)
    except (FileNotFoundError, KeyError):
        pass

    return result
