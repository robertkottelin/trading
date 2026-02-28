"""dYdX trade flow features (~8 features).

Source: dydx_trades_5m.csv (already aggregated to 5-minute bars by downloader).
Columns: timestamp, trade_count, volume, usd_volume, vwap,
         buy_volume, sell_volume, liq_count, liq_volume

5-minute resolution, direct merge (no lag — dYdX data is real-time).
Coverage: ~Nov 2023+ (~27% of full grid).
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_5m, rolling_zscore


def build_dydx_trades_features(grid: pd.DataFrame,
                                spot_close: pd.DataFrame) -> pd.DataFrame:
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    try:
        trades = load_csv("dydx_trades_5m.csv")
    except FileNotFoundError:
        return result

    value_cols = [c for c in ["trade_count", "volume", "usd_volume", "vwap",
                               "buy_volume", "sell_volume", "liq_count", "liq_volume"]
                  if c in trades.columns]
    if not value_cols:
        return result

    aligned = align_5m(trades, gms, "timestamp", value_cols, "dydx_")

    # --- Trade count ---
    if "dydx_trade_count" in aligned.columns:
        tc = aligned["dydx_trade_count"]
        result["dydx_trade_count"] = tc
        # 1-day z-score (288 bars = 1 day)
        result["dydx_trade_count_zscore_288"] = rolling_zscore(tc, 288)

    # --- Buy/sell flow ---
    if "dydx_buy_volume" in aligned.columns and "dydx_sell_volume" in aligned.columns:
        buy = aligned["dydx_buy_volume"]
        sell = aligned["dydx_sell_volume"]
        total = buy + sell

        # Ratio: buy / total (0.5 = balanced)
        result["dydx_buy_sell_ratio"] = (
            buy / total.replace(0, np.nan)
        ).astype(np.float32)

        # Imbalance: (buy - sell) / total (normalized, -1 to +1)
        result["dydx_buy_sell_imbalance"] = (
            (buy - sell) / total.replace(0, np.nan)
        ).astype(np.float32)

    # --- Liquidations ---
    if "dydx_liq_count" in aligned.columns:
        result["dydx_liq_count"] = aligned["dydx_liq_count"]

    if "dydx_liq_volume" in aligned.columns:
        lv = aligned["dydx_liq_volume"]
        result["dydx_liq_volume"] = lv
        # Rolling 1h intensity (12 bars)
        result["dydx_liq_intensity_1h"] = (
            lv.rolling(12, min_periods=1).sum().astype(np.float32)
        )

    # --- VWAP deviation ---
    if "dydx_vwap" in aligned.columns:
        vwap = aligned["dydx_vwap"]
        # Align spot_close to the grid before computing deviation
        close = spot_close.set_index("open_time_ms")["close"].reindex(
            gms.values
        ).values
        # (close - vwap) / close — positive = price above vwap
        result["dydx_vwap_deviation"] = np.where(
            (close != 0) & (vwap != 0) & np.isfinite(close) & np.isfinite(vwap),
            (close - vwap) / close,
            np.nan,
        ).astype(np.float32)

    return result
