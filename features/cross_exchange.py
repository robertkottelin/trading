"""Cross-exchange spread and volume features (~21 features).

Sources: bybit_klines_5m.csv, dydx_candles_5m.csv,
         binance_mark_price_klines.csv, binance_premium_index_klines.csv,
         binance_index_price_klines.csv, binance_futures_basis.csv
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, ts_col_to_ms, rolling_zscore


def _reindex_col(src_df, col, grid_values):
    """Reindex a column from a ms-indexed DataFrame onto the grid, return numpy array."""
    return src_df[col].reindex(grid_values).values


def build_cross_exchange_features(grid: pd.DataFrame,
                                  spot_close: pd.DataFrame) -> pd.DataFrame:
    """Build 15 cross-exchange features on the 5-minute grid."""
    grid_vals = grid["open_time_ms"].values
    result = grid[["open_time_ms"]].copy()

    # Spot close as numpy array aligned to grid
    spot_idx = spot_close.set_index("open_time_ms")
    sc = spot_idx["close"].reindex(grid_vals).values.astype(np.float64)

    # --- Bybit klines ---
    bybit = load_csv("bybit_klines_5m.csv")
    bybit = bybit.drop_duplicates(subset="open_time_ms", keep="last")
    bybit = bybit.set_index(ts_col_to_ms(bybit, "open_time_ms").values)
    bybit_close = _reindex_col(bybit, "close", grid_vals).astype(np.float64)
    bybit_vol = _reindex_col(bybit, "volume", grid_vals).astype(np.float64)

    # --- dYdX candles ---
    dydx = load_csv("dydx_candles_5m.csv")
    dydx_ms = ts_col_to_ms(dydx, "timestamp").values
    dydx = dydx.set_index(dydx_ms)
    dydx = dydx[~dydx.index.duplicated(keep="last")]
    dydx_close = _reindex_col(dydx, "close", grid_vals).astype(np.float64)
    dydx_usd_vol = _reindex_col(dydx, "usd_volume", grid_vals).astype(np.float64)
    dydx_oi = _reindex_col(dydx, "starting_oi", grid_vals).astype(np.float64)
    dydx_mid_open = _reindex_col(dydx, "orderbook_mid_open", grid_vals).astype(np.float64)
    dydx_mid_close = _reindex_col(dydx, "orderbook_mid_close", grid_vals).astype(np.float64)

    # --- Binance mark price ---
    mark = load_csv("binance_mark_price_klines.csv")
    mark = mark.drop_duplicates(subset="open_time_ms", keep="last")
    mark = mark.set_index(ts_col_to_ms(mark, "open_time_ms").values)
    mark_close = _reindex_col(mark, "close", grid_vals).astype(np.float64)

    # --- Binance premium index ---
    prem = load_csv("binance_premium_index_klines.csv")
    prem = prem.drop_duplicates(subset="open_time_ms", keep="last")
    prem = prem.set_index(ts_col_to_ms(prem, "open_time_ms").values)
    prem_close = _reindex_col(prem, "close", grid_vals).astype(np.float64)

    # --- Binance spot volume ---
    spot_raw = load_csv("binance_spot_klines_5m.csv")
    spot_raw = spot_raw.drop_duplicates(subset="open_time_ms", keep="last")
    spot_raw = spot_raw.set_index(ts_col_to_ms(spot_raw, "open_time_ms").values)
    spot_vol = _reindex_col(spot_raw, "volume", grid_vals).astype(np.float64)
    spot_quote_vol = _reindex_col(spot_raw, "quote_volume", grid_vals).astype(np.float64)

    # --- Compute features (all numpy → avoids index alignment issues) ---
    sc_safe = np.where(sc == 0, np.nan, sc)

    result["bybit_binance_spread"] = ((bybit_close - sc) / sc_safe).astype(np.float32)
    result["bybit_binance_spread_zscore_288"] = rolling_zscore(
        result["bybit_binance_spread"], 288)
    result["dydx_binance_spread"] = ((dydx_close - sc) / sc_safe).astype(np.float32)
    result["dydx_binance_spread_zscore_288"] = rolling_zscore(
        result["dydx_binance_spread"], 288)

    # Mark-spot deviation
    result["mark_spot_deviation"] = ((mark_close - sc) / sc_safe).astype(np.float32)
    result["mark_spot_deviation_ma_48"] = (
        result["mark_spot_deviation"]
        .rolling(48, min_periods=1).mean().astype(np.float32)
    )

    # Premium index
    result["premium_idx_close"] = prem_close.astype(np.float32)
    prem_s = pd.Series(prem_close)
    result["premium_idx_momentum_6"] = prem_s.diff(6).astype(np.float32).values
    result["premium_idx_momentum_24"] = prem_s.diff(24).astype(np.float32).values
    result["premium_idx_zscore_288"] = rolling_zscore(prem_s, 288).values

    # Volume ratios
    spot_vol_safe = np.where(spot_vol == 0, np.nan, spot_vol)
    spot_qv_safe = np.where(spot_quote_vol == 0, np.nan, spot_quote_vol)
    result["bybit_volume_ratio"] = (bybit_vol / spot_vol_safe).astype(np.float32)
    result["dydx_volume_ratio"] = (dydx_usd_vol / spot_qv_safe).astype(np.float32)

    # dYdX OI
    result["dydx_oi_start"] = dydx_oi.astype(np.float32)
    dydx_oi_s = pd.Series(dydx_oi)
    result["dydx_oi_change_12"] = dydx_oi_s.pct_change(12).astype(np.float32).values

    # dYdX orderbook mid spread
    dydx_close_safe = np.where(dydx_close == 0, np.nan, dydx_close)
    result["dydx_orderbook_mid_spread"] = (
        (dydx_mid_close - dydx_mid_open) / dydx_close_safe
    ).astype(np.float32)

    # --- Binance index price klines (649K rows, 5m) ---
    try:
        idx = load_csv("binance_index_price_klines.csv")
        idx = idx.drop_duplicates(subset="open_time_ms", keep="last")
        idx = idx.set_index(ts_col_to_ms(idx, "open_time_ms").values)
        idx_close = _reindex_col(idx, "close", grid_vals).astype(np.float64)

        # Index-spot spread (how index deviates from spot)
        result["index_spot_spread"] = ((idx_close - sc) / sc_safe).astype(np.float32)
        result["index_spot_spread_zscore_288"] = rolling_zscore(
            result["index_spot_spread"], 288)

        # Index-futures spread (index vs mark price, basis proxy)
        mark_safe = np.where(mark_close == 0, np.nan, mark_close)
        result["index_mark_spread"] = (
            (mark_close - idx_close) / (idx_close + 1e-10)
        ).astype(np.float32)
        result["index_mark_spread_zscore_288"] = rolling_zscore(
            result["index_mark_spread"], 288)
    except (FileNotFoundError, KeyError):
        pass

    # --- Binance futures basis (511 rows, recent only) ---
    try:
        basis = load_csv("binance_futures_basis.csv")
        basis = basis.drop_duplicates(subset="open_time_ms", keep="last")
        basis = basis.set_index(ts_col_to_ms(basis, "open_time_ms").values)
        if "basis_rate" in basis.columns:
            basis_rate = _reindex_col(basis, "basis_rate", grid_vals).astype(np.float64)
            result["bnc_basis_rate"] = pd.Series(basis_rate, dtype=np.float32)
            result["bnc_basis_rate_annualized"] = (
                result["bnc_basis_rate"] * 365.25 * 288  # annualize 5m rate
            ).astype(np.float32)
    except (FileNotFoundError, KeyError):
        pass

    return result
