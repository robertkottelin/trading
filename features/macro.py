"""Macro features: equities, FX, commodities, crypto-adjacent + FRED (~42 features).

Sources: macro_equities.csv, macro_commodities.csv, macro_fx.csv,
         macro_crypto_adjacent.csv, macro_rates.csv, macro_liquidity.csv,
         macro_credit.csv

All daily data, lagged +1 calendar day (market close → available next day).
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_daily, rolling_zscore

LAG_DAYS = 1


def _daily_return(s: pd.Series) -> pd.Series:
    """1-day return from close prices."""
    return s.pct_change(1).astype(np.float32)


def _nday_return(s: pd.Series, n: int) -> pd.Series:
    """N-day return from close prices."""
    return s.pct_change(n).astype(np.float32)


def build_macro_features(grid: pd.DataFrame,
                         spot_close: pd.DataFrame) -> pd.DataFrame:
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    # --- Equities ---
    eq = load_csv("macro_equities.csv")
    eq_aligned = align_daily(eq, gms, "date",
                             ["GSPC_close", "IXIC_close", "RUT_close", "N225_close"],
                             "macro_", lag_days=LAG_DAYS)

    # 1-day returns
    result["macro_spx_return"] = _daily_return(eq_aligned["macro_GSPC_close"])
    result["macro_ndx_return"] = _daily_return(eq_aligned["macro_IXIC_close"])
    result["macro_rut_return"] = _daily_return(eq_aligned["macro_RUT_close"])
    result["macro_nikkei_return"] = _daily_return(eq_aligned["macro_N225_close"])

    # NDX-SPX spread (tech vs broad market)
    result["macro_ndx_spx_spread"] = (
        _daily_return(eq_aligned["macro_IXIC_close"]) -
        _daily_return(eq_aligned["macro_GSPC_close"])
    ).astype(np.float32)

    # --- FX ---
    fx = load_csv("macro_fx.csv")
    fx_aligned = align_daily(fx, gms, "date",
                             ["DXYNYB_close", "EURUSDX_close", "USDJPYX_close"],
                             "macro_", lag_days=LAG_DAYS)

    result["macro_dxy_level"] = fx_aligned["macro_DXYNYB_close"]
    result["macro_dxy_return"] = _daily_return(fx_aligned["macro_DXYNYB_close"])
    result["macro_eur_return"] = _daily_return(fx_aligned["macro_EURUSDX_close"])
    result["macro_jpy_return"] = _daily_return(fx_aligned["macro_USDJPYX_close"])

    # --- Commodities ---
    cmd = load_csv("macro_commodities.csv")
    cmd_aligned = align_daily(cmd, gms, "date",
                              ["GCF_close", "CLF_close", "HGF_close"],
                              "macro_", lag_days=LAG_DAYS)

    result["macro_gold_return"] = _daily_return(cmd_aligned["macro_GCF_close"])
    result["macro_oil_return"] = _daily_return(cmd_aligned["macro_CLF_close"])
    result["macro_copper_return"] = _daily_return(cmd_aligned["macro_HGF_close"])

    # 5-day returns
    result["macro_gold_return_5d"] = _nday_return(cmd_aligned["macro_GCF_close"], 5)
    result["macro_copper_return_5d"] = _nday_return(cmd_aligned["macro_HGF_close"], 5)

    # Copper/Gold ratio (risk-on indicator)
    result["macro_copper_gold_ratio"] = (
        cmd_aligned["macro_HGF_close"] /
        cmd_aligned["macro_GCF_close"].replace(0, np.nan)
    ).astype(np.float32)

    # --- Crypto-adjacent ---
    crypto = load_csv("macro_crypto_adjacent.csv")
    crypto_aligned = align_daily(crypto, gms, "date",
                                 ["ETHUSD_close", "IBIT_volume", "GBTC_volume"],
                                 "macro_", lag_days=LAG_DAYS)

    # ETH/BTC ratio (use spot BTC close aligned to daily)
    spot_daily = spot_close.copy()
    spot_daily["_date_ms"] = (spot_daily["open_time_ms"] // 86_400_000) * 86_400_000
    btc_daily = spot_daily.groupby("_date_ms")["close"].last()

    eth_close = crypto_aligned["macro_ETHUSD_close"]
    # For ETH/BTC, we divide ETH price by the lagged daily BTC close
    # Simple approach: compute the ratio directly on the daily grid
    result["macro_eth_btc_ratio"] = eth_close  # placeholder, actual ratio below

    # Actually compute properly: both are on same ffilled daily grid
    # We need BTC daily close aligned to the same grid
    btc_price_aligned = align_daily(
        pd.DataFrame({"date": pd.to_datetime(btc_daily.index, unit="ms", utc=True).strftime("%Y-%m-%d"),
                       "btc_close": btc_daily.values}),
        gms, "date", ["btc_close"], "macro_", lag_days=LAG_DAYS
    )
    result["macro_eth_btc_ratio"] = (
        eth_close /
        btc_price_aligned["macro_btc_close"].replace(0, np.nan)
    ).astype(np.float32)
    result["macro_eth_btc_change"] = (
        result["macro_eth_btc_ratio"].pct_change(1).astype(np.float32)
    )

    # ETF volume z-scores
    result["macro_ibit_vol_zscore"] = rolling_zscore(
        crypto_aligned["macro_IBIT_volume"], 20)
    result["macro_gbtc_vol_zscore"] = rolling_zscore(
        crypto_aligned["macro_GBTC_volume"], 20)

    # SPX-Gold 20d correlation with BTC (rolling correlation)
    # Use SPX return and BTC return on daily aligned grid
    spx_ret = result["macro_spx_return"]
    gold_ret = result["macro_gold_return"]
    result["macro_spx_gold_corr_20d"] = (
        spx_ret.rolling(20 * 288, min_periods=288).corr(gold_ret).astype(np.float32)
    )

    # --- FRED: Rates & Yield Curve (from macro_rates.csv) ---
    try:
        rates = load_csv("macro_rates.csv")
        rates_cols = [c for c in ["VIXCLS", "T10Y2Y", "T10Y3M", "DFF",
                                   "DFII10", "T5YIE", "T10YIE", "DGS10"]
                      if c in rates.columns]
        if rates_cols:
            rates_aligned = align_daily(rates, gms, "date", rates_cols,
                                        "macro_", lag_days=LAG_DAYS)

            # VIX
            if "VIXCLS" in rates_cols:
                vix = rates_aligned["macro_VIXCLS"]
                result["macro_vix_level"] = vix
                result["macro_vix_change"] = vix.diff(288).astype(np.float32)  # 1d
                result["macro_vix_zscore_30d"] = rolling_zscore(vix, 30 * 288)

            # Yield curve
            if "T10Y2Y" in rates_cols:
                t10y2y = rates_aligned["macro_T10Y2Y"]
                result["macro_t10y2y"] = t10y2y
                result["macro_t10y2y_change_5d"] = t10y2y.diff(5 * 288).astype(np.float32)
            if "T10Y3M" in rates_cols:
                result["macro_t10y3m"] = rates_aligned["macro_T10Y3M"]

            # Fed funds
            if "DFF" in rates_cols:
                ff = rates_aligned["macro_DFF"]
                result["macro_fed_funds"] = ff
                result["macro_fed_funds_change"] = ff.diff(288).astype(np.float32)

            # Real rates & breakevens
            if "DFII10" in rates_cols:
                rr = rates_aligned["macro_DFII10"]
                result["macro_real_rate_10y"] = rr
                result["macro_real_rate_change_5d"] = rr.diff(5 * 288).astype(np.float32)
            if "T5YIE" in rates_cols:
                result["macro_breakeven_5y"] = rates_aligned["macro_T5YIE"]
            if "T10YIE" in rates_cols:
                result["macro_breakeven_10y"] = rates_aligned["macro_T10YIE"]

            # 10Y Treasury
            if "DGS10" in rates_cols:
                us10y = rates_aligned["macro_DGS10"]
                result["macro_us10y_level"] = us10y
                result["macro_us10y_return"] = us10y.diff(288).astype(np.float32)
    except FileNotFoundError:
        pass  # FRED data not yet downloaded

    # --- FRED: Liquidity (from macro_liquidity.csv) ---
    try:
        liq = load_csv("macro_liquidity.csv")
        liq_cols = [c for c in ["WALCL", "RRPONTSYD"] if c in liq.columns]
        if liq_cols:
            liq_aligned = align_daily(liq, gms, "date", liq_cols,
                                       "macro_", lag_days=LAG_DAYS)

            if "WALCL" in liq_cols:
                walcl = liq_aligned["macro_WALCL"]
                # 4-week change (weekly series → 4 * 7 * 288 = 8064 candles)
                result["macro_walcl_change_4w"] = walcl.pct_change(
                    4 * 7 * 288).astype(np.float32)
            if "RRPONTSYD" in liq_cols:
                rrp = liq_aligned["macro_RRPONTSYD"]
                result["macro_rrp_level"] = rrp
                result["macro_rrp_change_5d"] = rrp.diff(5 * 288).astype(np.float32)
    except FileNotFoundError:
        pass

    # --- FRED: Credit & Stress (from macro_credit.csv) ---
    try:
        credit = load_csv("macro_credit.csv")
        credit_cols = [c for c in ["BAMLH0A0HYM2", "STLFSI4"] if c in credit.columns]
        if credit_cols:
            credit_aligned = align_daily(credit, gms, "date", credit_cols,
                                          "macro_", lag_days=LAG_DAYS)

            if "BAMLH0A0HYM2" in credit_cols:
                hy = credit_aligned["macro_BAMLH0A0HYM2"]
                result["macro_hy_spread"] = hy
                result["macro_hy_spread_change_5d"] = hy.diff(5 * 288).astype(np.float32)
            if "STLFSI4" in credit_cols:
                result["macro_fin_stress"] = credit_aligned["macro_STLFSI4"]
    except FileNotFoundError:
        pass

    return result
