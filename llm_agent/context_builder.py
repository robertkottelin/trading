"""Market Context Summarizer — reads market_context_data/ CSVs into structured text.

Produces a multi-section text summary covering price, funding, OI, options,
on-chain, macro, sentiment, DeFi, and positioning data for the LLM prompt.
"""

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CONTEXT_DIR = Path(__file__).resolve().parent.parent / "market_context_data"


def _read_csv(filename: str) -> pd.DataFrame | None:
    """Safely read a CSV from market_context_data/."""
    path = CONTEXT_DIR / filename
    if not path.exists():
        log.debug("Missing: %s", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        log.warning("Failed to read %s: %s", filename, e)
        return None


def _latest_rows(df: pd.DataFrame, n: int = 1,
                 sort_col: str | None = None) -> pd.DataFrame:
    """Return the last n rows, optionally sorting first."""
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col)
    return df.tail(n)


def _fmt_num(val, decimals=2, prefix="", suffix=""):
    """Format a number for display, handling NaN."""
    if pd.isna(val):
        return "N/A"
    if abs(val) >= 1e9:
        return f"{prefix}{val/1e9:.{decimals}f}B{suffix}"
    if abs(val) >= 1e6:
        return f"{prefix}{val/1e6:.{decimals}f}M{suffix}"
    if abs(val) >= 1e3:
        return f"{prefix}{val/1e3:.{decimals}f}K{suffix}"
    return f"{prefix}{val:.{decimals}f}{suffix}"


def _pct_change(current, previous):
    """Compute percentage change, returning None if unavailable."""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return None
    return (current - previous) / abs(previous) * 100


def _section_price_volume() -> str:
    """Section 1: Price & Volume from dYdX and Binance candles."""
    lines = ["PRICE & VOLUME:"]

    dydx = _read_csv("dydx_candles_5m.csv")
    if dydx is not None and len(dydx) > 0:
        # Parse timestamp and sort
        if "timestamp" in dydx.columns:
            dydx["_ts"] = pd.to_datetime(dydx["timestamp"], utc=True)
            dydx = dydx.sort_values("_ts")

        latest = dydx.iloc[-1]
        price = float(latest["close"])
        lines.append(f"  BTC Price (dYdX): ${price:,.2f}")
        lines.append(f"  Latest candle: O={float(latest['open']):,.2f} "
                     f"H={float(latest['high']):,.2f} "
                     f"L={float(latest['low']):,.2f} "
                     f"C={float(latest['close']):,.2f}")

        if "usd_volume" in dydx.columns:
            lines.append(f"  dYdX 5m volume: {_fmt_num(float(latest.get('usd_volume', 0)), prefix='$')}")

        # 24h stats (288 candles = 24h at 5m)
        n_24h = min(288, len(dydx))
        last_24h = dydx.tail(n_24h)
        high_24h = last_24h["high"].astype(float).max()
        low_24h = last_24h["low"].astype(float).min()
        first_close_24h = float(last_24h.iloc[0]["close"])
        change_24h = _pct_change(price, first_close_24h)
        lines.append(f"  24h range: ${low_24h:,.2f} - ${high_24h:,.2f}")
        if change_24h is not None:
            lines.append(f"  24h change: {change_24h:+.2f}%")

        if "usd_volume" in dydx.columns:
            vol_24h = last_24h["usd_volume"].astype(float).sum()
            lines.append(f"  dYdX 24h volume: {_fmt_num(vol_24h, prefix='$')}")

    bnc = _read_csv("binance_futures_klines_5m.csv")
    if bnc is not None and len(bnc) > 0:
        bnc = bnc.sort_values("open_time_ms")
        latest = bnc.iloc[-1]
        lines.append(f"  Binance Futures price: ${float(latest['close']):,.2f}")
        if "quote_volume" in bnc.columns:
            n_24h = min(288, len(bnc))
            vol_24h = bnc.tail(n_24h)["quote_volume"].astype(float).sum()
            lines.append(f"  Binance Futures 24h volume: {_fmt_num(vol_24h, prefix='$')}")

    return "\n".join(lines)


def _section_funding() -> str:
    """Section 2: Funding rates across exchanges."""
    lines = ["FUNDING RATES:"]

    # (filename, exchange, sort_col, rate_col, payments_per_day)
    sources = [
        ("binance_funding_rates.csv", "Binance", "funding_time_ms", "funding_rate", 3),
        ("bybit_funding_rates.csv", "Bybit", "funding_time_ms", "funding_rate", 3),
        ("okx_funding_rates.csv", "OKX", "funding_time_ms", "funding_rate", 3),
        ("dydx_funding_rates.csv", "dYdX", None, "rate", 24),  # hourly funding
        ("deribit_funding_rates.csv", "Deribit", None, "interest_8h", 3),
    ]

    for filename, exchange, sort_col, rate_col, ppd in sources:
        df = _read_csv(filename)
        if df is None or len(df) == 0 or rate_col not in df.columns:
            continue
        latest = _latest_rows(df, 1, sort_col).iloc[-1]
        rate = float(latest[rate_col])
        ann_rate = rate * ppd * 365 * 100
        lines.append(f"  {exchange}: {rate:.6f} ({ann_rate:+.2f}% ann.)")

    # Predicted funding from Coinalyze
    pred = _read_csv("coinalyze_predicted_funding.csv")
    if pred is not None and len(pred) > 0 and "predicted_rate" in pred.columns:
        latest = _latest_rows(pred, 1, None).iloc[-1]
        rate = float(latest["predicted_rate"])
        lines.append(f"  Predicted next funding: {rate:.6f}")

    return "\n".join(lines)


def _section_open_interest() -> str:
    """Section 3: Open interest data."""
    lines = ["OPEN INTEREST:"]

    # Binance OI
    bnc_oi = _read_csv("binance_open_interest.csv")
    if bnc_oi is not None and len(bnc_oi) > 0:
        bnc_oi = bnc_oi.sort_values("timestamp_ms")
        latest = bnc_oi.iloc[-1]
        oi_val = float(latest.get("sum_open_interest_value", 0))
        oi_btc = float(latest.get("sum_open_interest", 0))
        lines.append(f"  Binance OI: {_fmt_num(oi_val, prefix='$')} ({oi_btc:,.2f} BTC)")
        if len(bnc_oi) > 1:
            prev = bnc_oi.iloc[0]
            chg = _pct_change(oi_val, float(prev.get("sum_open_interest_value", 0)))
            if chg is not None:
                lines.append(f"  Binance OI period change: {chg:+.2f}%")

    # Coinalyze aggregated OI
    ca_oi = _read_csv("coinalyze_oi_aggregated.csv")
    if ca_oi is not None and len(ca_oi) > 0:
        ca_oi = ca_oi.sort_values("timestamp_ms")
        latest = ca_oi.iloc[-1]
        oi = float(latest.get("open_interest_close", latest.get("open_interest", 0)))
        lines.append(f"  Aggregated OI (Coinalyze): {_fmt_num(oi)} BTC")

    # Bybit/OKX OI
    for fname, exchange in [("bybit_open_interest.csv", "Bybit"),
                             ("okx_open_interest.csv", "OKX")]:
        df = _read_csv(fname)
        if df is not None and len(df) > 0:
            latest = df.iloc[-1]
            for col in ["open_interest", "sum_open_interest", "oi"]:
                if col in latest.index:
                    lines.append(f"  {exchange} OI: {_fmt_num(float(latest[col]))}")
                    break

    return "\n".join(lines)


def _section_options() -> str:
    """Section 4: Options & implied volatility (Deribit)."""
    lines = ["OPTIONS & IMPLIED VOLATILITY:"]

    dvol = _read_csv("deribit_dvol.csv")
    if dvol is not None and len(dvol) > 0:
        dvol = dvol.sort_values("timestamp_ms")
        latest = dvol.iloc[-1]
        lines.append(f"  DVOL (BTC IV index): {float(latest['dvol_close']):.2f}")
        lines.append(f"  DVOL range: {float(latest['dvol_low']):.2f} - "
                     f"{float(latest['dvol_high']):.2f}")

    opts = _read_csv("deribit_options_summary.csv")
    if opts is not None and len(opts) > 0:
        latest = opts.iloc[-1]
        if "put_call_oi_ratio" in opts.columns:
            lines.append(f"  Put/Call OI ratio: {float(latest['put_call_oi_ratio']):.4f}")
        if "put_call_vol_ratio" in opts.columns:
            lines.append(f"  Put/Call volume ratio: {float(latest['put_call_vol_ratio']):.4f}")
        if "avg_call_iv" in opts.columns:
            lines.append(f"  Avg Call IV: {float(latest['avg_call_iv']):.2f}%")
            lines.append(f"  Avg Put IV: {float(latest['avg_put_iv']):.2f}%")
        if "iv_skew" in opts.columns:
            lines.append(f"  IV Skew (put-call): {float(latest['iv_skew']):.2f}")

    hvol = _read_csv("deribit_historical_vol.csv")
    if hvol is not None and len(hvol) > 0:
        latest = hvol.iloc[-1]
        for col in hvol.columns:
            if "realized" in col.lower() or "historical" in col.lower():
                lines.append(f"  {col}: {float(latest[col]):.2f}")
                break

    return "\n".join(lines)


def _section_onchain() -> str:
    """Section 5: On-chain metrics."""
    lines = ["ON-CHAIN METRICS:"]

    bc = _read_csv("blockchain_onchain.csv")
    if bc is not None and len(bc) > 0:
        bc = bc.sort_values("date")
        latest = bc.iloc[-1]
        lines.append(f"  Date: {latest['date']}")
        if "n_unique_addresses" in bc.columns and pd.notna(latest.get("n_unique_addresses")):
            lines.append(f"  Active addresses: {_fmt_num(float(latest['n_unique_addresses']), 0)}")
        if "n_transactions" in bc.columns and pd.notna(latest.get("n_transactions")):
            lines.append(f"  Transactions: {_fmt_num(float(latest['n_transactions']), 0)}")
        if "hash_rate" in bc.columns and pd.notna(latest.get("hash_rate")):
            hr = float(latest["hash_rate"])
            lines.append(f"  Hash rate: {hr:.2e} H/s")
        if "transaction_fees_usd" in bc.columns and pd.notna(latest.get("transaction_fees_usd")):
            lines.append(f"  Transaction fees: {_fmt_num(float(latest['transaction_fees_usd']), prefix='$')}")

    mempool = _read_csv("btc_network_mempool.csv")
    if mempool is not None and len(mempool) > 0:
        latest = mempool.iloc[-1]
        if "mempool_count" in mempool.columns:
            lines.append(f"  Mempool txs: {_fmt_num(float(latest['mempool_count']), 0)}")
        if "fee_fastest" in mempool.columns:
            lines.append(f"  Fee (fastest): {float(latest['fee_fastest']):.0f} sat/vB")

    mining = _read_csv("btc_network_mining.csv")
    if mining is not None and len(mining) > 0:
        latest = mining.iloc[-1]
        if "avg_hashrate" in mining.columns:
            lines.append(f"  Mining avg hashrate: {float(latest['avg_hashrate']):.2e}")
        if "difficulty" in mining.columns and pd.notna(latest.get("difficulty")):
            lines.append(f"  Difficulty: {float(latest['difficulty']):.2e}")

    return "\n".join(lines)


def _section_macro() -> str:
    """Section 6: Macro indicators."""
    lines = ["MACRO INDICATORS:"]

    # Equities
    eq = _read_csv("macro_equities.csv")
    if eq is not None and len(eq) > 0:
        eq = eq.sort_values("date")
        latest = eq.iloc[-1]
        lines.append(f"  Date: {latest['date']}")
        for sym, label in [("GSPC", "S&P 500"), ("IXIC", "NASDAQ"),
                            ("DJI", "Dow Jones"), ("N225", "Nikkei 225")]:
            close_col = f"{sym}_close"
            if close_col in eq.columns and pd.notna(latest.get(close_col)):
                lines.append(f"  {label}: {float(latest[close_col]):,.2f}")

    # FX
    fx = _read_csv("macro_fx.csv")
    if fx is not None and len(fx) > 0:
        fx = fx.sort_values("date")
        latest = fx.iloc[-1]
        if "DXYNYB_close" in fx.columns and pd.notna(latest.get("DXYNYB_close")):
            lines.append(f"  DXY: {float(latest['DXYNYB_close']):.2f}")
        if "USDJPYX_close" in fx.columns and pd.notna(latest.get("USDJPYX_close")):
            lines.append(f"  USD/JPY: {float(latest['USDJPYX_close']):.2f}")

    # Commodities
    comm = _read_csv("macro_commodities.csv")
    if comm is not None and len(comm) > 0:
        comm = comm.sort_values("date")
        latest = comm.iloc[-1]
        if "GCF_close" in comm.columns and pd.notna(latest.get("GCF_close")):
            lines.append(f"  Gold: ${float(latest['GCF_close']):,.2f}")
        if "CLF_close" in comm.columns and pd.notna(latest.get("CLF_close")):
            lines.append(f"  Crude Oil: ${float(latest['CLF_close']):.2f}")

    # Rates & VIX
    rates = _read_csv("macro_rates.csv")
    if rates is not None and len(rates) > 0:
        rates = rates.sort_values("date")
        latest = rates.iloc[-1]
        lines.append(f"  Rates date: {latest['date']}")
        for col, label in [("VIXCLS", "VIX"), ("DGS10", "10Y Treasury"),
                            ("DGS2", "2Y Treasury"), ("DFF", "Fed Funds"),
                            ("T10Y2Y", "10Y-2Y Spread")]:
            if col in rates.columns and pd.notna(latest.get(col)):
                lines.append(f"  {label}: {float(latest[col]):.2f}")

    return "\n".join(lines)


def _section_sentiment() -> str:
    """Section 7: Sentiment indicators."""
    lines = ["SENTIMENT:"]

    fng = _read_csv("sentiment_fear_greed.csv")
    if fng is not None and len(fng) > 0:
        fng = fng.sort_values("date")
        latest = fng.iloc[-1]
        lines.append(f"  Fear & Greed Index: {latest['fng_value']} ({latest['fng_classification']})")
        lines.append(f"  Date: {latest['date']}")

    mkt = _read_csv("sentiment_market.csv")
    if mkt is not None and len(mkt) > 0:
        mkt = mkt.sort_values("date")
        latest = mkt.iloc[-1]
        if "btc_market_cap" in mkt.columns:
            lines.append(f"  BTC market cap: {_fmt_num(float(latest['btc_market_cap']), prefix='$')}")
        if "btc_volume_usd" in mkt.columns:
            lines.append(f"  BTC 24h volume: {_fmt_num(float(latest['btc_volume_usd']), prefix='$')}")

    # Coinbase premium
    cb = _read_csv("coinbase_premium.csv")
    if cb is not None and len(cb) > 0:
        latest = cb.iloc[-1]
        if "premium_pct" in cb.columns:
            lines.append(f"  Coinbase premium: {float(latest['premium_pct']):.4f}%")

    return "\n".join(lines)


def _section_defi() -> str:
    """Section 8: DeFi metrics."""
    lines = ["DEFI:"]

    tvl = _read_csv("defi_tvl.csv")
    if tvl is not None and len(tvl) > 0:
        tvl = tvl.sort_values("date")
        latest = tvl.iloc[-1]
        lines.append(f"  Total DeFi TVL: {_fmt_num(float(latest['tvl']), prefix='$')}")

    stable = _read_csv("defi_stablecoin_supply.csv")
    if stable is not None and len(stable) > 0:
        # Filter for total
        total_rows = stable[stable["symbol"] == "TOTAL"] if "symbol" in stable.columns else stable
        if len(total_rows) > 0:
            latest = total_rows.iloc[-1]
            if "market_cap" in stable.columns:
                lines.append(f"  Total stablecoin supply: {_fmt_num(float(latest['market_cap']), prefix='$')}")

    return "\n".join(lines)


def _section_positioning() -> str:
    """Section 9: Positioning & liquidations."""
    lines = ["POSITIONING:"]

    # Binance long/short ratios
    for fname, label in [("binance_global_ls_ratio.csv", "Binance Global L/S"),
                          ("binance_top_ls_accounts.csv", "Binance Top Accounts L/S"),
                          ("binance_top_ls_positions.csv", "Binance Top Positions L/S")]:
        df = _read_csv(fname)
        if df is not None and len(df) > 0:
            latest = df.iloc[-1]
            for col in ["long_short_ratio", "longShortRatio", "ratio"]:
                if col in df.columns:
                    lines.append(f"  {label}: {float(latest[col]):.4f}")
                    break

    # Taker buy/sell
    taker = _read_csv("binance_taker_buy_sell.csv")
    if taker is not None and len(taker) > 0:
        latest = taker.iloc[-1]
        for col in ["buy_sell_ratio", "buySellRatio"]:
            if col in taker.columns:
                lines.append(f"  Binance Taker Buy/Sell: {float(latest[col]):.4f}")
                break

    # Coinalyze long/short
    ca_ls = _read_csv("coinalyze_long_short_ratio.csv")
    if ca_ls is not None and len(ca_ls) > 0:
        latest = ca_ls.iloc[-1]
        for col in ["long_short_ratio", "value", "ratio"]:
            if col in ca_ls.columns:
                lines.append(f"  Coinalyze L/S ratio: {float(latest[col]):.4f}")
                break

    # CFTC COT
    cot = _read_csv("cftc_cot_bitcoin.csv")
    if cot is not None and len(cot) > 0:
        cot = cot.sort_values("date")
        latest = cot.iloc[-1]
        lines.append(f"  CFTC COT date: {latest['date']}")
        if "open_interest" in cot.columns:
            lines.append(f"  CFTC OI: {int(latest['open_interest']):,}")
        if "asset_mgr_long" in cot.columns:
            lines.append(f"  Asset Mgr Long: {int(latest['asset_mgr_long']):,} "
                         f"| Short: {int(latest['asset_mgr_short']):,}")
        if "lev_money_long" in cot.columns:
            lines.append(f"  Lev Money Long: {int(latest['lev_money_long']):,} "
                         f"| Short: {int(latest['lev_money_short']):,}")

    # Liquidations
    for fname, label in [("okx_liquidations.csv", "OKX"),
                          ("coinalyze_liquidations.csv", "Coinalyze Agg")]:
        df = _read_csv(fname)
        if df is not None and len(df) > 0:
            recent = df.tail(20)
            if "side" in recent.columns and "size" in recent.columns:
                buy_liq = recent[recent["side"].isin(["buy", "long"])]["size"].astype(float).sum()
                sell_liq = recent[recent["side"].isin(["sell", "short"])]["size"].astype(float).sum()
                lines.append(f"  {label} recent liquidations: "
                             f"longs={buy_liq:.4f} BTC, shorts={sell_liq:.4f} BTC")

    return "\n".join(lines)


def build_context() -> str:
    """Build the full market context text summary from all CSVs.

    Returns a structured multi-section text string for the LLM prompt.
    """
    sections = []

    builders = [
        ("Price & Volume", _section_price_volume),
        ("Funding", _section_funding),
        ("Open Interest", _section_open_interest),
        ("Options", _section_options),
        ("On-chain", _section_onchain),
        ("Macro", _section_macro),
        ("Sentiment", _section_sentiment),
        ("DeFi", _section_defi),
        ("Positioning", _section_positioning),
    ]

    for name, builder_fn in builders:
        try:
            section = builder_fn()
            sections.append(section)
        except Exception as e:
            log.warning("Context section '%s' failed: %s", name, e)
            sections.append(f"{name.upper()}: [Error: {e}]")

    header = "MARKET CONTEXT (latest data from 58 sources):"
    return header + "\n\n" + "\n\n".join(sections)
