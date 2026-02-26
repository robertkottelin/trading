"""
dYdX v4 Indexer — candles, funding rates, trades (aggregated), orderbook, market stats.

Primary execution venue. All endpoints free, no auth required.
Base URL: https://indexer.dydx.trade/v4
History available since ~Oct 2023.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from downloaders.base import BaseDownloader


class DydxDownloader(BaseDownloader):
    name = "dydx"

    def __init__(self, full=False, **kwargs):
        super().__init__(full=full, **kwargs)
        self.base_url = self.cfg.get("base_url", "https://indexer.dydx.trade/v4")
        self.market = self.cfg.get("market", "BTC-USD")
        self.start_date = self.cfg.get("start_date", "2023-10-01")
        self.delay = self.cfg.get("rate_limit_delay", 0.5)
        self.candle_resolution = self.cfg.get("candle_resolution", "5MINS")
        self.candle_limit = self.cfg.get("candle_limit", 100)
        self.trade_limit = self.cfg.get("trade_limit", 100)
        self.funding_limit = self.cfg.get("funding_limit", 100)

    # ---- Candles ----------------------------------------------------------

    def _download_candles(self, from_iso=None):
        """Download 5m candles. dYdX returns newest-first; we paginate backward."""
        self.log.info("  Downloading dYdX candles (%s)...", self.candle_resolution)
        url = f"{self.base_url}/candles/perpetualMarkets/{self.market}"

        all_candles = []
        req_count = 0
        to_iso = None  # start from latest

        # Stop when we reach this date
        stop_iso = from_iso or f"{self.start_date}T00:00:00.000Z"

        while True:
            params = {"resolution": self.candle_resolution, "limit": self.candle_limit}
            if to_iso:
                params["toISO"] = to_iso

            try:
                resp = self._http_get(url, params, delay=self.delay)
                body = resp.json()
            except Exception as e:
                self.log.error("  Candle download failed after %d reqs: %s", req_count, e)
                break

            req_count += 1
            candles = body.get("candles", [])
            if not candles:
                break

            # Filter out candles before stop point
            filtered = []
            reached_stop = False
            for c in candles:
                if c.get("startedAt", "") < stop_iso:
                    reached_stop = True
                    continue
                filtered.append(c)
            all_candles.extend(filtered)

            if reached_stop or len(candles) < self.candle_limit:
                break

            # dYdX returns newest-first; last item is oldest
            oldest = candles[-1].get("startedAt")
            if not oldest or oldest == to_iso:
                break
            to_iso = oldest

            if len(all_candles) % 5000 < self.candle_limit:
                self.log.info("    ... %s candles (back to %s)", f"{len(all_candles):,}", oldest[:19])

        self.log.info("    Total: %s candles (%d requests)", f"{len(all_candles):,}", req_count)
        return all_candles

    def _parse_candles(self, candles):
        """Parse dYdX candle dicts into a DataFrame."""
        if not candles:
            return pd.DataFrame()
        rows = []
        for c in candles:
            rows.append({
                "timestamp": c.get("startedAt", ""),
                "open": float(c.get("open", 0)),
                "high": float(c.get("high", 0)),
                "low": float(c.get("low", 0)),
                "close": float(c.get("close", 0)),
                "volume": float(c.get("baseTokenVolume", 0)),
                "usd_volume": float(c.get("usdVolume", 0)),
                "trades": int(c.get("trades", 0)),
                "starting_oi": float(c.get("startingOpenInterest", 0)),
                "orderbook_mid_open": float(c.get("orderbookMidPriceOpen", 0) or 0),
                "orderbook_mid_close": float(c.get("orderbookMidPriceClose", 0) or 0),
            })
        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _save_candles(self, candles):
        df = self._parse_candles(candles)
        if self.full:
            self._save_csv(df, "dydx_candles_5m.csv", "dYdX 5m candles",
                           sort_by="timestamp", dedup_col="timestamp")
        else:
            self._append_csv(df, "dydx_candles_5m.csv",
                             sort_by="timestamp", dedup_col="timestamp")

    # ---- Funding Rates ----------------------------------------------------

    def _download_funding(self, from_iso=None):
        """Download historical funding rates."""
        self.log.info("  Downloading dYdX funding rates...")
        url = f"{self.base_url}/historicalFunding/{self.market}"

        all_funding = []
        req_count = 0
        before_val = None
        stop_iso = from_iso or f"{self.start_date}T00:00:00.000Z"

        while True:
            params = {"limit": self.funding_limit}
            if before_val:
                params["effectiveBeforeOrAt"] = before_val

            try:
                resp = self._http_get(url, params, delay=self.delay)
                body = resp.json()
            except Exception as e:
                self.log.error("  Funding download failed after %d reqs: %s", req_count, e)
                break

            req_count += 1
            rates = body.get("historicalFunding", [])
            if not rates:
                break

            filtered = []
            reached_stop = False
            for r in rates:
                if r.get("effectiveAt", "") < stop_iso:
                    reached_stop = True
                    continue
                filtered.append(r)
            all_funding.extend(filtered)

            if reached_stop or len(rates) < self.funding_limit:
                break

            oldest = rates[-1].get("effectiveAt")
            if not oldest or oldest == before_val:
                break
            before_val = oldest

            if len(all_funding) % 2000 < self.funding_limit:
                self.log.info("    ... %s funding rates (back to %s)",
                              f"{len(all_funding):,}", oldest[:19])

        self.log.info("    Total: %s funding rates (%d requests)", f"{len(all_funding):,}", req_count)
        return all_funding

    def _parse_funding(self, funding):
        if not funding:
            return pd.DataFrame()
        rows = []
        for f in funding:
            rows.append({
                "timestamp": f.get("effectiveAt", ""),
                "ticker": f.get("ticker", self.market),
                "rate": float(f.get("rate", 0)),
                "price": float(f.get("price", 0)),
            })
        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _save_funding(self, funding):
        df = self._parse_funding(funding)
        if self.full:
            self._save_csv(df, "dydx_funding_rates.csv", "dYdX funding rates",
                           sort_by="timestamp", dedup_col="timestamp")
        else:
            self._append_csv(df, "dydx_funding_rates.csv",
                             sort_by="timestamp", dedup_col="timestamp")

    # ---- Trades (aggregated to 5m bars) -----------------------------------

    def _download_trades(self, from_iso=None):
        """Download trades and aggregate to 5m bars."""
        self.log.info("  Downloading dYdX trades (will aggregate to 5m)...")
        url = f"{self.base_url}/trades/perpetualMarket/{self.market}"

        all_trades = []
        req_count = 0
        before_val = None
        stop_iso = from_iso or f"{self.start_date}T00:00:00.000Z"

        while True:
            params = {"limit": self.trade_limit}
            if before_val:
                params["createdBeforeOrAt"] = before_val

            try:
                resp = self._http_get(url, params, delay=self.delay)
                body = resp.json()
            except Exception as e:
                self.log.error("  Trades download failed after %d reqs: %s", req_count, e)
                break

            req_count += 1
            trades = body.get("trades", [])
            if not trades:
                break

            filtered = []
            reached_stop = False
            for t in trades:
                if t.get("createdAt", "") < stop_iso:
                    reached_stop = True
                    continue
                filtered.append(t)
            all_trades.extend(filtered)

            if reached_stop or len(trades) < self.trade_limit:
                break

            oldest = trades[-1].get("createdAt")
            if not oldest or oldest == before_val:
                break
            before_val = oldest

            if len(all_trades) % 10000 < self.trade_limit:
                self.log.info("    ... %s trades (back to %s)", f"{len(all_trades):,}", oldest[:19])

        self.log.info("    Total: %s raw trades (%d requests)", f"{len(all_trades):,}", req_count)
        return all_trades

    def _aggregate_trades_5m(self, trades):
        """Aggregate tick-level trades into 5-minute bars."""
        if not trades:
            return pd.DataFrame()

        rows = []
        for t in trades:
            is_liq = t.get("type", "") in ("LIQUIDATED", "LIQUIDATION")
            size = float(t.get("size", 0))
            price = float(t.get("price", 0))
            side = t.get("side", "")
            rows.append({
                "created_at": t.get("createdAt", ""),
                "price": price,
                "size": size,
                "usd_value": size * price,
                "side": side,
                "is_liquidation": is_liq,
            })

        df = pd.DataFrame(rows)
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
        df["bucket"] = df["created_at"].dt.floor("5min")

        # Pre-compute filtered columns to avoid fragile lambda + outer scope reference
        df["buy_size"] = df["size"].where(df["side"] == "BUY", 0)
        df["sell_size"] = df["size"].where(df["side"] == "SELL", 0)
        df["liq_size"] = df["size"].where(df["is_liquidation"], 0)

        agg = df.groupby("bucket").agg(
            trade_count=("size", "count"),
            volume=("size", "sum"),
            usd_volume=("usd_value", "sum"),
            vwap=("usd_value", "sum"),  # placeholder, divide below
            buy_volume=("buy_size", "sum"),
            sell_volume=("sell_size", "sum"),
            liq_count=("is_liquidation", "sum"),
            liq_volume=("liq_size", "sum"),
        ).reset_index()

        # VWAP = total_usd / total_volume
        agg["vwap"] = np.where(agg["volume"] > 0, agg["vwap"] / agg["volume"], 0)
        agg.rename(columns={"bucket": "timestamp"}, inplace=True)
        agg["timestamp"] = agg["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        agg = agg.sort_values("timestamp").reset_index(drop=True)
        return agg

    def _save_trades(self, trades):
        df = self._aggregate_trades_5m(trades)
        if self.full:
            self._save_csv(df, "dydx_trades_5m.csv", "dYdX 5m trade bars",
                           sort_by="timestamp", dedup_col="timestamp")
        else:
            self._append_csv(df, "dydx_trades_5m.csv",
                             sort_by="timestamp", dedup_col="timestamp")

    # ---- Orderbook Snapshots (no historical backfill) ---------------------

    def _download_orderbook(self):
        """Single snapshot of the current orderbook."""
        self.log.info("  Downloading dYdX orderbook snapshot...")
        url = f"{self.base_url}/orderbooks/perpetualMarket/{self.market}"
        try:
            resp = self._http_get(url, {})
            body = resp.json()
        except Exception as e:
            self.log.error("  Orderbook snapshot failed: %s", e)
            return

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        bids = body.get("bids", [])
        asks = body.get("asks", [])

        # Summarize: best bid/ask, spread, depth at 1%/5%
        best_bid = float(bids[0].get("price", 0)) if bids else 0
        best_ask = float(asks[0].get("price", 0)) if asks else 0
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        spread = (best_ask - best_bid) / mid * 100 if mid else 0

        # Depth: sum of sizes within X% of mid
        bid_depth_1pct = sum(float(b.get("size", 0)) for b in bids
                            if mid and float(b.get("price", 0)) >= mid * 0.99)
        ask_depth_1pct = sum(float(a.get("size", 0)) for a in asks
                            if mid and float(a.get("price", 0)) <= mid * 1.01)
        bid_depth_5pct = sum(float(b.get("size", 0)) for b in bids
                            if mid and float(b.get("price", 0)) >= mid * 0.95)
        ask_depth_5pct = sum(float(a.get("size", 0)) for a in asks
                            if mid and float(a.get("price", 0)) <= mid * 1.05)

        row = pd.DataFrame([{
            "timestamp": now,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_pct": round(spread, 6),
            "bid_depth_1pct": bid_depth_1pct,
            "ask_depth_1pct": ask_depth_1pct,
            "bid_depth_5pct": bid_depth_5pct,
            "ask_depth_5pct": ask_depth_5pct,
            "num_bid_levels": len(bids),
            "num_ask_levels": len(asks),
        }])
        self._append_csv(row, "dydx_orderbook_snapshots.csv",
                         sort_by="timestamp", dedup_col="timestamp")

    # ---- Market Stats (snapshot) ------------------------------------------

    def _download_market_stats(self):
        """Snapshot of market metadata (OI, volume, etc.)."""
        self.log.info("  Downloading dYdX market stats...")
        url = f"{self.base_url}/perpetualMarkets"
        params = {"ticker": self.market}
        try:
            resp = self._http_get(url, params)
            body = resp.json()
        except Exception as e:
            self.log.error("  Market stats failed: %s", e)
            return

        markets = body.get("markets", {})
        m = markets.get(self.market, {})
        if not m:
            self.log.warning("  No market data for %s", self.market)
            return

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        row = pd.DataFrame([{
            "timestamp": now,
            "oracle_price": float(m.get("oraclePrice", 0)),
            "next_funding_rate": float(m.get("nextFundingRate", 0)),
            "initial_margin": float(m.get("initialMarginFraction", 0)),
            "maintenance_margin": float(m.get("maintenanceMarginFraction", 0)),
            "open_interest": float(m.get("openInterest", 0)),
            "open_interest_usd": float(m.get("openInterestUSDC", 0) or 0),
            "volume_24h": float(m.get("volume24H", 0)),
            "trades_24h": int(m.get("trades24H", 0)),
            "price_change_24h": float(m.get("priceChange24H", 0)),
        }])
        self._append_csv(row, "dydx_market_stats.csv",
                         sort_by="timestamp", dedup_col="timestamp")

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        """Full historical download."""
        candles = self._download_candles()
        self._save_candles(candles)

        funding = self._download_funding()
        self._save_funding(funding)

        # Trades skipped — tick-level backfill is too slow (millions of records)
        # and candle data already provides OHLCV + volume.

        self._download_orderbook()
        self._download_market_stats()

    def download_incremental(self):
        """Incremental: fetch only since last CSV row."""
        # Candles
        last_ts = self._get_last_timestamp("dydx_candles_5m.csv")
        candles = self._download_candles(from_iso=last_ts)
        self._save_candles(candles)

        # Funding
        last_ts = self._get_last_timestamp("dydx_funding_rates.csv")
        funding = self._download_funding(from_iso=last_ts)
        self._save_funding(funding)

        # Trades skipped — tick-level backfill too slow

        # Snapshots always append
        self._download_orderbook()
        self._download_market_stats()

    def download_recent(self, hours=24):
        """Download only recent data using start_override_iso."""
        from_iso = self.start_override_iso
        candles = self._download_candles(from_iso=from_iso)
        self._save_candles(candles)

        funding = self._download_funding(from_iso=from_iso)
        self._save_funding(funding)

        self._download_orderbook()
        self._download_market_stats()


if __name__ == "__main__":
    DydxDownloader.main()
