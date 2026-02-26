"""
OKX — BTC derivatives data. FREE, no auth required.

Provides: klines, funding rates, open interest, liquidation orders,
taker volume ratios, long/short ratios.
"""

import time
import pandas as pd
from datetime import datetime, timezone, timedelta

from downloaders.base import BaseDownloader


class OkxDownloader(BaseDownloader):
    name = "okx"

    def __init__(self, full=False, **kwargs):
        super().__init__(full=full, **kwargs)
        self.base_url = self.cfg.get("base_url", "https://www.okx.com")
        self.delay = self.cfg.get("rate_limit_delay", 0.5)
        self.inst_id = "BTC-USDT-SWAP"
        self.ccy = "BTC"

    def _okx_get(self, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        return self._http_get(url, params or {}, delay=self.delay)

    # ---- Funding Rates ----------------------------------------------------

    def _download_funding(self):
        """Download historical funding rates.

        OKX returns newest-first.  The ``after`` parameter means
        "return records with fundingTime earlier than this value",
        which lets us paginate backward through history.
        """
        self.log.info("  Downloading OKX funding rates...")

        all_rows = []
        after = ""  # pagination cursor (fundingTime ms string)

        while True:
            params = {"instId": self.inst_id, "limit": "100"}
            if after:
                params["after"] = after

            try:
                resp = self._okx_get("/api/v5/public/funding-rate-history", params)
                body = resp.json()
            except Exception as e:
                self.log.error("  Funding download failed: %s", e)
                break

            data = body.get("data", [])
            if not data:
                break

            for d in data:
                ts = int(d.get("fundingTime", 0))
                all_rows.append({
                    "funding_time_ms": ts,
                    "timestamp": self._ms_to_str(ts) + " UTC",
                    "funding_rate": float(d.get("fundingRate", 0)),
                    "realized_rate": float(d.get("realizedRate", 0) or 0),
                })

            # Use the oldest entry's fundingTime as cursor for next page
            after = data[-1].get("fundingTime", "")
            if len(data) < 100:
                break

            if len(all_rows) % 1000 < 100:
                self.log.info("    ... %s funding rates", f"{len(all_rows):,}")
            time.sleep(self.delay)

        self.log.info("    Total: %s funding rates", f"{len(all_rows):,}")

        if all_rows:
            df = pd.DataFrame(all_rows)
            if self.full:
                self._save_csv(df, "okx_funding_rates.csv", "OKX funding",
                               sort_by="funding_time_ms", dedup_col="funding_time_ms")
            else:
                self._append_csv(df, "okx_funding_rates.csv",
                                 sort_by="funding_time_ms", dedup_col="funding_time_ms")

    # ---- Open Interest ----------------------------------------------------

    def _download_open_interest(self):
        """Download current + recent open interest."""
        self.log.info("  Downloading OKX open interest...")

        # OKX provides OI history via /api/v5/rubik/stat/contracts-open-interest-history
        # but also current via /api/v5/public/open-interest
        params = {"instType": "SWAP", "instId": self.inst_id}
        try:
            resp = self._okx_get("/api/v5/public/open-interest", params)
            body = resp.json()
        except Exception as e:
            self.log.error("  OI download failed: %s", e)
            return

        data = body.get("data", [])
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        for d in data:
            ts = int(d.get("ts", 0))
            row = pd.DataFrame([{
                "timestamp": now,
                "timestamp_ms": ts,
                "open_interest": float(d.get("oi", 0)),
                "open_interest_ccy": float(d.get("oiCcy", 0)),
            }])
            self._append_csv(row, "okx_open_interest.csv",
                             sort_by="timestamp_ms", dedup_col="timestamp_ms")

    # ---- Taker Volume (Buy/Sell) ------------------------------------------

    def _download_taker_volume(self):
        """Download taker buy/sell volume from ticker data."""
        self.log.info("  Downloading OKX taker volume from ticker...")
        try:
            resp = self._okx_get("/api/v5/market/ticker", {"instId": self.inst_id})
            body = resp.json()
        except Exception as e:
            self.log.error("  Taker volume failed: %s", e)
            return

        data = body.get("data", [])
        if not data:
            self.log.warning("  No taker volume data")
            return

        t = data[0]
        ts = int(t.get("ts", 0))
        row = pd.DataFrame([{
            "timestamp_ms": ts,
            "timestamp": self._ms_to_str(ts) + " UTC",
            "last_price": float(t.get("last", 0)),
            "ask_price": float(t.get("askPx", 0)),
            "bid_price": float(t.get("bidPx", 0)),
            "volume_24h": float(t.get("vol24h", 0)),
            "volume_ccy_24h": float(t.get("volCcy24h", 0)),
            "open_24h": float(t.get("open24h", 0)),
            "high_24h": float(t.get("high24h", 0)),
            "low_24h": float(t.get("low24h", 0)),
        }])
        self._append_csv(row, "okx_taker_volume.csv",
                         sort_by="timestamp_ms", dedup_col="timestamp_ms")

    # ---- Liquidations (public) --------------------------------------------

    def _download_liquidations(self):
        """Download recent liquidation orders (OKX makes these public)."""
        self.log.info("  Downloading OKX liquidations...")

        params = {"instType": "SWAP", "uly": "BTC-USDT", "state": "filled"}
        try:
            resp = self._okx_get("/api/v5/public/liquidation-orders", params)
            body = resp.json()
        except Exception as e:
            self.log.error("  Liquidations failed: %s", e)
            return

        data = body.get("data", [])
        if not data:
            self.log.warning("  No liquidation data")
            return

        rows = []
        for d in data:
            details = d.get("details", [])
            for det in details:
                ts = int(det.get("ts", 0))
                rows.append({
                    "timestamp_ms": ts,
                    "timestamp": self._ms_to_str(ts) + " UTC",
                    "side": det.get("side", ""),
                    "price": float(det.get("bkPx", 0)),
                    "size": float(det.get("sz", 0)),
                })

        if rows:
            df = pd.DataFrame(rows)
            self._append_csv(df, "okx_liquidations.csv",
                             sort_by="timestamp_ms", dedup_col="timestamp_ms")

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        self._download_funding()
        self._download_open_interest()
        self._download_taker_volume()
        self._download_liquidations()

    def download_incremental(self):
        # Funding always re-fetches (API returns newest-first, limited depth)
        # Snapshots always append (point-in-time data)
        self._download_funding()
        self._download_open_interest()
        self._download_taker_volume()
        self._download_liquidations()


if __name__ == "__main__":
    OkxDownloader.main()
