"""
Bybit — BTC derivatives data. FREE, no auth required.

Provides: klines, funding history, open interest, tickers.
Cross-exchange comparison with Binance/dYdX for divergence signals.
"""

import time
import pandas as pd
from datetime import datetime, timezone, timedelta

from downloaders.base import BaseDownloader


class BybitDownloader(BaseDownloader):
    name = "bybit"

    def __init__(self, full=False):
        super().__init__(full=full)
        self.base_url = self.cfg.get("base_url", "https://api.bybit.com")
        self.delay = self.cfg.get("rate_limit_delay", 0.5)
        self.symbol = "BTCUSDT"
        self.category = "linear"  # USDT perpetual

    def _bybit_get(self, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        return self._http_get(url, params or {}, delay=self.delay)

    # ---- Klines -----------------------------------------------------------

    def _download_klines(self):
        """Download 5-minute klines.

        Bybit v5 kline API returns newest-first.  When `start` and `end` are
        both provided, the API returns the *most recent* `limit` candles
        inside that window.  To walk backward through history we therefore
        paginate by moving `end` backward after each batch.
        """
        self.log.info("  Downloading Bybit klines...")

        start_ms = int(datetime(2020, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
        if not self.full:
            last_ms = self._get_last_timestamp_ms("bybit_klines_5m.csv", "open_time_ms")
            if last_ms:
                start_ms = last_ms + 300000

        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        all_rows = []
        current_end = end_ms
        limit = 1000

        while current_end > start_ms:
            params = {
                "category": self.category,
                "symbol": self.symbol,
                "interval": "5",
                "start": start_ms,
                "end": current_end,
                "limit": limit,
            }
            try:
                resp = self._bybit_get("/v5/market/kline", params)
                body = resp.json()
            except Exception as e:
                self.log.error("  Kline download failed at %s: %s",
                               self._ms_to_str(current_end), e)
                break

            result = body.get("result", {})
            klines = result.get("list", [])

            if not klines:
                break

            for k in klines:
                ts = int(k[0])
                all_rows.append({
                    "open_time_ms": ts,
                    "timestamp": self._ms_to_str(ts) + " UTC",
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "turnover": float(k[6]),
                })

            # Move end backward: oldest candle in this batch is klines[-1]
            # (newest-first order), step back by one candle
            oldest_ts = int(klines[-1][0])
            new_end = oldest_ts - 1
            if new_end >= current_end:
                # Safety: prevent infinite loop if API returns same data
                self.log.warning("  Pagination stuck at %s, breaking", self._ms_to_str(oldest_ts))
                break
            current_end = new_end
            if len(klines) < limit:
                break

            if len(all_rows) % 10000 < limit:
                self.log.info("    ... %s klines", f"{len(all_rows):,}")
            time.sleep(self.delay)

        self.log.info("    Total: %s klines", f"{len(all_rows):,}")

        if all_rows:
            df = pd.DataFrame(all_rows)
            if self.full:
                self._save_csv(df, "bybit_klines_5m.csv", "Bybit 5m klines",
                               sort_by="open_time_ms", dedup_col="open_time_ms")
            else:
                self._append_csv(df, "bybit_klines_5m.csv",
                                 sort_by="open_time_ms", dedup_col="open_time_ms")

    # ---- Funding Rate History ---------------------------------------------

    def _download_funding(self):
        """Download historical funding rates.

        Bybit v5 funding history returns newest-first.  Paginate backward
        by moving ``endTime`` to just before the oldest entry in each batch.
        """
        self.log.info("  Downloading Bybit funding rates...")

        start_ms = int(datetime(2020, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)
        if not self.full:
            last_ms = self._get_last_timestamp_ms("bybit_funding_rates.csv", "funding_time_ms")
            if last_ms:
                start_ms = last_ms + 1

        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        all_rows = []
        limit = 200  # Bybit max per page
        current_end = end_ms

        while current_end > start_ms:
            params = {
                "category": self.category,
                "symbol": self.symbol,
                "startTime": start_ms,
                "endTime": current_end,
                "limit": limit,
            }
            try:
                resp = self._bybit_get("/v5/market/funding/history", params)
                body = resp.json()
            except Exception as e:
                self.log.error("  Funding download failed: %s", e)
                break

            result = body.get("result", {})
            rates = result.get("list", [])

            if not rates:
                break

            for r in rates:
                ts = int(r.get("fundingRateTimestamp", 0))
                all_rows.append({
                    "funding_time_ms": ts,
                    "timestamp": self._ms_to_str(ts) + " UTC",
                    "funding_rate": float(r.get("fundingRate", 0)),
                })

            # Move end backward past the oldest entry in this batch
            oldest_ts = int(rates[-1].get("fundingRateTimestamp", 0))
            new_end = oldest_ts - 1
            if new_end >= current_end:
                self.log.warning("  Funding pagination stuck, breaking")
                break
            current_end = new_end
            if len(rates) < limit:
                break

            if len(all_rows) % 2000 < limit:
                self.log.info("    ... %s funding rates", f"{len(all_rows):,}")
            time.sleep(self.delay)

        self.log.info("    Total: %s funding rates", f"{len(all_rows):,}")

        if all_rows:
            df = pd.DataFrame(all_rows)
            if self.full:
                self._save_csv(df, "bybit_funding_rates.csv", "Bybit funding",
                               sort_by="funding_time_ms", dedup_col="funding_time_ms")
            else:
                self._append_csv(df, "bybit_funding_rates.csv",
                                 sort_by="funding_time_ms", dedup_col="funding_time_ms")

    # ---- Open Interest (5-min intervals) ----------------------------------

    def _download_open_interest(self):
        """Download open interest history (5-min intervals)."""
        self.log.info("  Downloading Bybit open interest...")

        # Bybit OI history only provides recent data
        start_ms = int((datetime.now(timezone.utc) - timedelta(days=200)).timestamp() * 1000)
        if not self.full:
            last_ms = self._get_last_timestamp_ms("bybit_open_interest.csv", "timestamp_ms")
            if last_ms:
                start_ms = last_ms + 1

        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        all_rows = []
        limit = 200
        # Use cursor-based pagination
        cursor = ""

        while True:
            params = {
                "category": self.category,
                "symbol": self.symbol,
                "intervalTime": "5min",
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                resp = self._bybit_get("/v5/market/open-interest", params)
                body = resp.json()
            except Exception as e:
                self.log.error("  OI download failed: %s", e)
                break

            result = body.get("result", {})
            entries = result.get("list", [])

            if not entries:
                break

            for e_item in entries:
                ts = int(e_item.get("timestamp", 0))
                all_rows.append({
                    "timestamp_ms": ts,
                    "timestamp": self._ms_to_str(ts) + " UTC",
                    "open_interest": float(e_item.get("openInterest", 0)),
                })

            next_cursor = result.get("nextPageCursor", "")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

            if len(all_rows) % 5000 < limit:
                self.log.info("    ... %s OI entries", f"{len(all_rows):,}")
            time.sleep(self.delay)

        self.log.info("    Total: %s OI entries", f"{len(all_rows):,}")

        if all_rows:
            df = pd.DataFrame(all_rows)
            if self.full:
                self._save_csv(df, "bybit_open_interest.csv", "Bybit OI",
                               sort_by="timestamp_ms", dedup_col="timestamp_ms")
            else:
                self._append_csv(df, "bybit_open_interest.csv",
                                 sort_by="timestamp_ms", dedup_col="timestamp_ms")

    # ---- Current Ticker (snapshot with funding + OI) ----------------------

    def _download_ticker(self):
        """Snapshot of current market state."""
        self.log.info("  Downloading Bybit ticker snapshot...")
        try:
            resp = self._bybit_get("/v5/market/tickers", {
                "category": self.category,
                "symbol": self.symbol,
            })
            body = resp.json()
        except Exception as e:
            self.log.error("  Ticker failed: %s", e)
            return

        result = body.get("result", {})
        tickers = result.get("list", [])
        if not tickers:
            return

        t = tickers[0]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        row = pd.DataFrame([{
            "timestamp": now,
            "last_price": float(t.get("lastPrice", 0)),
            "index_price": float(t.get("indexPrice", 0)),
            "mark_price": float(t.get("markPrice", 0)),
            "funding_rate": float(t.get("fundingRate", 0)),
            "next_funding_time": t.get("nextFundingTime", ""),
            "open_interest_value": float(t.get("openInterestValue", 0)),
            "volume_24h": float(t.get("volume24h", 0)),
            "turnover_24h": float(t.get("turnover24h", 0)),
            "bid1_price": float(t.get("bid1Price", 0)),
            "ask1_price": float(t.get("ask1Price", 0)),
        }])
        self._append_csv(row, "bybit_ticker_snapshots.csv",
                         sort_by="timestamp", dedup_col="timestamp")

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        self._download_klines()
        self._download_funding()
        self._download_open_interest()
        self._download_ticker()

    def download_incremental(self):
        self._download_klines()
        self._download_funding()
        self._download_open_interest()
        self._download_ticker()


if __name__ == "__main__":
    BybitDownloader.main()
