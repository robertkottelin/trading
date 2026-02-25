"""
Hyperliquid — BTC DEX perpetual data. FREE, no auth required.

Provides: funding rates, open interest, mark/oracle price, premium.
Complements CEX data with on-chain leveraged positioning.
Hyperliquid funding updates hourly (not 8-hourly like most CEXes).
"""

import time
import pandas as pd
from datetime import datetime, timezone, timedelta

from downloaders.base import BaseDownloader


class HyperliquidDownloader(BaseDownloader):
    name = "hyperliquid"

    def __init__(self, full=False):
        super().__init__(full=full)
        self.base_url = self.cfg.get("base_url", "https://api.hyperliquid.xyz")
        self.delay = self.cfg.get("rate_limit_delay", 0.5)

    def _hl_post(self, payload):
        """POST to Hyperliquid info endpoint with retry logic."""
        import requests as _requests
        url = f"{self.base_url}/info"
        headers = {"Content-Type": "application/json"}
        max_retries = self.cfg.get("max_retries", 3)
        backoff = 2.0
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = _requests.post(url, json=payload, headers=headers, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", backoff * attempt))
                    self.log.warning("Rate-limited (429) — sleeping %ds", retry_after)
                    time.sleep(retry_after)
                    continue
                if resp.status_code >= 500:
                    self.log.warning("Server error %d — retry %d/%d", resp.status_code, attempt, max_retries)
                    time.sleep(backoff * attempt)
                    continue
                resp.raise_for_status()
                time.sleep(self.delay)
                return resp
            except _requests.exceptions.ConnectionError as e:
                last_exc = e
                self.log.warning("Connection error — retry %d/%d: %s", attempt, max_retries, e)
                time.sleep(backoff * attempt)
            except _requests.exceptions.Timeout as e:
                last_exc = e
                self.log.warning("Timeout — retry %d/%d", attempt, max_retries)
                time.sleep(backoff * attempt)
            except _requests.exceptions.HTTPError:
                raise
        raise _requests.exceptions.ConnectionError(f"POST failed after {max_retries} retries: {last_exc}")

    # ---- Market Snapshot (OI, funding, prices) --------------------------------

    def _download_market_snapshot(self):
        """Download current BTC market data snapshot."""
        self.log.info("  Downloading Hyperliquid BTC market snapshot...")
        try:
            resp = self._hl_post({"type": "metaAndAssetCtxs"})
            data = resp.json()
        except Exception as e:
            self.log.error("  Market snapshot failed: %s", e)
            return

        if not isinstance(data, list) or len(data) < 2:
            self.log.warning("  Unexpected response format")
            return

        meta = data[0]
        ctxs = data[1]

        # Find BTC index
        btc_idx = None
        for i, u in enumerate(meta.get("universe", [])):
            if u.get("name") == "BTC":
                btc_idx = i
                break

        if btc_idx is None:
            self.log.error("  BTC not found in Hyperliquid universe")
            return

        ctx = ctxs[btc_idx]
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        row = pd.DataFrame([{
            "timestamp": now,
            "timestamp_ms": now_ms,
            "funding_rate": float(ctx.get("funding", 0)),
            "open_interest": float(ctx.get("openInterest", 0)),
            "mark_price": float(ctx.get("markPx", 0)),
            "oracle_price": float(ctx.get("oraclePx", 0)),
            "mid_price": float(ctx.get("midPx", 0)),
            "premium": float(ctx.get("premium", 0)),
            "day_ntl_vlm": float(ctx.get("dayNtlVlm", 0)),
            "day_base_vlm": float(ctx.get("dayBaseVlm", 0)),
            "prev_day_px": float(ctx.get("prevDayPx", 0)),
        }])
        self._append_csv(row, "hyperliquid_market.csv",
                         sort_by="timestamp_ms", dedup_col="timestamp_ms")

    # ---- Funding Rate History -------------------------------------------------

    def _download_funding(self):
        """Download historical funding rates (hourly)."""
        self.log.info("  Downloading Hyperliquid BTC funding history...")

        # Hyperliquid launched ~Nov 2023; funding history available since then
        start_ms = int(datetime(2023, 11, 1, tzinfo=timezone.utc).timestamp() * 1000)
        if not self.full:
            last_ms = self._get_last_timestamp_ms(
                "hyperliquid_funding_rates.csv", "timestamp_ms")
            if last_ms:
                start_ms = last_ms + 1

        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        all_rows = []

        # API returns max 500 entries per request
        current = start_ms

        while current < end_ms:
            try:
                resp = self._hl_post({
                    "type": "fundingHistory",
                    "coin": "BTC",
                    "startTime": current,
                    "endTime": end_ms,
                })
                data = resp.json()
            except Exception as e:
                self.log.error("  Funding history failed at %s: %s",
                               self._ms_to_str(current), e)
                break

            if not data:
                break

            for d in data:
                ts = int(d.get("time", 0))
                all_rows.append({
                    "timestamp_ms": ts,
                    "timestamp": self._ms_to_str(ts) + " UTC",
                    "funding_rate": float(d.get("fundingRate", 0)),
                    "premium": float(d.get("premium", 0)),
                })

            # Advance past last entry
            last_ts = int(data[-1].get("time", 0))
            if last_ts <= current:
                break  # Safety: prevent infinite loop
            current = last_ts + 1

            if len(data) < 500:
                break

            if len(all_rows) % 5000 < 500:
                self.log.info("    ... %s funding rates", f"{len(all_rows):,}")

        self.log.info("    Total: %s funding rates", f"{len(all_rows):,}")

        if all_rows:
            df = pd.DataFrame(all_rows)
            if self.full:
                self._save_csv(df, "hyperliquid_funding_rates.csv",
                               "Hyperliquid funding",
                               sort_by="timestamp_ms", dedup_col="timestamp_ms")
            else:
                self._append_csv(df, "hyperliquid_funding_rates.csv",
                                 sort_by="timestamp_ms", dedup_col="timestamp_ms")

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        self._download_market_snapshot()
        self._download_funding()

    def download_incremental(self):
        self._download_market_snapshot()
        self._download_funding()


if __name__ == "__main__":
    HyperliquidDownloader.main()
