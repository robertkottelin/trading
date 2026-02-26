"""
Deribit — BTC options + futures data. FREE, no auth required.

Provides: DVOL (implied vol index), options book summaries (put/call OI, IV),
historical volatility, funding rates, futures basis.

This is the richest free source of options data available.
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from downloaders.base import BaseDownloader


class DeribitDownloader(BaseDownloader):
    name = "deribit"

    def __init__(self, full=False, **kwargs):
        super().__init__(full=full, **kwargs)
        self.base_url = self.cfg.get("base_url", "https://www.deribit.com/api/v2")
        self.delay = self.cfg.get("rate_limit_delay", 0.5)

    def _deribit_get(self, method, params=None):
        """Call Deribit public API."""
        url = f"{self.base_url}/public/{method}"
        return self._http_get(url, params or {}, delay=self.delay)

    # ---- DVOL (BTC Implied Volatility Index) ------------------------------

    def _download_dvol(self):
        """Download DVOL index history — BTC's VIX equivalent.

        Uses the ticker endpoint for BTC-DVOL. For historical data,
        we use the get_tradingview_chart_data endpoint which provides
        OHLC data for the DVOL index.
        """
        self.log.info("  Downloading Deribit DVOL (BTC implied vol index)...")

        # Historical DVOL via get_volatility_index_data endpoint
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        # DVOL available since ~2021
        start_ts = self.start_override_ms or int(datetime(2021, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)

        if not self.full and not self.start_override_ms:
            last_ms = self._get_last_timestamp_ms("deribit_dvol.csv", "timestamp_ms")
            if last_ms:
                start_ts = last_ms + 3600000  # advance 1 hour to avoid overlap

        all_rows = []
        current = start_ts
        # Paginate in 60-day chunks to avoid oversized responses
        chunk_ms = 60 * 24 * 3600 * 1000

        while current < end_ts:
            chunk_end = min(current + chunk_ms, end_ts)
            params = {
                "currency": "BTC",
                "start_timestamp": current,
                "end_timestamp": chunk_end,
                "resolution": "3600",  # 1-hour OHLC
            }
            try:
                resp = self._deribit_get("get_volatility_index_data", params)
                result = resp.json().get("result", {})
            except Exception as e:
                self.log.warning("  DVOL data failed at %s: %s",
                                 self._ms_to_str(current), e)
                current = chunk_end
                continue

            data_points = result.get("data", [])

            for d in data_points:
                # Each entry is [timestamp_ms, open, high, low, close]
                if isinstance(d, list) and len(d) >= 5:
                    all_rows.append({
                        "timestamp_ms": d[0],
                        "timestamp": self._ms_to_str(d[0]) + " UTC",
                        "dvol_open": d[1],
                        "dvol_high": d[2],
                        "dvol_low": d[3],
                        "dvol_close": d[4],
                    })

            current = chunk_end
            if len(all_rows) % 5000 < 2000:
                self.log.info("    ... %s DVOL points (to %s)",
                              f"{len(all_rows):,}", self._ms_to_str(chunk_end))
            time.sleep(self.delay)

        self.log.info("    Total: %s DVOL data points", f"{len(all_rows):,}")

        if all_rows:
            df = pd.DataFrame(all_rows)
            if self.full:
                self._save_csv(df, "deribit_dvol.csv", "Deribit DVOL",
                               sort_by="timestamp_ms", dedup_col="timestamp_ms")
            else:
                self._append_csv(df, "deribit_dvol.csv",
                                 sort_by="timestamp_ms", dedup_col="timestamp_ms")

    # ---- Options Book Summary (current snapshot) --------------------------

    def _download_options_summary(self):
        """Download current options book summary — all BTC options."""
        self.log.info("  Downloading Deribit options book summary...")
        try:
            resp = self._deribit_get("get_book_summary_by_currency", {
                "currency": "BTC", "kind": "option"
            })
            data = resp.json().get("result", [])
        except Exception as e:
            self.log.error("  Options book summary failed: %s", e)
            return

        if not data:
            self.log.warning("  No options data returned")
            return

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Aggregate: total put OI, total call OI, weighted avg IV, etc.
        total_call_oi = 0
        total_put_oi = 0
        total_call_volume = 0
        total_put_volume = 0
        iv_sum_call = 0
        iv_count_call = 0
        iv_sum_put = 0
        iv_count_put = 0

        for opt in data:
            name = opt.get("instrument_name", "")
            oi = opt.get("open_interest", 0) or 0
            vol = opt.get("volume_usd", 0) or 0
            iv = opt.get("mark_iv", 0) or 0

            if "-C" in name:
                total_call_oi += oi
                total_call_volume += vol
                if iv > 0:
                    iv_sum_call += iv * oi
                    iv_count_call += oi
            elif "-P" in name:
                total_put_oi += oi
                total_put_volume += vol
                if iv > 0:
                    iv_sum_put += iv * oi
                    iv_count_put += oi

        put_call_oi_ratio = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        put_call_vol_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        avg_call_iv = iv_sum_call / iv_count_call if iv_count_call > 0 else 0
        avg_put_iv = iv_sum_put / iv_count_put if iv_count_put > 0 else 0

        row = pd.DataFrame([{
            "timestamp": now,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "put_call_oi_ratio": round(put_call_oi_ratio, 4),
            "total_call_volume_usd": total_call_volume,
            "total_put_volume_usd": total_put_volume,
            "put_call_vol_ratio": round(put_call_vol_ratio, 4),
            "avg_call_iv": round(avg_call_iv, 2),
            "avg_put_iv": round(avg_put_iv, 2),
            "iv_skew": round(avg_put_iv - avg_call_iv, 2),
            "num_options": len(data),
        }])
        self._append_csv(row, "deribit_options_summary.csv",
                         sort_by="timestamp", dedup_col="timestamp")

    # ---- Futures Book Summary ---------------------------------------------

    def _download_futures_summary(self):
        """Download current futures summary (all BTC futures)."""
        self.log.info("  Downloading Deribit futures summary...")
        try:
            resp = self._deribit_get("get_book_summary_by_currency", {
                "currency": "BTC", "kind": "future"
            })
            data = resp.json().get("result", [])
        except Exception as e:
            self.log.error("  Futures summary failed: %s", e)
            return

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        rows = []
        for f in data:
            rows.append({
                "timestamp": now,
                "instrument": f.get("instrument_name", ""),
                "mark_price": f.get("mark_price", 0) or 0,
                "oi": f.get("open_interest", 0) or 0,
                "volume_usd": f.get("volume_usd", 0) or 0,
                "bid_price": f.get("bid_price", 0) or 0,
                "ask_price": f.get("ask_price", 0) or 0,
                "estimated_delivery_price": f.get("estimated_delivery_price", 0) or 0,
            })

        if rows:
            df = pd.DataFrame(rows)
            self._append_csv(df, "deribit_futures_summary.csv",
                             sort_by="timestamp", dedup_col="timestamp")

    # ---- Historical Volatility (realized vol) -----------------------------

    def _download_historical_vol(self):
        """Download historical realized volatility from Deribit."""
        self.log.info("  Downloading Deribit historical volatility...")
        try:
            resp = self._deribit_get("get_historical_volatility", {"currency": "BTC"})
            data = resp.json().get("result", [])
        except Exception as e:
            self.log.error("  Historical vol failed: %s", e)
            return

        if not data:
            return

        rows = []
        for d in data:
            # Each entry is [timestamp, realized_vol]
            if isinstance(d, list) and len(d) >= 2:
                rows.append({
                    "timestamp_ms": d[0],
                    "timestamp": self._ms_to_str(d[0]) + " UTC",
                    "realized_vol": d[1],
                })

        if rows:
            df = pd.DataFrame(rows)
            # API returns rolling window (~16 days), append to build history
            self._append_csv(df, "deribit_historical_vol.csv",
                             sort_by="timestamp_ms", dedup_col="timestamp_ms")

    # ---- Funding Rate History (Perpetual) ---------------------------------

    def _download_funding(self):
        """Download BTC-PERPETUAL funding rate history.

        Deribit returns max ~744 records per request and returns the *newest*
        records in the window.  We must use narrow end_timestamp windows
        (32-day chunks) advancing forward to get full history.
        """
        self.log.info("  Downloading Deribit perpetual funding rates...")

        final_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ts = self.start_override_ms or int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

        if not self.full and not self.start_override_ms:
            last_ms = self._get_last_timestamp_ms("deribit_funding_rates.csv", "timestamp_ms")
            if last_ms:
                start_ts = last_ms

        all_rows = []
        current = start_ts
        chunk_ms = 32 * 24 * 3600 * 1000  # 32-day chunks

        while current < final_ts:
            chunk_end = min(current + chunk_ms, final_ts)
            params = {
                "instrument_name": "BTC-PERPETUAL",
                "start_timestamp": current,
                "end_timestamp": chunk_end,
                "count": 1000,
            }
            try:
                resp = self._deribit_get("get_funding_rate_history", params)
                data = resp.json().get("result", [])
            except Exception as e:
                self.log.error("  Funding rate history failed: %s", e)
                current = chunk_end
                continue

            if data:
                for d in data:
                    all_rows.append({
                        "timestamp_ms": d.get("timestamp", 0),
                        "timestamp": self._ms_to_str(d.get("timestamp", 0)) + " UTC",
                        "interest_8h": d.get("interest_8h", 0),
                        "interest_1h": d.get("interest_1h", 0),
                        "prev_index_price": d.get("prev_index_price", 0),
                    })

            current = chunk_end

            if len(all_rows) % 5000 < 1000:
                self.log.info("    ... %s funding entries (to %s)",
                              f"{len(all_rows):,}",
                              self._ms_to_str(current))
            time.sleep(self.delay)

        self.log.info("    Total: %s funding rate entries", f"{len(all_rows):,}")

        if all_rows:
            df = pd.DataFrame(all_rows)
            if self.full:
                self._save_csv(df, "deribit_funding_rates.csv", "Deribit funding",
                               sort_by="timestamp_ms", dedup_col="timestamp_ms")
            else:
                self._append_csv(df, "deribit_funding_rates.csv",
                                 sort_by="timestamp_ms", dedup_col="timestamp_ms")

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        self._download_dvol()
        self._download_options_summary()
        self._download_futures_summary()
        self._download_historical_vol()
        self._download_funding()

    def download_incremental(self):
        self._download_dvol()
        self._download_options_summary()
        self._download_futures_summary()
        self._download_historical_vol()
        self._download_funding()

    def download_recent(self, hours=24):
        """Download only recent data. Snapshots + recent DVOL/funding."""
        # The internal methods check CSV for last timestamp (empty in
        # market_context_data/) and fall back to hardcoded dates.
        # We override the full flag to force using start_override_ms as base.
        self._download_dvol()
        self._download_options_summary()
        self._download_futures_summary()
        self._download_historical_vol()
        self._download_funding()


if __name__ == "__main__":
    DeribitDownloader.main()
