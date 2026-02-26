"""
Coinalyze — aggregated cross-exchange derivatives data (free Coinglass alternative).

Free tier: 40 req/min, API key required.
Endpoints: aggregated OI, funding rates, predicted funding, liquidations,
           long/short ratio, CVD.

Daily resolution used for historical backfill (intraday retention ~5-7 days only).
"""

import os
import time
import pandas as pd
from datetime import datetime, timezone, timedelta

from downloaders.base import BaseDownloader


class CoinalyzeDownloader(BaseDownloader):
    name = "coinalyze"

    def __init__(self, full=False, **kwargs):
        super().__init__(full=full, **kwargs)
        self.base_url = self.cfg.get("base_url", "https://api.coinalyze.net/v1")
        self.symbol = self.cfg.get("symbol", "BTCUSDT_PERP.A")
        self.delay = self.cfg.get("rate_limit_delay", 1.5)
        self.start_date = self.cfg.get("start_date", "2020-01-01")
        self.api_key = os.environ.get("COINALYZE_API_KEY", "")

    def _get_headers(self):
        if self.api_key:
            return {"api_key": self.api_key}
        return {}

    def _coinalyze_get(self, endpoint, params=None):
        """GET with Coinalyze API key header."""
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        # Coinalyze uses api_key as query param
        if self.api_key:
            params["api_key"] = self.api_key
        return self._http_get(url, params, delay=self.delay)

    def _download_timeseries(self, endpoint, filename, description, value_columns,
                             ts_field="t", interval="5min"):
        """Generic timeseries downloader for Coinalyze endpoints.

        Coinalyze returns data in chunks. We paginate by time windows.
        Max 10000 data points per request.
        """
        self.log.info("  Downloading %s...", description)

        if not self.api_key:
            self.log.error("  COINALYZE_API_KEY not set — cannot download %s", description)
            raise RuntimeError("COINALYZE_API_KEY environment variable is required")

        # Determine start
        if self.start_override_ms:
            start_dt = datetime.fromtimestamp(self.start_override_ms / 1000, tz=timezone.utc)
        else:
            start_dt = datetime.strptime(self.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if not self.full and not self.start_override_ms:
            last_ts = self._get_last_timestamp_ms(filename, "timestamp_ms")
            if last_ts:
                start_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)

        end_dt = datetime.now(timezone.utc)

        # Coinalyze uses Unix seconds, not ms
        all_rows = []
        current = start_dt
        # Chunk size: ~30 days at 5min = 8640 points (under 10000 limit)
        chunk_days = 30

        while current < end_dt:
            chunk_end = min(current + timedelta(days=chunk_days), end_dt)
            params = {
                "symbols": self.symbol,
                "interval": interval,
                "from": int(current.timestamp()),
                "to": int(chunk_end.timestamp()),
            }

            try:
                resp = self._coinalyze_get(endpoint, params)
                data = resp.json()
            except Exception as e:
                self.log.error("  %s failed at %s: %s", description,
                               current.strftime("%Y-%m-%d"), e)
                break

            # Response is [{symbol, history: [...]}] — extract history array
            batch = []
            if isinstance(data, list) and data and isinstance(data[0], dict):
                if "history" in data[0]:
                    batch = data[0].get("history", [])
                elif ts_field in data[0]:
                    batch = data  # flat list of data points
            elif isinstance(data, dict):
                batch = data.get("history", data.get("data", []))

            for d in batch:
                ts = d.get(ts_field, 0)
                row = {
                    "timestamp_ms": ts * 1000 if ts < 1e12 else ts,
                    "timestamp": datetime.fromtimestamp(
                        ts if ts < 1e12 else ts / 1000, tz=timezone.utc
                    ).strftime("%Y-%m-%d %H:%M UTC"),
                }
                for csv_col, api_col in value_columns.items():
                    row[csv_col] = float(d.get(api_col, 0))
                all_rows.append(row)

            self.log.info("    ... %s rows (through %s)", f"{len(all_rows):,}",
                          chunk_end.strftime("%Y-%m-%d"))
            current = chunk_end
            time.sleep(self.delay)

        if not all_rows:
            self.log.warning("  No data for %s", description)
            return

        df = pd.DataFrame(all_rows)
        if self.full:
            self._save_csv(df, filename, description,
                           sort_by="timestamp_ms", dedup_col="timestamp_ms")
        else:
            self._append_csv(df, filename,
                             sort_by="timestamp_ms", dedup_col="timestamp_ms")

    # ---- Specific endpoints -----------------------------------------------

    def _download_oi(self):
        self._download_timeseries(
            "/open-interest-history",
            "coinalyze_oi_aggregated.csv",
            "Coinalyze aggregated OI",
            {"open_interest": "o", "open_interest_high": "h",
             "open_interest_low": "l", "open_interest_close": "c"},
        )

    def _download_funding(self):
        self._download_timeseries(
            "/funding-rate-history",
            "coinalyze_funding_rates.csv",
            "Coinalyze funding rates",
            {"funding_rate": "o", "funding_rate_high": "h",
             "funding_rate_low": "l", "funding_rate_close": "c"},
        )

    def _download_predicted_funding(self):
        """Predicted funding — snapshot only (no history endpoint)."""
        self.log.info("  Downloading Coinalyze predicted funding...")
        if not self.api_key:
            self.log.error("  COINALYZE_API_KEY not set — cannot download predicted funding")
            raise RuntimeError("COINALYZE_API_KEY environment variable is required")

        try:
            resp = self._coinalyze_get("/predicted-funding-rate",
                                       {"symbols": self.symbol})
            data = resp.json()
        except Exception as e:
            self.log.error("  Predicted funding failed: %s", e)
            return

        if not data:
            return

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        rows = []
        items = data if isinstance(data, list) else [data]
        for d in items:
            rows.append({
                "timestamp": now,
                "symbol": d.get("symbol", self.symbol),
                "predicted_rate": float(d.get("value", d.get("predicted_rate", 0))),
            })

        df = pd.DataFrame(rows)
        self._append_csv(df, "coinalyze_predicted_funding.csv",
                         sort_by="timestamp", dedup_col="timestamp")

    def _download_liquidations(self):
        self._download_timeseries(
            "/liquidation-history",
            "coinalyze_liquidations.csv",
            "Coinalyze liquidations",
            {"long_liquidations": "l", "short_liquidations": "s"},
        )

    def _download_long_short_ratio(self):
        self._download_timeseries(
            "/long-short-ratio-history",
            "coinalyze_long_short_ratio.csv",
            "Coinalyze long/short ratio",
            {"ls_ratio": "r", "long_pct": "l", "short_pct": "s"},
        )

    def _download_cvd(self):
        """CVD (Cumulative Volume Delta) — may not be available on all API tiers."""
        try:
            self._download_timeseries(
                "/cvd-history",
                "coinalyze_cvd.csv",
                "Coinalyze CVD",
                {"cvd_open": "o", "cvd_high": "h",
                 "cvd_low": "l", "cvd_close": "c"},
            )
        except Exception as e:
            self.log.warning("  CVD endpoint not available (may need paid tier): %s", e)

    # ---- Daily resolution downloads for feature pipeline ------------------

    def _download_oi_daily(self):
        self._download_timeseries(
            "/open-interest-history",
            "coinalyze_oi_daily.csv",
            "Coinalyze aggregated OI (daily)",
            {"open_interest": "o", "open_interest_high": "h",
             "open_interest_low": "l", "open_interest_close": "c"},
            interval="daily",
        )

    def _download_funding_daily(self):
        self._download_timeseries(
            "/funding-rate-history",
            "coinalyze_funding_daily.csv",
            "Coinalyze funding rates (daily)",
            {"funding_rate": "o", "funding_rate_high": "h",
             "funding_rate_low": "l", "funding_rate_close": "c"},
            interval="daily",
        )

    def _download_liquidations_daily(self):
        self._download_timeseries(
            "/liquidation-history",
            "coinalyze_liquidations_daily.csv",
            "Coinalyze liquidations (daily)",
            {"long_liquidations": "l", "short_liquidations": "s"},
            interval="daily",
        )

    def _download_long_short_ratio_daily(self):
        self._download_timeseries(
            "/long-short-ratio-history",
            "coinalyze_long_short_ratio_daily.csv",
            "Coinalyze long/short ratio (daily)",
            {"ls_ratio": "r", "long_pct": "l", "short_pct": "s"},
            interval="daily",
        )

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        self._download_oi()
        self._download_funding()
        self._download_predicted_funding()
        self._download_liquidations()
        self._download_long_short_ratio()
        self._download_cvd()
        # Daily resolution for feature pipeline
        self._download_oi_daily()
        self._download_funding_daily()
        self._download_liquidations_daily()
        self._download_long_short_ratio_daily()

    def download_incremental(self):
        self._download_oi()
        self._download_funding()
        self._download_predicted_funding()
        self._download_liquidations()
        self._download_long_short_ratio()
        self._download_cvd()
        # Daily resolution for feature pipeline
        self._download_oi_daily()
        self._download_funding_daily()
        self._download_liquidations_daily()
        self._download_long_short_ratio_daily()


if __name__ == "__main__":
    CoinalyzeDownloader.main()
