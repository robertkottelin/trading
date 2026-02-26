"""
CFTC Commitments of Traders (COT) — Bitcoin futures positioning data.
FREE, no auth required.

Provides: Weekly institutional positioning breakdown for CME Bitcoin futures.
- Leveraged Money (hedge funds): long/short
- Other Reportables (institutional): long/short
- Dealer: long/short
- Non-reportable (retail): long/short
- Concentration ratios

Updated weekly (Friday, data as of prior Tuesday).
Slow-moving regime signal — not for intraday but for market structure context.
"""

import pandas as pd
from datetime import datetime, timezone

from downloaders.base import BaseDownloader


class CftcDownloader(BaseDownloader):
    name = "cftc"

    CFTC_BASE = "https://publicreporting.cftc.gov/resource"
    # Traders in Financial Futures (TFF) report
    TFF_ENDPOINT = "gpe5-46if.json"
    # CME Bitcoin Futures market name
    BTC_MARKET = "BITCOIN - CHICAGO MERCANTILE EXCHANGE"

    def __init__(self, full=False, **kwargs):
        super().__init__(full=full, **kwargs)
        self.delay = self.cfg.get("rate_limit_delay", 1.0)

    def _download_cot(self):
        """Download CFTC COT positioning data for CME Bitcoin futures."""
        self.log.info("  Downloading CFTC COT Bitcoin positioning...")

        url = f"{self.CFTC_BASE}/{self.TFF_ENDPOINT}"
        params = {
            "$where": f"market_and_exchange_names = '{self.BTC_MARKET}'",
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$limit": 5000,
        }

        try:
            resp = self._http_get(url, params, delay=self.delay)
            data = resp.json()
        except Exception as e:
            self.log.error("  CFTC COT download failed: %s", e)
            return

        if not data:
            self.log.warning("  No CFTC COT data returned")
            return

        rows = []
        for d in data:
            date = d.get("report_date_as_yyyy_mm_dd", "")
            if not date:
                continue
            # Parse ISO date to clean format
            date_clean = date[:10]

            rows.append({
                "date": date_clean,
                "open_interest": int(d.get("open_interest_all", 0) or 0),
                # Leveraged Money (hedge funds)
                "lev_money_long": int(d.get("lev_money_positions_long", 0) or 0),
                "lev_money_short": int(d.get("lev_money_positions_short", 0) or 0),
                "lev_money_spread": int(d.get("lev_money_positions_spread", 0) or 0),
                # Other Reportable (institutional)
                "other_rept_long": int(d.get("other_rept_positions_long", 0) or 0),
                "other_rept_short": int(d.get("other_rept_positions_short", 0) or 0),
                "other_rept_spread": int(d.get("other_rept_positions_spread", 0) or 0),
                # Dealer
                "dealer_long": int(d.get("dealer_positions_long_all", 0) or 0),
                "dealer_short": int(d.get("dealer_positions_short_all", 0) or 0),
                "dealer_spread": int(d.get("dealer_positions_spread_all", 0) or 0),
                # Asset Manager
                "asset_mgr_long": int(d.get("asset_mgr_positions_long", 0) or 0),
                "asset_mgr_short": int(d.get("asset_mgr_positions_short", 0) or 0),
                "asset_mgr_spread": int(d.get("asset_mgr_positions_spread", 0) or 0),
                # Non-reportable (retail)
                "nonrept_long": int(d.get("nonrept_positions_long_all", 0) or 0),
                "nonrept_short": int(d.get("nonrept_positions_short_all", 0) or 0),
                # Changes from prior week
                "chg_oi": int(d.get("change_in_open_interest_all", 0) or 0),
                "chg_lev_money_long": int(d.get("change_in_lev_money_long", 0) or 0),
                "chg_lev_money_short": int(d.get("change_in_lev_money_short", 0) or 0),
                "chg_other_rept_long": int(d.get("change_in_other_rept_long", 0) or 0),
                "chg_other_rept_short": int(d.get("change_in_other_rept_short", 0) or 0),
                # Concentration
                "conc_4_tdr_long": float(d.get("conc_gross_le_4_tdr_long", 0) or 0),
                "conc_4_tdr_short": float(d.get("conc_gross_le_4_tdr_short", 0) or 0),
                "conc_8_tdr_long": float(d.get("conc_gross_le_8_tdr_long", 0) or 0),
                "conc_8_tdr_short": float(d.get("conc_gross_le_8_tdr_short", 0) or 0),
                # Pct of OI
                "pct_lev_money_long": float(d.get("pct_of_oi_lev_money_long", 0) or 0),
                "pct_lev_money_short": float(d.get("pct_of_oi_lev_money_short", 0) or 0),
                "pct_other_rept_long": float(d.get("pct_of_oi_other_rept_long", 0) or 0),
                "pct_other_rept_short": float(d.get("pct_of_oi_other_rept_short", 0) or 0),
                # Trader counts
                "traders_lev_money_long": int(d.get("traders_lev_money_long_all", 0) or 0),
                "traders_lev_money_short": int(d.get("traders_lev_money_short_all", 0) or 0),
            })

        if rows:
            df = pd.DataFrame(rows)
            self._save_csv(df, "cftc_cot_bitcoin.csv", "CFTC COT Bitcoin",
                           sort_by="date", dedup_col="date")

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        self._download_cot()

    def download_incremental(self):
        # Always re-fetch all (single request, data is small)
        self._download_cot()


if __name__ == "__main__":
    CftcDownloader.main()
