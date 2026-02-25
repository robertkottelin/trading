"""
Coinbase Premium — computes the Coinbase-Binance price spread.

FREE, no auth. Signals US institutional demand.
Positive premium = US buying pressure. Negative = US selling.
"""

import time
import pandas as pd
from datetime import datetime, timezone

from downloaders.base import BaseDownloader


class CoinbasePremiumDownloader(BaseDownloader):
    name = "coinbase_premium"

    def __init__(self, full=False):
        super().__init__(full=full)
        self.delay = self.cfg.get("rate_limit_delay", 0.3)

    def _download_premium_snapshot(self):
        """Compute current Coinbase-Binance premium."""
        self.log.info("  Computing Coinbase premium...")

        # Coinbase spot price
        try:
            resp_cb = self._http_get(
                "https://api.coinbase.com/v2/prices/BTC-USD/spot", {})
            cb_data = resp_cb.json().get("data", {})
            cb_price = float(cb_data.get("amount", 0))
        except Exception as e:
            self.log.error("  Coinbase price failed: %s", e)
            return

        time.sleep(0.2)

        # Binance spot price
        try:
            resp_bn = self._http_get(
                "https://api.binance.com/api/v3/ticker/price",
                {"symbol": "BTCUSDT"})
            bn_price = float(resp_bn.json().get("price", 0))
        except Exception as e:
            self.log.error("  Binance price failed: %s", e)
            return

        if cb_price <= 0 or bn_price <= 0:
            self.log.warning("  Invalid prices: CB=%s BN=%s", cb_price, bn_price)
            return

        premium = (cb_price - bn_price) / bn_price * 100
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        row = pd.DataFrame([{
            "timestamp": now,
            "coinbase_price": cb_price,
            "binance_price": bn_price,
            "premium_pct": round(premium, 6),
            "premium_usd": round(cb_price - bn_price, 2),
        }])
        self._append_csv(row, "coinbase_premium.csv",
                         sort_by="timestamp", dedup_col="timestamp")
        self.log.info("    CB=%.2f  BN=%.2f  Premium=%.4f%%",
                      cb_price, bn_price, premium)

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        # Snapshot-only — no historical backfill via API
        self._download_premium_snapshot()

    def download_incremental(self):
        self._download_premium_snapshot()


if __name__ == "__main__":
    CoinbasePremiumDownloader.main()
