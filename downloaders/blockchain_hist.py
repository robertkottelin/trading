"""
Blockchain.com Charts API — BTC on-chain metrics (daily).

FREE, no auth. Rate limit: 1 req/10s.
Provides: active addresses, transaction count/volume, hash rate,
miners revenue, fees, UTXO set stats.
"""

import time
import pandas as pd
from datetime import datetime, timezone

from downloaders.base import BaseDownloader


class BlockchainDownloader(BaseDownloader):
    name = "blockchain"

    # Charts to download — {output_col: chart_name}
    CHARTS = {
        "n_unique_addresses": "n-unique-addresses",
        "n_transactions": "n-transactions",
        "estimated_transaction_volume_usd": "estimated-transaction-volume-usd",
        "hash_rate": "hash-rate",
        "difficulty": "difficulty",
        "miners_revenue": "miners-revenue",
        "transaction_fees_usd": "transaction-fees-usd",
        "avg_block_size": "avg-block-size",
        "n_transactions_per_block": "n-transactions-per-block",
        "median_confirmation_time": "median-confirmation-time",
        "mempool_size": "mempool-size",
        "utxo_count": "utxo-count",
        "cost_per_transaction": "cost-per-transaction",
        "market_price": "market-price",
        "total_bitcoins": "total-bitcoins",
        "output_volume": "output-volume",
    }

    def __init__(self, full=False):
        super().__init__(full=full)
        self.base_url = "https://api.blockchain.info/charts"
        self.delay = self.cfg.get("rate_limit_delay", 10.0)  # 1 req/10s

    def _download_chart(self, chart_name, col_name, timespan="all"):
        """Download a single chart metric."""
        url = f"{self.base_url}/{chart_name}"
        params = {"timespan": timespan, "format": "json", "sampled": "true"}
        try:
            resp = self._http_get(url, params, delay=self.delay, timeout=60)
            data = resp.json()
        except Exception as e:
            self.log.warning("    %s failed: %s", chart_name, e)
            return pd.Series(dtype=float, name=col_name)

        values = data.get("values", [])
        if not values:
            return pd.Series(dtype=float, name=col_name)

        dates = []
        vals = []
        for v in values:
            ts = v.get("x", 0)
            dates.append(datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"))
            vals.append(v.get("y", 0))

        return pd.Series(vals, index=pd.Index(dates, name="date"), name=col_name)

    def _download_all_charts(self):
        """Download all BTC on-chain metrics into one wide CSV."""
        self.log.info("  Downloading blockchain.com on-chain metrics (%d charts)...",
                      len(self.CHARTS))

        all_series = []
        for col_name, chart_name in self.CHARTS.items():
            self.log.info("    Downloading %s...", chart_name)
            s = self._download_chart(chart_name, col_name)
            if not s.empty:
                all_series.append(s)
                self.log.info("      %s: %d data points", chart_name, len(s))
            else:
                self.log.warning("      %s: no data", chart_name)

        if not all_series:
            self.log.error("  No blockchain data downloaded")
            return

        df = pd.concat(all_series, axis=1)
        df.index.name = "date"
        df = df.reset_index()

        self._save_csv(df, "blockchain_onchain.csv", "blockchain.com on-chain",
                       sort_by="date", dedup_col="date")

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        self._download_all_charts()

    def download_incremental(self):
        # Re-fetch all — each chart is a single request returning full history.
        # With 16 charts at 10s rate limit = ~3 min total. Not worth optimizing.
        self._download_all_charts()


if __name__ == "__main__":
    BlockchainDownloader.main()
