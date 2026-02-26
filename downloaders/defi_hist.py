"""
DeFi data — DefiLlama stablecoin supply + DeFi TVL.

All free, no auth required.
"""

import pandas as pd
from datetime import datetime, timezone

from downloaders.base import BaseDownloader


class DefiDownloader(BaseDownloader):
    name = "defi"

    def __init__(self, full=False, **kwargs):
        super().__init__(full=full, **kwargs)
        self.stablecoin_url = self.cfg.get("stablecoin_url",
                                           "https://stablecoins.llama.fi/stablecoins")
        self.tvl_url = self.cfg.get("tvl_url",
                                    "https://api.llama.fi/v2/historicalChainTvl")
        self.delay = self.cfg.get("rate_limit_delay", 0.5)

    # ---- Stablecoin Supply ------------------------------------------------

    def _download_stablecoin_supply(self):
        """Download stablecoin supply data from DefiLlama."""
        self.log.info("  Downloading stablecoin supply...")
        params = {"includePrices": "true"}
        try:
            resp = self._http_get(self.stablecoin_url, params, delay=self.delay)
            body = resp.json()
        except Exception as e:
            self.log.error("  Stablecoin supply failed: %s", e)
            return

        peggedAssets = body.get("peggedAssets", [])
        if not peggedAssets:
            self.log.warning("  No stablecoin data returned")
            return

        # Extract top stablecoins by market cap
        top_stables = ["USDT", "USDC", "DAI", "USDE", "PYUSD", "FRAX", "FDUSD", "TUSD"]
        rows = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

        total_mcap = 0
        for asset in peggedAssets:
            symbol = asset.get("symbol", "")
            chains = asset.get("chainCirculating", {})
            mcap = 0
            for chain_data in chains.values():
                current = chain_data.get("current", {})
                mcap += current.get("peggedUSD", 0)
            total_mcap += mcap

            if symbol in top_stables:
                rows.append({
                    "timestamp": now,
                    "symbol": symbol,
                    "name": asset.get("name", ""),
                    "market_cap": mcap,
                    "peg_type": asset.get("pegType", ""),
                })

        # Add total
        rows.append({
            "timestamp": now,
            "symbol": "TOTAL",
            "name": "Total Stablecoins",
            "market_cap": total_mcap,
            "peg_type": "",
        })

        df = pd.DataFrame(rows)
        self._append_csv(df, "defi_stablecoin_supply.csv",
                         sort_by="timestamp", dedup_col="timestamp")

    # ---- Historical Stablecoin Charts -------------------------------------

    def _download_stablecoin_history(self):
        """Download historical stablecoin supply (time series)."""
        self.log.info("  Downloading historical stablecoin supply...")
        # Use the stablecoincharts endpoint for total supply over time
        url = "https://stablecoins.llama.fi/stablecoincharts/all"
        params = {"stablecoin": 1}  # 1 = USDT (largest)
        try:
            resp = self._http_get(url, params, delay=self.delay)
            data = resp.json()
        except Exception as e:
            self.log.warning("  Historical stablecoin charts failed: %s", e)
            return

        if not data:
            return

        rows = []
        for d in data:
            ts = d.get("date", 0)
            supply = d.get("totalCirculating", {})
            rows.append({
                "date": datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d"),
                "total_circulating_usd": supply.get("peggedUSD", 0),
            })

        if rows:
            df = pd.DataFrame(rows)
            self._save_csv(df, "defi_stablecoin_history.csv",
                           "historical stablecoin supply",
                           sort_by="date", dedup_col="date")

    # ---- DeFi TVL ---------------------------------------------------------

    def _download_tvl(self):
        """Download historical total DeFi TVL from DefiLlama."""
        self.log.info("  Downloading DeFi TVL history...")
        try:
            resp = self._http_get(self.tvl_url, {}, delay=self.delay)
            data = resp.json()
        except Exception as e:
            self.log.error("  DeFi TVL failed: %s", e)
            return

        if not data:
            self.log.warning("  No TVL data returned")
            return

        rows = []
        for d in data:
            ts = d.get("date", 0)
            rows.append({
                "date": datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d"),
                "tvl": d.get("tvl", 0),
            })

        df = pd.DataFrame(rows)
        self._save_csv(df, "defi_tvl.csv", "DeFi TVL",
                       sort_by="date", dedup_col="date")

    # ---- Chain-specific TVL -----------------------------------------------

    def _download_chain_tvl(self):
        """Download TVL for key chains (Ethereum, BSC, etc.)."""
        self.log.info("  Downloading per-chain TVL...")
        chains = ["Ethereum", "BSC", "Solana", "Arbitrum", "Base", "Polygon",
                  "OP Mainnet", "Avalanche"]
        all_rows = []

        for chain in chains:
            url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
            try:
                resp = self._http_get(url, {}, delay=self.delay)
                data = resp.json()
            except Exception as e:
                self.log.warning("  Chain TVL for %s failed: %s", chain, e)
                continue

            for d in data:
                ts = d.get("date", 0)
                all_rows.append({
                    "date": datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d"),
                    "chain": chain,
                    "tvl": d.get("tvl", 0),
                })

        if all_rows:
            df = pd.DataFrame(all_rows)
            # Pivot: one column per chain
            pivot = df.pivot_table(index="date", columns="chain", values="tvl", aggfunc="last")
            pivot.columns = [f"tvl_{c.lower()}" for c in pivot.columns]
            pivot = pivot.reset_index()
            self._save_csv(pivot, "defi_chain_tvl.csv", "per-chain TVL",
                           sort_by="date", dedup_col="date")

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        self._download_stablecoin_supply()
        self._download_stablecoin_history()
        self._download_tvl()
        self._download_chain_tvl()

    def download_incremental(self):
        # All cheap single-request endpoints — just re-fetch
        self.download_all()


if __name__ == "__main__":
    DefiDownloader.main()
