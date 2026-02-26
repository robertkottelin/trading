"""
BTC on-chain / network data from mempool.space — hashrate, difficulty, fees, mempool, Lightning.

All free, no auth required.
"""

import pandas as pd
from datetime import datetime, timezone

from downloaders.base import BaseDownloader


class BtcNetworkDownloader(BaseDownloader):
    name = "btc_network"

    def __init__(self, full=False, **kwargs):
        super().__init__(full=full, **kwargs)
        self.base_url = self.cfg.get("mempool_base_url", "https://mempool.space/api")
        self.delay = self.cfg.get("rate_limit_delay", 0.5)

    # ---- Mining: Hashrate + Difficulty ------------------------------------

    def _download_mining(self):
        """Download hashrate and difficulty adjustment history."""
        self.log.info("  Downloading hashrate (3y)...")

        # Hashrate
        url = f"{self.base_url}/v1/mining/hashrate/3y"
        try:
            resp = self._http_get(url, {}, delay=self.delay)
            body = resp.json()
        except Exception as e:
            self.log.error("  Hashrate download failed: %s", e)
            return

        hashrates = body.get("hashrates", [])
        difficulty = body.get("difficulty", [])
        current_hashrate = body.get("currentHashrate", 0)
        current_difficulty = body.get("currentDifficulty", 0)

        rows = []
        for h in hashrates:
            ts = h.get("timestamp", 0)
            rows.append({
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d"),
                "timestamp_unix": ts,
                "avg_hashrate": float(h.get("avgHashrate", 0)),
            })

        df_hash = pd.DataFrame(rows) if rows else pd.DataFrame()

        # Difficulty adjustments
        self.log.info("  Downloading difficulty adjustments...")
        url_diff = f"{self.base_url}/v1/mining/difficulty-adjustments/3y"
        try:
            resp = self._http_get(url_diff, {}, delay=self.delay)
            diff_data = resp.json()
        except Exception as e:
            self.log.warning("  Difficulty adjustments failed: %s", e)
            diff_data = []

        if diff_data:
            diff_rows = []
            for d in diff_data:
                # Each entry is [timestamp, height, difficulty, adjustment]
                if isinstance(d, list) and len(d) >= 4:
                    diff_rows.append({
                        "date": datetime.fromtimestamp(d[0], tz=timezone.utc).strftime("%Y-%m-%d"),
                        "block_height": int(d[1]),
                        "difficulty": float(d[2]),
                        "adjustment_pct": float(d[3]),
                    })
            df_diff = pd.DataFrame(diff_rows) if diff_rows else pd.DataFrame()
        else:
            df_diff = pd.DataFrame()

        # Merge hashrate + difficulty on date
        if not df_hash.empty and not df_diff.empty:
            df = pd.merge(df_hash, df_diff[["date", "difficulty", "adjustment_pct"]],
                          on="date", how="outer")
            df["difficulty"] = df["difficulty"].ffill()
            df["adjustment_pct"] = df["adjustment_pct"].fillna(0)
        elif not df_hash.empty:
            df = df_hash
            df["difficulty"] = current_difficulty
            df["adjustment_pct"] = 0.0
        else:
            self.log.warning("  No mining data")
            return

        self._save_csv(df, "btc_network_mining.csv", "BTC mining (hashrate + difficulty)",
                       sort_by="date", dedup_col="date")

    # ---- Mempool + Fees (snapshot) ----------------------------------------

    def _download_mempool(self):
        """Snapshot of current mempool stats + fee estimates."""
        self.log.info("  Downloading mempool snapshot...")

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        row = {"timestamp": now}

        # Fee estimates
        url_fees = f"{self.base_url}/v1/fees/recommended"
        try:
            resp = self._http_get(url_fees, {}, delay=self.delay)
            fees = resp.json()
            row["fee_fastest"] = fees.get("fastestFee", 0)
            row["fee_half_hour"] = fees.get("halfHourFee", 0)
            row["fee_hour"] = fees.get("hourFee", 0)
            row["fee_economy"] = fees.get("economyFee", 0)
            row["fee_minimum"] = fees.get("minimumFee", 0)
        except Exception as e:
            self.log.warning("  Fee estimate failed: %s", e)

        # Mempool stats
        url_mempool = f"{self.base_url}/mempool"
        try:
            resp = self._http_get(url_mempool, {}, delay=self.delay)
            mp = resp.json()
            row["mempool_count"] = mp.get("count", 0)
            row["mempool_vsize"] = mp.get("vsize", 0)
            row["mempool_total_fee"] = mp.get("total_fee", 0)
        except Exception as e:
            self.log.warning("  Mempool stats failed: %s", e)

        df = pd.DataFrame([row])
        self._append_csv(df, "btc_network_mempool.csv",
                         sort_by="timestamp", dedup_col="timestamp")

    # ---- Lightning Network ------------------------------------------------

    def _download_lightning(self):
        """Download Lightning Network statistics (3y)."""
        self.log.info("  Downloading Lightning Network stats (3y)...")
        url = f"{self.base_url}/v1/lightning/statistics/3y"
        try:
            resp = self._http_get(url, {}, delay=self.delay)
            data = resp.json()
        except Exception as e:
            self.log.error("  Lightning stats failed: %s", e)
            return

        if not data:
            self.log.warning("  No Lightning data")
            return

        # data is a list of snapshots
        if isinstance(data, dict):
            data = data.get("latest", [data])

        rows = []
        for d in data:
            added = d.get("added", d.get("latest", d))
            if isinstance(added, str):
                continue
            ts = added if isinstance(added, dict) else d
            rows.append({
                "date": datetime.fromtimestamp(
                    ts.get("added", ts.get("timestamp", 0)),
                    tz=timezone.utc
                ).strftime("%Y-%m-%d") if ts.get("added", ts.get("timestamp")) else "",
                "channel_count": ts.get("channel_count", 0),
                "node_count": ts.get("node_count", 0),
                "total_capacity": ts.get("total_capacity", 0),
                "avg_capacity": ts.get("avg_capacity", 0),
                "med_capacity": ts.get("med_capacity", 0),
            })

        if not rows:
            self.log.warning("  Could not parse Lightning data")
            return

        df = pd.DataFrame(rows)
        df = df[df["date"] != ""]
        self._save_csv(df, "btc_network_lightning.csv", "Lightning Network",
                       sort_by="date", dedup_col="date")

    # ---- Main entry points ------------------------------------------------

    def download_all(self):
        self._download_mining()
        self._download_mempool()
        self._download_lightning()

    def download_incremental(self):
        # Mining + Lightning: re-fetch (cheap, single request for 3y)
        # Mempool: append snapshot
        self.download_all()


if __name__ == "__main__":
    BtcNetworkDownloader.main()
