"""On-chain features from blockchain.com + BTC network data (~18 features).

Sources: blockchain_onchain.csv, btc_network_mining.csv, btc_network_lightning.csv
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_daily, rolling_zscore


def build_onchain_features(grid: pd.DataFrame) -> pd.DataFrame:
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    onchain = load_csv("blockchain_onchain.csv")

    # Align all needed columns to grid (daily, no lag — on-chain data is public)
    cols = ["n_unique_addresses", "n_transactions",
            "estimated_transaction_volume_usd", "hash_rate",
            "difficulty", "miners_revenue", "mempool_size",
            "transaction_fees_usd", "utxo_count"]

    oc = align_daily(onchain, gms, "date", cols, "oc_", lag_days=0)

    # Active addresses + 7d change
    result["oc_active_addresses"] = oc["oc_n_unique_addresses"]
    result["oc_addresses_change_7d"] = (
        result["oc_active_addresses"].pct_change(7 * 288).astype(np.float32)
    )

    # Transaction volume + 7d change
    result["oc_tx_volume_usd"] = oc["oc_estimated_transaction_volume_usd"]
    result["oc_tx_volume_change_7d"] = (
        result["oc_tx_volume_usd"].pct_change(7 * 288).astype(np.float32)
    )

    # Hash rate + 14d change
    result["oc_hash_rate"] = oc["oc_hash_rate"]
    result["oc_hash_rate_change_14d"] = (
        result["oc_hash_rate"].pct_change(14 * 288).astype(np.float32)
    )

    # Difficulty change (daily data ffilled to 5m grid: 288 bars per day)
    result["oc_difficulty_change"] = (
        oc["oc_difficulty"].pct_change(288).astype(np.float32)
    )

    # Miner revenue + z-score
    result["oc_miner_revenue"] = oc["oc_miners_revenue"]
    result["oc_miner_revenue_zscore"] = rolling_zscore(
        result["oc_miner_revenue"], 30 * 288)

    # Mempool size
    result["oc_mempool_size"] = oc["oc_mempool_size"]

    # Transaction fees
    result["oc_tx_fees"] = oc["oc_transaction_fees_usd"]

    # UTXO count change (daily data ffilled to 5m grid: 288 bars per day)
    result["oc_utxo_change"] = (
        oc["oc_utxo_count"].pct_change(288).astype(np.float32)
    )

    # --- BTC network mining (daily, 1.1K rows) ---
    try:
        mining = load_csv("btc_network_mining.csv")
        mining_cols = [c for c in ["avg_hashrate", "difficulty", "adjustment_pct"]
                       if c in mining.columns]
        if mining_cols:
            m = align_daily(mining, gms, "date", mining_cols,
                            "net_", lag_days=1)
            if "net_avg_hashrate" in m.columns:
                result["net_hashrate"] = m["net_avg_hashrate"]
                result["net_hashrate_change_14d"] = (
                    result["net_hashrate"].pct_change(14 * 288).astype(np.float32)
                )
            if "net_difficulty" in m.columns:
                result["net_difficulty_change"] = (
                    m["net_difficulty"].pct_change(288).astype(np.float32)
                )
            if "net_adjustment_pct" in m.columns:
                result["net_difficulty_adj_pct"] = m["net_adjustment_pct"]
    except (FileNotFoundError, KeyError):
        pass

    # --- BTC Lightning network (daily, 800 rows) ---
    try:
        ln = load_csv("btc_network_lightning.csv")
        ln_cols = [c for c in ["channel_count", "total_capacity", "node_count"]
                   if c in ln.columns]
        if ln_cols:
            l = align_daily(ln, gms, "date", ln_cols,
                            "net_ln_", lag_days=1)
            if "net_ln_total_capacity" in l.columns:
                result["net_ln_capacity"] = l["net_ln_total_capacity"]
                result["net_ln_capacity_change_7d"] = (
                    result["net_ln_capacity"].pct_change(7 * 288).astype(np.float32)
                )
            if "net_ln_channel_count" in l.columns:
                result["net_ln_channels"] = l["net_ln_channel_count"]
    except (FileNotFoundError, KeyError):
        pass

    return result
