"""DeFi features: TVL + stablecoin supply (8 features).

Sources: defi_tvl.csv, defi_chain_tvl.csv, defi_stablecoin_history.csv
"""

import numpy as np
import pandas as pd
from features.alignment import load_csv, align_daily, rolling_zscore


def build_defi_features(grid: pd.DataFrame) -> pd.DataFrame:
    gms = grid["open_time_ms"]
    result = grid[["open_time_ms"]].copy()

    # --- Total TVL ---
    tvl = load_csv("defi_tvl.csv")
    tvl_aligned = align_daily(tvl, gms, "date", ["tvl"], "defi_", lag_days=0)
    result["defi_tvl"] = tvl_aligned["defi_tvl"]

    # 7d and 30d change
    result["defi_tvl_change_7d"] = (
        result["defi_tvl"].pct_change(7 * 288).astype(np.float32)
    )
    result["defi_tvl_change_30d"] = (
        result["defi_tvl"].pct_change(30 * 288).astype(np.float32)
    )

    # --- Chain TVL (ETH + SOL share) ---
    chain = load_csv("defi_chain_tvl.csv")
    chain_aligned = align_daily(chain, gms, "date",
                                ["tvl_ethereum", "tvl_solana"], "defi_",
                                lag_days=0)

    # ETH and SOL TVL as share of total
    result["defi_eth_tvl_share"] = (
        chain_aligned["defi_tvl_ethereum"] /
        result["defi_tvl"].replace(0, np.nan)
    ).astype(np.float32)
    result["defi_sol_tvl_share"] = (
        chain_aligned["defi_tvl_solana"] /
        result["defi_tvl"].replace(0, np.nan)
    ).astype(np.float32)

    # --- Stablecoin supply ---
    stable = load_csv("defi_stablecoin_history.csv")
    stable_aligned = align_daily(stable, gms, "date",
                                 ["total_circulating_usd"], "defi_",
                                 lag_days=0)

    result["defi_stable_supply"] = stable_aligned["defi_total_circulating_usd"]
    result["defi_stable_change_7d"] = (
        result["defi_stable_supply"].pct_change(7 * 288).astype(np.float32)
    )
    result["defi_stable_zscore"] = rolling_zscore(
        result["defi_stable_supply"], 30 * 288)

    return result
