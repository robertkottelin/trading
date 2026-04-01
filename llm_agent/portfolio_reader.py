"""dYdX v4 Portfolio Reader — queries Indexer REST API (read-only, no auth).

Fetches current equity, open positions, and recent fills from dYdX v4.
"""

import logging
import os

import requests

log = logging.getLogger(__name__)

MAINNET_URL = "https://indexer.dydx.trade/v4"
TESTNET_URL = "https://indexer.v4testnet.dydx.exchange/v4"
TIMEOUT = 15  # seconds


def _get_address(network: str = "testnet") -> str:
    """Get the dYdX address from environment based on network."""
    env_var = "TEST_ADDRESS" if network == "testnet" else "ADDRESS"
    addr = os.environ.get(env_var, "")
    if not addr:
        # Fallback: try the other var
        fallback = "ADDRESS" if network == "testnet" else "TEST_ADDRESS"
        addr = os.environ.get(fallback, "")
    if not addr:
        raise ValueError(f"{env_var} not set in environment / .env")
    return addr


def _api_get(endpoint: str, base_url: str = MAINNET_URL) -> dict | None:
    """Make a GET request to the dYdX Indexer API."""
    url = f"{base_url}{endpoint}"
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        log.warning("dYdX API request failed (%s): %s", endpoint, e)
        return None


def get_portfolio(network: str = "testnet") -> str:
    """Fetch portfolio state and return formatted text for LLM prompt.

    Args:
        network: "testnet" or "mainnet" — selects Indexer URL and address env var.

    Returns:
        Multi-line text summary of portfolio equity, positions, and recent fills.
    """
    base_url = TESTNET_URL if network == "testnet" else MAINNET_URL
    address = _get_address(network)
    lines = [f"PORTFOLIO STATE (dYdX v4 — {network}):"]

    # Subaccount info (equity, free collateral)
    sub = _api_get(f"/addresses/{address}/subaccountNumber/0", base_url)
    if sub and "subaccount" in sub:
        acct = sub["subaccount"]
        equity = acct.get("equity")
        free_collateral = acct.get("freeCollateral")
        if equity is not None:
            lines.append(f"  Equity: ${float(equity):,.2f}")
        if free_collateral is not None:
            lines.append(f"  Free collateral: ${float(free_collateral):,.2f}")

        # Margin usage
        if equity and free_collateral:
            used = float(equity) - float(free_collateral)
            if float(equity) > 0:
                margin_pct = used / float(equity) * 100
                lines.append(f"  Margin used: ${used:,.2f} ({margin_pct:.1f}%)")
    else:
        lines.append("  Account info: unavailable")

    # Open positions
    positions = _api_get(
        f"/perpetualPositions?address={address}&subaccountNumber=0&status=OPEN",
        base_url,
    )
    if positions and "positions" in positions:
        pos_list = positions["positions"]
        if pos_list:
            lines.append(f"  Open positions ({len(pos_list)}):")
            for pos in pos_list:
                market = pos.get("market", "?")
                side = pos.get("side", "?")
                size = pos.get("size", "0")
                entry = pos.get("entryPrice", "?")
                pnl = pos.get("unrealizedPnl", "0")
                lines.append(
                    f"    {market} {side}: size={size}, "
                    f"entry=${float(entry):,.2f}, "
                    f"unrealizedPnL=${float(pnl):+,.2f}"
                )
        else:
            lines.append("  No open positions")
    else:
        lines.append("  Positions: unavailable")

    # Recent fills
    fills = _api_get(
        f"/fills?address={address}&subaccountNumber=0&limit=20",
        base_url,
    )
    if fills and "fills" in fills:
        fill_list = fills["fills"]
        if fill_list:
            lines.append(f"  Recent fills (last {len(fill_list)}):")
            for fill in fill_list[:10]:  # Show max 10 in summary
                ts = fill.get("createdAt", "?")[:19]
                side = fill.get("side", "?")
                size = fill.get("size", "?")
                price = fill.get("price", "?")
                lines.append(f"    {ts} {side} {size} BTC @ ${float(price):,.2f}")
        else:
            lines.append("  No recent fills")
    else:
        lines.append("  Recent fills: unavailable")

    return "\n".join(lines)
