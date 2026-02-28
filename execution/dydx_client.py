"""Async dYdX v4 SDK wrapper — connect, wallet, portfolio, market data.

Requires: pip install dydx-v4-client>=1.1.6
Environment: DYDX_MNEMONIC (mainnet) or DYDX_TEST_MNEMONIC (testnet) in .env
"""

import json
import logging
import os
from datetime import datetime, timezone

import yaml

from dydxv4.clients import CompositeClient, Subaccount
from dydxv4.clients.constants import BECH32_PREFIX, Network
from dydxv4.clients.helpers.chain_helpers import (
    ORDER_FLAGS_SHORT_TERM,
    ORDER_FLAGS_LONG_TERM,
    ORDER_FLAGS_CONDITIONAL,
)
from dydxv4.clients.node.market import Market

log = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class DydxClient:
    """Async wrapper around the dYdX v4 Python SDK."""

    def __init__(self, config: dict | None = None):
        full_cfg = config or _load_config()
        self.cfg = full_cfg.get("execution", {})
        self.network_name = self.cfg.get("network", "testnet")
        self.market_name = self.cfg.get("market", "BTC-USD")
        self.subaccount_number = self.cfg.get("subaccount_number", 0)
        self.state_dir = self.cfg.get("state_data_dir", "state_data")
        os.makedirs(self.state_dir, exist_ok=True)

        self.client: CompositeClient | None = None
        self.subaccount: Subaccount | None = None
        self.market: Market | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self):
        """Initialize SDK client, wallet, and market info."""
        network = self._resolve_network()
        self.client = await CompositeClient.create(network)

        mnemonic = self._get_mnemonic()
        from dydxv4.clients.helpers.chain_helpers import DYDX_MNEMONIC_TO_ADDRESS
        self.subaccount = Subaccount.from_mnemonic(mnemonic)

        # Load market configuration
        markets_response = await self.client.indexer_client.markets.get_perpetual_markets(
            self.market_name
        )
        market_data = markets_response["markets"][self.market_name]
        self.market = Market(market_data)

        log.info(
            "Connected to dYdX %s — address: %s",
            self.network_name,
            self.subaccount.address,
        )

    async def disconnect(self):
        """Clean up resources."""
        self.client = None
        self.subaccount = None
        self.market = None
        log.info("Disconnected from dYdX")

    # ------------------------------------------------------------------
    # Data queries
    # ------------------------------------------------------------------

    async def get_portfolio_state(self) -> dict:
        """Fetch equity, free collateral, positions, and recent fills."""
        assert self.client is not None, "call connect() first"

        address = self.subaccount.address
        sub_num = self.subaccount_number

        sub_response = await self.client.indexer_client.account.get_subaccount(
            address, sub_num
        )
        acct = sub_response.get("subaccount", {})
        equity = float(acct.get("equity", 0))
        free_collateral = float(acct.get("freeCollateral", 0))

        pos_response = await self.client.indexer_client.account.get_subaccount_perpetual_positions(
            address, sub_num, status="OPEN"
        )
        positions = []
        for p in pos_response.get("positions", []):
            positions.append({
                "market": p.get("market", ""),
                "side": p.get("side", ""),
                "size": p.get("size", "0"),
                "entry_price": p.get("entryPrice", "0"),
                "unrealized_pnl": p.get("unrealizedPnl", "0"),
            })

        fills_response = await self.client.indexer_client.account.get_subaccount_fills(
            address, sub_num, limit=20
        )
        fills = fills_response.get("fills", [])

        return {
            "equity": equity,
            "free_collateral": free_collateral,
            "positions": positions,
            "fills": fills,
        }

    async def get_current_price(self) -> float:
        """Get latest BTC-USD price from Indexer."""
        assert self.client is not None, "call connect() first"

        candles = await self.client.indexer_client.markets.get_perpetual_market_candles(
            self.market_name, resolution="1MIN", limit=1
        )
        if candles and "candles" in candles and candles["candles"]:
            price = float(candles["candles"][0]["close"])
            if price <= 0 or price != price:  # NaN check: NaN != NaN
                raise RuntimeError(f"Invalid price from Indexer: {price}")
            return price
        raise RuntimeError("Could not fetch current price from Indexer")

    async def get_latest_block_height(self) -> int:
        """Get the latest block height for short-term order expiry."""
        assert self.client is not None, "call connect() first"
        block = await self.client.validator_client.get.latest_block_height()
        return block

    async def get_open_orders(self) -> list[dict]:
        """Fetch all open orders for this subaccount."""
        assert self.client is not None, "call connect() first"

        address = self.subaccount.address
        sub_num = self.subaccount_number

        orders_response = await self.client.indexer_client.account.get_subaccount_orders(
            address, sub_num, status="OPEN", limit=100,
            return_latest_orders=True,
        )
        raw = orders_response if isinstance(orders_response, list) else orders_response.get("orders", [])
        orders = []
        for o in raw:
            orders.append({
                "order_id": o.get("id", ""),
                "client_id": o.get("clientId", ""),
                "market": o.get("ticker", ""),
                "side": o.get("side", ""),
                "size": o.get("size", "0"),
                "price": o.get("price", "0"),
                "type": o.get("type", ""),
                "status": o.get("status", ""),
                "order_flags": o.get("orderFlags", ""),
                "good_til_block": o.get("goodTilBlock"),
                "good_til_block_time": o.get("goodTilBlockTime"),
            })
        return orders

    async def cancel_order(self, client_id: int, order_flags: str,
                           good_til_block: int | None = None,
                           good_til_block_time: str | None = None):
        """Cancel an open order.

        For short-term orders, good_til_block is required.
        For conditional orders, good_til_block_time is required.
        """
        assert self.client is not None, "call connect() first"

        # Determine numeric order flags
        flags = ORDER_FLAGS_CONDITIONAL
        if order_flags == "SHORT_TERM" or good_til_block is not None:
            flags = ORDER_FLAGS_SHORT_TERM
        elif order_flags == "LONG_TERM":
            flags = ORDER_FLAGS_LONG_TERM

        market_info = await self.client.indexer_client.markets.get_perpetual_markets(
            self.market_name
        )
        clob_pair_id = int(market_info["markets"][self.market_name].get("clobPairId", 0))

        tx = await self.client.cancel_order(
            self.subaccount,
            client_id=client_id,
            order_flags=flags,
            clob_pair_id=clob_pair_id,
            good_til_block=good_til_block,
            good_til_block_time=good_til_block_time,
        )
        log.info("Cancel TX submitted for client_id=%d: %s", client_id, tx)
        return tx

    # ------------------------------------------------------------------
    # Portfolio snapshot logging
    # ------------------------------------------------------------------

    def write_portfolio_snapshot(self, state: dict):
        """Append a portfolio snapshot to state_data/portfolio.jsonl."""
        equity = state.get("equity", 0)
        free = state.get("free_collateral", 0)
        margin_pct = ((equity - free) / equity * 100) if equity > 0 else 0
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "equity": equity,
            "free_collateral": free,
            "margin_pct": round(margin_pct, 2),
            "positions": state.get("positions", []),
        }
        path = os.path.join(self.state_dir, "portfolio.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(snapshot) + "\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_network(self) -> Network:
        if self.network_name == "mainnet":
            return Network.mainnet()
        return Network.testnet()

    def _get_mnemonic(self) -> str:
        if self.network_name == "mainnet":
            mnemonic = os.environ.get("DYDX_MNEMONIC", "")
        else:
            mnemonic = os.environ.get("DYDX_TEST_MNEMONIC", "")
        if not mnemonic:
            env_var = "DYDX_MNEMONIC" if self.network_name == "mainnet" else "DYDX_TEST_MNEMONIC"
            raise ValueError(f"{env_var} not set in environment / .env")
        return mnemonic
