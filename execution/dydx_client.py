"""Async dYdX v4 SDK wrapper — connect, wallet, portfolio, market data.

Requires: pip install dydx-v4-client>=1.1.6
Environment: DYDX_MNEMONIC (mainnet) or DYDX_TEST_MNEMONIC (testnet) in .env
"""

import json
import logging
import os
from datetime import datetime, timezone

import yaml

from dydx4.clients import CompositeClient, Subaccount
from dydx4.clients.constants import BECH32_PREFIX, Network
from dydx4.clients.helpers.chain_helpers import (
    ORDER_FLAGS_SHORT_TERM,
    ORDER_FLAGS_LONG_TERM,
    ORDER_FLAGS_CONDITIONAL,
)

log = logging.getLogger(__name__)


def _patch_post_for_testnet():
    """Monkey-patch dydx4 Post.send_message to use testnet config.

    The SDK hardcodes NetworkConfig.fetch_mainnet() in send_message,
    making testnet transactions fail with 'account not found'.
    """
    from dydx4.clients.modules.post import Post
    from dydx4.chain.aerial.config import NetworkConfig
    from dydx4.chain.aerial.client import LedgerClient
    from dydx4.chain.aerial.tx import Transaction
    from dydx4.chain.aerial.client.utils import prepare_and_broadcast_basic_transaction
    from dydx4.chain.aerial.tx_helpers import SubmittedTx

    _original_send = Post.send_message

    def _testnet_send_message(self, subaccount, msg, zeroFee=False, broadcast_mode=None):
        wallet = subaccount.wallet
        network = NetworkConfig(
            chain_id="dydx-testnet-4",
            url="grpc+https://test-dydx-grpc.kingnodes.com:443",
            fee_minimum_gas_price=4630550000000000,
            fee_denomination="adv4tnt",
            staking_denomination="dv4tnt",
            faucet_url="https://faucet.v4testnet.dydx.exchange",
        )
        ledger = LedgerClient(network)
        tx = Transaction()
        tx.add_message(msg)
        gas_limit = 0 if zeroFee else None
        return prepare_and_broadcast_basic_transaction(
            client=ledger,
            tx=tx,
            sender=wallet,
            gas_limit=gas_limit,
            memo=None,
            broadcast_mode=broadcast_mode
            if (broadcast_mode is not None)
            else self.default_broadcast_mode(msg),
            fee=0 if zeroFee else None,
        )

    Post.send_message = _testnet_send_message

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class DydxClient:
    """Async wrapper around the dYdX v4 Python SDK."""

    def __init__(self, config: dict | None = None):
        full_cfg = config or _load_config()
        # Accept either full config (with 'execution' key) or exec section directly
        self.cfg = full_cfg.get("execution", full_cfg)
        self.network_name = self.cfg.get("network", "testnet")
        self.market_name = self.cfg.get("market", "BTC-USD")
        self.subaccount_number = self.cfg.get("subaccount_number", 0)
        self.state_dir = self.cfg.get("state_data_dir", "state_data")
        os.makedirs(self.state_dir, exist_ok=True)

        self.client: CompositeClient | None = None
        self.subaccount: Subaccount | None = None
        self.market: dict | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self):
        """Initialize SDK client, wallet, and market info.

        Note: dydx4 SDK is synchronous but we keep the async interface
        for compatibility with the executor's asyncio.run() pattern.
        """
        network = self._resolve_network()
        if self.network_name == "testnet":
            _patch_post_for_testnet()
        self.client = CompositeClient(network)

        mnemonic = self._get_mnemonic()
        self.subaccount = Subaccount.from_mnemonic(mnemonic)

        # Load market configuration
        markets_response = self.client.indexer_client.markets.get_perpetual_markets(
            self.market_name
        )
        markets_data = self._unwrap(markets_response)
        market_data = markets_data["markets"][self.market_name]
        self.market = market_data

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

    @staticmethod
    def _unwrap(response) -> dict:
        """Unwrap SDK Response object to get data dict."""
        if hasattr(response, 'data'):
            return response.data
        return response

    # ------------------------------------------------------------------
    # Data queries
    # ------------------------------------------------------------------

    async def get_portfolio_state(self) -> dict:
        """Fetch equity, free collateral, positions, and recent fills."""
        assert self.client is not None, "call connect() first"

        address = self.subaccount.address
        sub_num = self.subaccount_number

        sub_response = self._unwrap(self.client.indexer_client.account.get_subaccount(
            address, sub_num
        ))
        acct = sub_response.get("subaccount", {})
        equity = float(acct.get("equity", 0))
        free_collateral = float(acct.get("freeCollateral", 0))

        pos_response = self._unwrap(self.client.indexer_client.account.get_subaccount_perpetual_positions(
            address, sub_num, status="OPEN"
        ))
        positions = []
        for p in pos_response.get("positions", []):
            positions.append({
                "market": p.get("market", ""),
                "side": p.get("side", ""),
                "size": p.get("size", "0"),
                "entry_price": p.get("entryPrice", "0"),
                "unrealized_pnl": p.get("unrealizedPnl", "0"),
            })

        fills_response = self._unwrap(self.client.indexer_client.account.get_subaccount_fills(
            address, sub_num, limit=20
        ))
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

        candles = self._unwrap(self.client.indexer_client.markets.get_perpetual_market_candles(
            self.market_name, resolution="1MIN", limit=1
        ))
        if candles and "candles" in candles and candles["candles"]:
            return float(candles["candles"][0]["close"])
        raise RuntimeError("Could not fetch current price from Indexer")

    async def get_latest_block_height(self) -> int:
        """Get the latest block height for short-term order expiry."""
        assert self.client is not None, "call connect() first"
        block = self.client.get_current_block()
        return block

    async def get_open_orders(self) -> list[dict]:
        """Fetch all open and untriggered orders for this subaccount."""
        assert self.client is not None, "call connect() first"

        address = self.subaccount.address
        sub_num = self.subaccount_number

        orders = []
        for status in ("OPEN", "UNTRIGGERED"):
            orders_data = self._unwrap(self.client.indexer_client.account.get_subaccount_orders(
                address, sub_num, status=status, limit=100,
                return_latest_orders=True,
            ))
            raw = orders_data if isinstance(orders_data, list) else orders_data.get("orders", [])
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
        """Cancel an open order."""
        assert self.client is not None, "call connect() first"

        # Determine numeric order flags
        flags = ORDER_FLAGS_CONDITIONAL
        if order_flags == "SHORT_TERM" or good_til_block is not None:
            flags = ORDER_FLAGS_SHORT_TERM
        elif order_flags == "LONG_TERM":
            flags = ORDER_FLAGS_LONG_TERM

        # For conditional orders, the cancellation goodTilBlockTime must be >=
        # the order's. Use a safe far-future value (30 days) for cancellation.
        gtt_seconds = 0
        if flags == ORDER_FLAGS_CONDITIONAL:
            gtt_seconds = 30 * 86400  # 30 days — always >= any order expiry
        elif good_til_block_time:
            try:
                from datetime import datetime, timezone
                expiry = datetime.fromisoformat(good_til_block_time.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                gtt_seconds = max(0, int((expiry - now).total_seconds()))
            except (ValueError, TypeError):
                gtt_seconds = 86400  # default 24h

        tx = self.client.cancel_order(
            self.subaccount,
            client_id=client_id,
            market=self.market_name,
            order_flags=flags,
            good_til_block=good_til_block or 0,
            good_til_time_in_seconds=gtt_seconds,
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
        # Default testnet() has broken imperator.co indexer — use official endpoints
        from dydx4.clients.constants import IndexerConfig, ValidatorConfig
        return Network(
            env="testnet",
            validator_config=ValidatorConfig(
                grpc_endpoint="test-dydx-grpc.kingnodes.com:443",
                chain_id="dydx-testnet-4",
                ssl_enabled=True,
            ),
            indexer_config=IndexerConfig(
                rest_endpoint="https://indexer.v4testnet.dydx.exchange",
                websocket_endpoint="wss://indexer.v4testnet.dydx.exchange/v4/ws",
            ),
            faucet_endpoint="https://faucet.v4testnet.dydx.exchange",
        )

    def _get_mnemonic(self) -> str:
        if self.network_name == "mainnet":
            mnemonic = os.environ.get("DYDX_MNEMONIC", "")
        else:
            mnemonic = os.environ.get("DYDX_TEST_MNEMONIC", "")
        if not mnemonic:
            env_var = "DYDX_MNEMONIC" if self.network_name == "mainnet" else "DYDX_TEST_MNEMONIC"
            raise ValueError(f"{env_var} not set in environment / .env")
        return mnemonic
