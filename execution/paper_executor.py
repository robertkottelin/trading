"""Paper trading executor — simulates trades without the dYdX SDK.

Uses only the Indexer REST API (read-only, no auth) to fetch prices and
portfolio state.  Safe to run anywhere, no mnemonic required.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from execution.risk_manager import RiskManager

log = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
TAKER_FEE_BPS = 5  # 0.05%


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PaperExecutor:
    """Synchronous paper-trade executor.  No SDK dependency."""

    def __init__(self, config: dict | None = None):
        full_cfg = config or _load_config()
        self.cfg = full_cfg.get("execution", {})
        self.base_url = self.cfg.get(
            "mainnet_rest_indexer", "https://indexer.dydx.trade/v4"
        )
        self.market = self.cfg.get("market", "BTC-USD")
        self.state_dir = self.cfg.get("state_data_dir", "state_data")
        self.risk = RiskManager(self.cfg)
        os.makedirs(self.state_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_decision(self, decision: dict) -> dict:
        """Run paper trade for *decision*.  Returns the trade record."""
        portfolio = self._fetch_portfolio()
        price = self._fetch_current_price()

        # Log decision
        self._append_jsonl("decisions.jsonl", {
            "timestamp": _ts(),
            **decision,
        })

        # Risk validation
        passed, reason = self.risk.validate_decision(decision, portfolio)
        if not passed:
            record = self._rejection_record(decision, reason)
            self._append_jsonl("trades.jsonl", record)
            log.info("Paper trade rejected: %s", reason)
            return record

        # Simulate fill
        record = self._simulate_fill(decision, portfolio, price)
        if record.get("size_btc", 0) < 0.001:
            reject = self._rejection_record(
                decision, "computed position size below 0.001 BTC minimum"
            )
            self._append_jsonl("trades.jsonl", reject)
            log.info("Paper trade rejected: size too small")
            return reject
        self._append_jsonl("trades.jsonl", record)

        # Portfolio snapshot
        self._write_portfolio_snapshot(portfolio)

        log.info(
            "Paper %s %s %.4f BTC @ $%,.2f  (TP $%,.2f / SL $%,.2f)",
            record["action"],
            record["direction"],
            record["size_btc"],
            record["fill_price"],
            record.get("take_profit", 0),
            record.get("stop_loss", 0),
        )
        return record

    # ------------------------------------------------------------------
    # Indexer REST helpers
    # ------------------------------------------------------------------

    def _api_get(self, endpoint: str) -> dict | None:
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            log.warning("Indexer request failed (%s): %s", endpoint, e)
            return None

    def _fetch_current_price(self) -> float:
        """Get latest BTC-USD price from Indexer candles."""
        data = self._api_get(
            f"/candles/perpetualMarkets/{self.market}"
            "?resolution=1MIN&limit=1"
        )
        if data and "candles" in data and data["candles"]:
            return float(data["candles"][0]["close"])
        raise RuntimeError("Could not fetch current BTC price from Indexer")

    def _fetch_portfolio(self) -> dict:
        """Build portfolio dict from Indexer REST (same data as portfolio_reader)."""
        address = os.environ.get("ADDRESS", "")
        portfolio = {
            "equity": 0.0,
            "free_collateral": 0.0,
            "positions": [],
        }
        if not address:
            log.warning("ADDRESS not set — using zero-equity portfolio for paper mode")
            return portfolio

        sub = self._api_get(f"/addresses/{address}/subaccountNumber/0")
        if sub and "subaccount" in sub:
            acct = sub["subaccount"]
            portfolio["equity"] = float(acct.get("equity", 0))
            portfolio["free_collateral"] = float(acct.get("freeCollateral", 0))

        pos = self._api_get(
            f"/perpetualPositions?address={address}&subaccountNumber=0&status=OPEN"
        )
        if pos and "positions" in pos:
            for p in pos["positions"]:
                portfolio["positions"].append({
                    "market": p.get("market", ""),
                    "side": p.get("side", ""),
                    "size": p.get("size", "0"),
                })

        return portfolio

    # ------------------------------------------------------------------
    # Simulation logic
    # ------------------------------------------------------------------

    def _simulate_fill(
        self, decision: dict, portfolio: dict, market_price: float
    ) -> dict:
        equity = portfolio["equity"]
        size_pct = decision.get("position_size_pct", 0.05)
        entry_price = market_price  # paper fills at market
        max_btc = self.cfg.get("max_position_size_btc", 0.05)

        size_btc = (equity * size_pct) / entry_price if entry_price > 0 else 0
        size_btc = min(size_btc, max_btc)
        size_btc = round(size_btc, 3)  # dYdX BTC-USD minimum tick
        if size_btc < 0.001:
            size_btc = 0  # signal caller to reject — too small to trade

        notional = size_btc * entry_price
        fee = notional * (TAKER_FEE_BPS / 10_000)
        direction = decision.get("direction", "LONG")

        return {
            "timestamp": _ts(),
            "action": "ENTRY",
            "direction": direction,
            "side": "BUY" if direction == "LONG" else "SELL",
            "size_btc": size_btc,
            "entry_price": decision.get("entry_price", entry_price),
            "fill_price": entry_price,
            "take_profit": decision.get("take_profit", 0),
            "stop_loss": decision.get("stop_loss", 0),
            "duration_minutes": decision.get("duration_minutes", 60),
            "confidence": decision.get("confidence", 0),
            "notional_usd": round(notional, 2),
            "fee_usd": round(fee, 3),
            "equity_at_entry": equity,
            "mode": "paper",
            "status": "FILLED",
        }

    def _rejection_record(self, decision: dict, reason: str) -> dict:
        return {
            "timestamp": _ts(),
            "action": "REJECTED",
            "direction": decision.get("direction", "NO_TRADE"),
            "confidence": decision.get("confidence", 0),
            "rejection_reason": reason,
            "mode": "paper",
            "status": "REJECTED",
        }

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _append_jsonl(self, filename: str, record: dict):
        path = os.path.join(self.state_dir, filename)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _write_portfolio_snapshot(self, portfolio: dict):
        equity = portfolio.get("equity", 0)
        free = portfolio.get("free_collateral", 0)
        margin_pct = ((equity - free) / equity * 100) if equity > 0 else 0
        snapshot = {
            "timestamp": _ts(),
            "equity": equity,
            "free_collateral": free,
            "margin_pct": round(margin_pct, 2),
            "positions": portfolio.get("positions", []),
        }
        self._append_jsonl("portfolio.jsonl", snapshot)
