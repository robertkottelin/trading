"""Live dYdX v4 order executor — places market entry + TP/SL orders.

Requires: dydx-v4-client SDK and a funded wallet mnemonic in .env.

Usage:
    python -m execution.dydx_executor             # live (uses config mode)
    python -m execution.dydx_executor --paper      # force paper mode
    python -m execution.dydx_executor --live       # force live mode
    python -m execution.dydx_executor --decision path/to/decision.json
"""

import argparse
import asyncio
import fcntl
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

from execution.risk_manager import RiskManager

log = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
DECISION_PATH = os.path.join(os.path.dirname(__file__), "..", "llm_agent", "decision.json")
EXECUTION_LOCK_PATH = os.path.join(os.path.dirname(__file__), "..", "state_data", "execution.lock")
MAX_CLIENT_ID = 2**31 - 1


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DydxExecutor:
    """Async executor that places entry + TP + SL orders on dYdX v4."""

    def __init__(self, client, risk_manager: RiskManager | None = None, config: dict | None = None):
        from execution.dydx_client import DydxClient
        self.dydx: DydxClient = client
        full_cfg = config or _load_config()
        self.cfg = full_cfg.get("execution", {})
        self.risk = risk_manager or RiskManager(self.cfg)
        self.state_dir = self.cfg.get("state_data_dir", "state_data")
        os.makedirs(self.state_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_decision(self, decision: dict | None = None) -> dict:
        """Execute a trading decision.  Returns the trade record.

        Acquires an exclusive file lock to prevent concurrent pipeline runs
        from placing duplicate orders.
        """
        os.makedirs(os.path.dirname(EXECUTION_LOCK_PATH), exist_ok=True)
        lock_fd = open(EXECUTION_LOCK_PATH, "a")
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                log.warning("Another execution is already in progress — skipping")
                return {
                    "timestamp": _ts(),
                    "action": "REJECTED",
                    "rejection_reason": "concurrent execution blocked by lock",
                    "mode": "live",
                    "status": "REJECTED",
                }

            try:
                return await self._execute_decision_locked(decision)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            lock_fd.close()

    async def _execute_decision_locked(self, decision: dict | None = None) -> dict:
        """Inner execution logic, called while holding the execution lock."""

        # 1. Load decision
        if decision is None:
            decision = self._load_decision_file()

        # 2. Log decision
        self._append_jsonl("decisions.jsonl", {"timestamp": _ts(), **decision})

        # 3. Fetch portfolio + snapshot
        portfolio = await self.dydx.get_portfolio_state()
        self.dydx.write_portfolio_snapshot(portfolio)

        # 4. Risk validation
        passed, reason = self.risk.validate_decision(decision, portfolio)
        if not passed:
            record = self._rejection_record(decision, reason)
            self._append_jsonl("trades.jsonl", record)
            log.info("Trade rejected: %s", reason)
            return record

        # 5. Build order parameters
        price = await self.dydx.get_current_price()
        order_params = self._build_order_params(decision, portfolio, price)

        # 5b. Reject if computed size is below dYdX minimum
        if order_params["size_btc"] < 0.001:
            record = self._rejection_record(
                decision, "computed position size below 0.001 BTC minimum"
            )
            self._append_jsonl("trades.jsonl", record)
            log.info("Trade rejected: size too small")
            return record

        # 6. Place entry (market) order
        entry_record = await self._place_entry_order(decision, order_params, portfolio)
        self._append_jsonl("trades.jsonl", entry_record)

        if entry_record["status"] == "FAILED":
            return entry_record

        if entry_record["status"] == "UNVERIFIED":
            # Fill wasn't confirmed in time — but the order may have filled on-chain.
            # Check for actual open position before deciding whether to place TP/SL.
            log.warning("Entry fill unverified — checking for actual position on-chain...")
            try:
                check_portfolio = await self.dydx.get_portfolio_state()
                target_market = self.cfg.get("market", "BTC-USD")
                has_position = any(
                    abs(float(p.get("size", 0))) > 0
                    for p in check_portfolio.get("positions", [])
                    if p.get("market") == target_market
                )
            except Exception as e:
                log.error("Position check failed: %s", e)
                has_position = False

            if has_position:
                log.warning("Position found on-chain despite unverified fill — placing TP/SL")
                entry_record["status"] = "FILLED_LATE"
                self._append_jsonl("trades.jsonl", {
                    "timestamp": _ts(),
                    "action": "FILL_LATE_DETECTED",
                    "reason": "position found on-chain after poll timeout",
                    "mode": "live",
                    "status": "INFO",
                })
                # Fall through to TP/SL placement below
            else:
                log.info("No position found on-chain — entry likely did not fill")
                self._append_jsonl("trades.jsonl", {
                    "timestamp": _ts(),
                    "action": "TP_SL_SKIPPED",
                    "reason": "entry fill unverified, no position on-chain",
                    "mode": "live",
                    "status": "ALERT",
                })
                return entry_record

        # 7-8. Place TP and SL conditional orders (with rollback on failure)
        await self._place_tp_sl_orders(decision, order_params, entry_record)

        # 9. Post-trade snapshot
        post_portfolio = await self.dydx.get_portfolio_state()
        self.dydx.write_portfolio_snapshot(post_portfolio)

        return entry_record

    async def cleanup_orphan_orders(self) -> int:
        """Cancel open orders that don't correspond to active positions,
        and cancel duplicate TP/SL orders for the same position.

        Returns the number of orders cancelled.
        """
        try:
            portfolio = await self.dydx.get_portfolio_state()
            open_orders = await self.dydx.get_open_orders()
        except Exception as e:
            log.error("Failed to fetch state for orphan cleanup: %s", e)
            return 0

        if not open_orders:
            log.info("Orphan cleanup: no open orders found")
            return 0

        # Markets with active positions — their TP/SL orders are legitimate
        active_markets = {p["market"] for p in portfolio.get("positions", [])}

        # --- Pass 1: Cancel orders in markets with no active position ---
        orphans = [o for o in open_orders if o["market"] not in active_markets]
        cancelled = 0

        if orphans:
            log.warning("Orphan cleanup: found %d orphan order(s) to cancel", len(orphans))
            for order in orphans:
                try:
                    client_id = int(order["client_id"]) if str(order["client_id"]).isdigit() else 0
                    await self.dydx.cancel_order(
                        client_id=client_id,
                        order_flags=order.get("order_flags", ""),
                        good_til_block=order.get("good_til_block"),
                        good_til_block_time=order.get("good_til_block_time"),
                    )
                    cancelled += 1
                    self._append_jsonl("trades.jsonl", {
                        "timestamp": _ts(),
                        "action": "ORPHAN_ORDER_CANCELLED",
                        "order_id": order["order_id"],
                        "market": order["market"],
                        "side": order["side"],
                        "size": order["size"],
                        "type": order["type"],
                        "mode": "live",
                        "status": "CANCELLED",
                    })
                    log.info("Cancelled orphan order: %s %s %s %s",
                             order["type"], order["side"], order["size"], order["market"])
                except Exception as e:
                    log.error("Failed to cancel orphan order %s: %s", order["order_id"], e)

        # --- Pass 2: Cancel duplicate TP/SL orders per active position ---
        # Re-fetch if we cancelled any (state changed)
        if cancelled > 0:
            try:
                open_orders = await self.dydx.get_open_orders()
            except Exception as e:
                log.error("Failed to re-fetch orders for duplicate check: %s", e)
                return cancelled

        tp_types = {"TAKE_PROFIT", "TAKE_PROFIT_LIMIT"}
        sl_types = {"STOP_LIMIT", "STOP_MARKET", "STOP"}

        orders_by_market: dict[str, list[dict]] = {}
        for o in open_orders:
            if o["market"] in active_markets:
                orders_by_market.setdefault(o["market"], []).append(o)

        for market, market_orders in orders_by_market.items():
            tp_orders = [o for o in market_orders if o["type"] in tp_types]
            sl_orders = [o for o in market_orders if o["type"] in sl_types]

            # Cancel excess TP orders (keep first, cancel rest)
            for excess in tp_orders[1:]:
                try:
                    client_id = int(excess["client_id"]) if str(excess["client_id"]).isdigit() else 0
                    await self.dydx.cancel_order(
                        client_id=client_id,
                        order_flags=excess.get("order_flags", ""),
                        good_til_block=excess.get("good_til_block"),
                        good_til_block_time=excess.get("good_til_block_time"),
                    )
                    cancelled += 1
                    self._append_jsonl("trades.jsonl", {
                        "timestamp": _ts(),
                        "action": "DUPLICATE_TP_CANCELLED",
                        "order_id": excess["order_id"],
                        "market": market,
                        "mode": "live",
                        "status": "CANCELLED",
                    })
                    log.warning("Cancelled duplicate TP order %s in %s",
                                excess["order_id"], market)
                except Exception as e:
                    log.error("Failed to cancel duplicate TP %s: %s", excess["order_id"], e)

            # Cancel excess SL orders (keep first, cancel rest)
            for excess in sl_orders[1:]:
                try:
                    client_id = int(excess["client_id"]) if str(excess["client_id"]).isdigit() else 0
                    await self.dydx.cancel_order(
                        client_id=client_id,
                        order_flags=excess.get("order_flags", ""),
                        good_til_block=excess.get("good_til_block"),
                        good_til_block_time=excess.get("good_til_block_time"),
                    )
                    cancelled += 1
                    self._append_jsonl("trades.jsonl", {
                        "timestamp": _ts(),
                        "action": "DUPLICATE_SL_CANCELLED",
                        "order_id": excess["order_id"],
                        "market": market,
                        "mode": "live",
                        "status": "CANCELLED",
                    })
                    log.warning("Cancelled duplicate SL order %s in %s",
                                excess["order_id"], market)
                except Exception as e:
                    log.error("Failed to cancel duplicate SL %s: %s", excess["order_id"], e)

        if cancelled == 0:
            log.info("Orphan cleanup: all %d open orders are valid", len(open_orders))

        return cancelled

    async def verify_position_protection(self) -> int:
        """Check that every open position has TP and SL orders.

        Returns the number of unprotected positions found and emergency-closed.
        """
        try:
            portfolio = await self.dydx.get_portfolio_state()
            open_orders = await self.dydx.get_open_orders()
        except Exception as e:
            log.error("Failed to fetch state for position protection check: %s", e)
            return 0

        positions = portfolio.get("positions", [])
        if not positions:
            log.info("Position protection: no open positions")
            return 0

        # Group orders by market
        orders_by_market: dict[str, list[dict]] = {}
        for o in open_orders:
            orders_by_market.setdefault(o["market"], []).append(o)

        unprotected = 0
        for pos in positions:
            market = pos["market"]
            market_orders = orders_by_market.get(market, [])
            has_tp = any(
                o["type"] in ("TAKE_PROFIT", "TAKE_PROFIT_LIMIT")
                for o in market_orders
            )
            has_sl = any(
                o["type"] in ("STOP_LIMIT", "STOP_MARKET", "STOP")
                for o in market_orders
            )

            if has_tp and has_sl:
                continue

            missing = []
            if not has_tp:
                missing.append("TP")
            if not has_sl:
                missing.append("SL")

            log.critical(
                "UNPROTECTED POSITION: %s %s %s — missing %s",
                pos["side"], pos["size"], market, "+".join(missing),
            )
            self._append_jsonl("trades.jsonl", {
                "timestamp": _ts(),
                "action": "UNPROTECTED_POSITION_DETECTED",
                "market": market,
                "side": pos["side"],
                "size": pos["size"],
                "missing_orders": missing,
                "mode": "live",
                "status": "CRITICAL",
            })

            # Emergency close if SL is missing (no downside protection)
            if not has_sl:
                from dydxv4.clients.constants import OrderSide
                close_side = (
                    OrderSide.SELL if pos["side"] == "LONG" else OrderSide.BUY
                )
                size = abs(float(pos.get("size", 0)))
                await self._emergency_close(
                    {"size_btc": size, "direction": pos["side"]}, close_side
                )
                unprotected += 1

        if unprotected == 0:
            log.info(
                "Position protection: all %d position(s) have TP+SL orders",
                len(positions),
            )
        return unprotected

    # ------------------------------------------------------------------
    # Order building
    # ------------------------------------------------------------------

    def _build_order_params(
        self, decision: dict, portfolio: dict, market_price: float
    ) -> dict:
        direction = decision["direction"]
        equity = portfolio["equity"]
        size_pct = decision.get("position_size_pct", 0.05)
        max_btc = self.cfg.get("max_position_size_btc", 0.05)

        size_btc = (equity * size_pct) / market_price if market_price > 0 else 0
        size_btc = min(size_btc, max_btc)
        size_btc = round(size_btc, 3)  # dYdX BTC-USD step size
        if size_btc < 0.001:
            size_btc = 0  # signal caller to reject — too small to trade

        from dydxv4.clients.constants import OrderSide
        side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL

        # dYdX v4 has no true market orders — all are limit orders.
        # For IOC "market" orders, set a limit price with slippage buffer
        # to sweep the book: BUY high (accept asks up to N% above), SELL low.
        # BTC-USD tick size on dYdX v4 is $1 — round to whole dollars.
        slippage_pct = self.cfg.get("market_order_slippage_pct", 0.05)  # 5%
        if side == OrderSide.BUY:
            limit_price = float(int(market_price * (1 + slippage_pct)) + 1)
        else:
            limit_price = float(int(market_price * (1 - slippage_pct)))
            if limit_price <= 0:
                limit_price = 1.0  # minimum valid price

        return {
            "direction": direction,
            "side": side,
            "size_btc": size_btc,
            "market_price": market_price,
            "limit_price": limit_price,
            "entry_price": decision.get("entry_price", market_price),
            "take_profit": decision.get("take_profit", 0),
            "stop_loss": decision.get("stop_loss", 0),
        }

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    async def _place_entry_order(
        self, decision: dict, params: dict, portfolio: dict
    ) -> dict:
        """Place a short-term market order for entry."""
        from dydxv4.clients.constants import OrderType, OrderSide, OrderTimeInForce

        record_base = {
            "timestamp": _ts(),
            "action": "ENTRY",
            "direction": params["direction"],
            "side": "BUY" if params["direction"] == "LONG" else "SELL",
            "size_btc": params["size_btc"],
            "entry_price": params["entry_price"],
            "take_profit": params["take_profit"],
            "stop_loss": params["stop_loss"],
            "duration_minutes": decision.get("duration_minutes", 60),
            "confidence": decision.get("confidence", 0),
            "notional_usd": round(params["size_btc"] * params["market_price"], 2),
            "fee_usd": 0,  # updated after fill
            "equity_at_entry": portfolio["equity"],
            "mode": "live",
        }

        try:
            block_height = await self.dydx.get_latest_block_height()
            good_til_block = block_height + self.cfg.get("short_term_block_offset", 10)

            market = self.dydx.market
            subaccount = self.dydx.subaccount

            client_id = random.randint(0, MAX_CLIENT_ID)

            tx = await self.dydx.client.place_short_term_order(
                subaccount,
                market=self.cfg.get("market", "BTC-USD"),
                side=params["side"],
                price=params["limit_price"],
                size=params["size_btc"],
                client_id=client_id,
                good_til_block=good_til_block,
                time_in_force=OrderTimeInForce.IOC,
                reduce_only=False,
            )
            log.info("Entry order TX submitted: %s", tx)

        except Exception as e:
            # Retry once on sequence mismatch
            if "sequence" in str(e).lower():
                log.warning("Wallet sequence mismatch, retrying: %s", e)
                try:
                    # Re-fetch block height — the original may have expired
                    block_height = await self.dydx.get_latest_block_height()
                    good_til_block = block_height + self.cfg.get("short_term_block_offset", 10)
                    # Generate new client_id and capture it so _wait_for_fill uses it
                    client_id = random.randint(0, MAX_CLIENT_ID)
                    tx = await self.dydx.client.place_short_term_order(
                        self.dydx.subaccount,
                        market=self.cfg.get("market", "BTC-USD"),
                        side=params["side"],
                        price=params.get("limit_price", 0),
                        size=params["size_btc"],
                        client_id=client_id,
                        good_til_block=good_til_block,
                        time_in_force=OrderTimeInForce.IOC,
                        reduce_only=False,
                    )
                except Exception as retry_err:
                    log.error("Entry order retry failed: %s", retry_err)
                    record_base["status"] = "FAILED"
                    record_base["error"] = str(retry_err)
                    return record_base
            else:
                log.error("Entry order failed: %s", e)
                record_base["status"] = "FAILED"
                record_base["error"] = str(e)
                return record_base

        # Wait for fill confirmation (polling loop)
        record_base["client_id"] = client_id
        fill = await self._wait_for_fill(client_id)
        if fill:
            record_base["fill_price"] = fill.get("price", params["market_price"])
            record_base["fee_usd"] = float(fill.get("fee", 0))
            record_base["status"] = "FILLED"
        else:
            record_base["fill_price"] = params["market_price"]
            record_base["status"] = "UNVERIFIED"
            log.warning("Fill not confirmed within timeout — logged as UNVERIFIED")

        return record_base

    async def _wait_for_fill(self, client_id: int) -> dict | None:
        """Poll Indexer for fill confirmation with retries.

        Only returns fills matched by clientId to avoid misattributing
        fills from other orders (e.g., after a crash/restart).
        """
        max_attempts = self.cfg.get("fill_poll_max_attempts", 10)
        poll_interval_s = self.cfg.get("fill_poll_interval_s", 2)

        for attempt in range(max_attempts):
            await asyncio.sleep(poll_interval_s)
            try:
                address = self.dydx.subaccount.address
                fills_resp = await self.dydx.client.indexer_client.account.get_subaccount_fills(
                    address, self.dydx.subaccount_number, limit=10
                )
                fills = fills_resp.get("fills", [])
                for fill in fills:
                    if str(fill.get("clientId", "")) == str(client_id):
                        log.info("Fill confirmed (matched clientId) on attempt %d", attempt + 1)
                        return fill
            except Exception as e:
                log.warning("Fill poll attempt %d failed: %s", attempt + 1, e)
        log.warning("No fill matched clientId=%d after %d attempts", client_id, max_attempts)
        return None

    async def _place_tp_sl_orders(
        self, decision: dict, params: dict, entry_record: dict
    ):
        """Place take-profit and stop-loss conditional orders.

        If the critical SL order fails, attempt to close the position
        via a market order to avoid an unprotected position.
        """
        from dydxv4.clients.constants import OrderSide, OrderTimeInForce, OrderType

        # Closing side is opposite of entry
        close_side = OrderSide.SELL if params["direction"] == "LONG" else OrderSide.BUY
        tp_ok = False
        sl_ok = False

        # Expiry: trade duration + 1 hour buffer, clamped to [1h, 24h]
        duration_min = decision.get("duration_minutes", 60)
        gtt_seconds = min(max((duration_min + 60) * 60, 3600), 86400)

        # Take profit — round to $1 tick size (dYdX BTC-USD requirement)
        tp_price = params["take_profit"]
        if tp_price > 0:
            if close_side == OrderSide.SELL:
                tp_tick_price = float(int(tp_price))       # floor for sell
            else:
                tp_tick_price = float(int(tp_price) + 1)   # ceil for buy
            try:
                await self.dydx.client.place_order(
                    self.dydx.subaccount,
                    market=self.cfg.get("market", "BTC-USD"),
                    type=OrderType.TAKE_PROFIT,
                    side=close_side,
                    price=tp_tick_price,
                    trigger_price=tp_tick_price,
                    size=params["size_btc"],
                    client_id=random.randint(0, MAX_CLIENT_ID),
                    time_in_force=OrderTimeInForce.GTT,
                    good_til_time_in_seconds=gtt_seconds,
                    reduce_only=True,
                )
                log.info("TP order placed at $%,.0f (raw $%,.2f)", tp_tick_price, tp_price)
                tp_ok = True
            except Exception as e:
                log.error("TP order failed: %s", e)
                self._append_jsonl("trades.jsonl", {
                    "timestamp": _ts(),
                    "action": "TP_ORDER_FAILED",
                    "error": str(e),
                    "mode": "live",
                    "status": "ALERT",
                })

        # Stop loss (critical — position is unprotected without it)
        # Round trigger to $1 tick size, and add slippage buffer to limit
        # price so the order fills even if the market gaps past the trigger.
        sl_price = params["stop_loss"]
        if sl_price > 0:
            slippage_pct = self.cfg.get("market_order_slippage_pct", 0.05)
            if close_side == OrderSide.SELL:
                sl_trigger = float(int(sl_price) + 1)   # ceil → trigger sooner
                sl_limit = float(int(sl_price * (1 - slippage_pct)))  # sell lower to fill on gap
            else:
                sl_trigger = float(int(sl_price))        # floor → trigger sooner
                sl_limit = float(int(sl_price * (1 + slippage_pct)) + 1)  # buy higher to fill on gap
            try:
                await self.dydx.client.place_order(
                    self.dydx.subaccount,
                    market=self.cfg.get("market", "BTC-USD"),
                    type=OrderType.STOP_LIMIT,
                    side=close_side,
                    price=sl_limit,
                    trigger_price=sl_trigger,
                    size=params["size_btc"],
                    client_id=random.randint(0, MAX_CLIENT_ID),
                    time_in_force=OrderTimeInForce.GTT,
                    good_til_time_in_seconds=gtt_seconds,
                    reduce_only=True,
                )
                log.info("SL order placed: trigger $%,.0f, limit $%,.0f (raw $%,.2f)",
                         sl_trigger, sl_limit, sl_price)
                sl_ok = True
            except Exception as e:
                log.error("SL order failed — position is UNPROTECTED: %s", e)
                self._append_jsonl("trades.jsonl", {
                    "timestamp": _ts(),
                    "action": "SL_ORDER_FAILED",
                    "error": str(e),
                    "mode": "live",
                    "status": "ALERT",
                })

        # Rollback: if SL failed, close the position immediately to avoid
        # holding an unprotected position
        if not sl_ok:
            log.warning("SL order failed — attempting emergency position close")
            await self._emergency_close(params, close_side)

    # ------------------------------------------------------------------
    # Emergency position management
    # ------------------------------------------------------------------

    async def _emergency_close(self, params: dict, close_side):
        """Attempt to close a position via market order when SL placement fails.

        Fetches the actual on-chain position size (not the pre-calculated
        params size) to ensure the close covers the real exposure.
        Polls for fill confirmation after submitting. Logs CRITICAL if
        the close cannot be verified.
        """
        from dydxv4.clients.constants import OrderTimeInForce

        try:
            # Use actual on-chain position size, not the pre-calculated size
            # which could differ due to partial fills or rounding
            close_size = params["size_btc"]
            try:
                portfolio = await self.dydx.get_portfolio_state()
                market = self.cfg.get("market", "BTC-USD")
                for pos in portfolio.get("positions", []):
                    if pos.get("market") == market:
                        actual_size = abs(float(pos.get("size", 0)))
                        if actual_size > 0:
                            close_size = round(actual_size, 3)
                            log.info("Emergency close using on-chain size %.3f BTC "
                                     "(params had %.3f)", close_size, params["size_btc"])
                        break
            except Exception as e:
                log.warning("Could not fetch on-chain position size, "
                            "falling back to params size: %s", e)

            # Compute aggressive limit price to sweep the book for IOC close
            from dydxv4.clients.constants import OrderSide
            try:
                close_market_price = await self.dydx.get_current_price()
            except Exception:
                close_market_price = params.get("market_price", 0)
            if close_market_price <= 0:
                log.critical("Cannot compute close price — market price unavailable. "
                             "Manual intervention required.")
                self._append_jsonl("trades.jsonl", {
                    "timestamp": _ts(),
                    "action": "EMERGENCY_CLOSE_FAILED",
                    "error": "market price unavailable for limit price computation",
                    "mode": "live",
                    "status": "CRITICAL",
                })
                return
            slippage_pct = self.cfg.get("market_order_slippage_pct", 0.05)
            if close_side == OrderSide.BUY:
                close_limit_price = float(int(close_market_price * (1 + slippage_pct)) + 1)
            else:
                close_limit_price = float(int(close_market_price * (1 - slippage_pct)))
                if close_limit_price <= 0:
                    close_limit_price = 1.0

            block_height = await self.dydx.get_latest_block_height()
            good_til_block = block_height + self.cfg.get("short_term_block_offset", 10)
            close_client_id = random.randint(0, MAX_CLIENT_ID)
            await self.dydx.client.place_short_term_order(
                self.dydx.subaccount,
                market=self.cfg.get("market", "BTC-USD"),
                side=close_side,
                price=close_limit_price,
                size=close_size,
                client_id=close_client_id,
                good_til_block=good_til_block,
                time_in_force=OrderTimeInForce.IOC,
                reduce_only=True,
            )
            log.info("Emergency close order submitted for %.3f BTC", close_size)

            # Verify the close order filled
            fill = await self._wait_for_fill(close_client_id)
            if fill:
                log.info("Emergency close CONFIRMED — fill price: %s", fill.get("price", "?"))
                self._append_jsonl("trades.jsonl", {
                    "timestamp": _ts(),
                    "action": "EMERGENCY_CLOSE",
                    "size_btc": close_size,
                    "fill_price": fill.get("price"),
                    "mode": "live",
                    "status": "FILLED",
                })
            else:
                log.critical(
                    "Emergency close UNVERIFIED — position may still be open! "
                    "Manual intervention required."
                )
                self._append_jsonl("trades.jsonl", {
                    "timestamp": _ts(),
                    "action": "EMERGENCY_CLOSE",
                    "size_btc": close_size,
                    "mode": "live",
                    "status": "UNVERIFIED",
                })
        except Exception as e:
            log.error("CRITICAL: Emergency close FAILED — manual intervention required: %s", e)
            self._append_jsonl("trades.jsonl", {
                "timestamp": _ts(),
                "action": "EMERGENCY_CLOSE_FAILED",
                "error": str(e),
                "mode": "live",
                "status": "CRITICAL",
            })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_decision_file(self, path: str | None = None) -> dict:
        fpath = path or DECISION_PATH
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Decision file not found: {fpath}")
        with open(fpath) as f:
            return json.load(f)

    def _rejection_record(self, decision: dict, reason: str) -> dict:
        return {
            "timestamp": _ts(),
            "action": "REJECTED",
            "direction": decision.get("direction", "NO_TRADE"),
            "confidence": decision.get("confidence", 0),
            "rejection_reason": reason,
            "mode": "live",
            "status": "REJECTED",
        }

    def _append_jsonl(self, filename: str, record: dict):
        path = os.path.join(self.state_dir, filename)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ======================================================================
# Standalone CLI
# ======================================================================

async def _run_live(decision_path: str | None):
    """Run live execution via DydxClient + DydxExecutor."""
    from execution.dydx_client import DydxClient

    client = DydxClient()
    await client.connect()
    try:
        executor = DydxExecutor(client)
        decision = None
        if decision_path:
            with open(decision_path) as f:
                decision = json.load(f)
        result = await executor.execute_decision(decision)
        _print_result(result)
    finally:
        await client.disconnect()


def _run_paper(decision_path: str | None):
    """Run paper execution (no SDK needed)."""
    from execution.paper_executor import PaperExecutor

    executor = PaperExecutor()
    if decision_path:
        with open(decision_path) as f:
            decision = json.load(f)
    else:
        fpath = DECISION_PATH
        if not os.path.exists(fpath):
            print(f"ERROR: No decision file at {fpath}")
            sys.exit(1)
        with open(fpath) as f:
            decision = json.load(f)
    result = executor.execute_decision(decision)
    _print_result(result)


def _print_result(result: dict):
    """Print human-readable execution result."""
    status = result.get("status", "?")
    action = result.get("action", "?")
    print(f"\n{'=' * 50}")
    print(f"EXECUTION RESULT: {action} — {status}")
    print(f"{'=' * 50}")

    if action == "REJECTED":
        print(f"  Reason: {result.get('rejection_reason', 'unknown')}")
    elif action == "ENTRY":
        print(f"  Direction: {result.get('direction')}")
        print(f"  Size:      {result.get('size_btc')} BTC")
        print(f"  Fill:      ${result.get('fill_price', 0):,.2f}")
        print(f"  Notional:  ${result.get('notional_usd', 0):,.2f}")
        print(f"  TP:        ${result.get('take_profit', 0):,.2f}")
        print(f"  SL:        ${result.get('stop_loss', 0):,.2f}")
        print(f"  Mode:      {result.get('mode')}")
    print(f"{'=' * 50}")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="dYdX v4 Trade Executor")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--paper", action="store_true", help="Force paper mode")
    group.add_argument("--live", action="store_true", help="Force live mode")
    parser.add_argument("--decision", type=str, default=None,
                        help="Path to decision.json (default: llm_agent/decision.json)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = _load_config().get("execution", {})

    if args.paper:
        mode = "paper"
    elif args.live:
        mode = "live"
    else:
        mode = cfg.get("mode", "paper")

    log.info("Execution mode: %s", mode)

    if mode == "paper":
        _run_paper(args.decision)
    else:
        asyncio.run(_run_live(args.decision))


if __name__ == "__main__":
    main()
