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
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv

from execution.risk_manager import RiskManager

log = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")
DECISION_PATH = os.path.join(os.path.dirname(__file__), "..", "llm_agent", "decision.json")
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
        """Execute a trading decision.  Returns the trade record."""

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
        if order_params["size_btc"] < 0.0001:
            record = self._rejection_record(
                decision, "computed position size below 0.0001 BTC minimum"
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
                has_position = len(check_portfolio.get("positions", [])) > 0
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
        """Cancel open orders that don't correspond to active positions.

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

        orphans = [o for o in open_orders if o["market"] not in active_markets]
        if not orphans:
            log.info("Orphan cleanup: all %d open orders have matching positions",
                     len(open_orders))
            return 0

        log.warning("Orphan cleanup: found %d orphan order(s) to cancel", len(orphans))
        cancelled = 0
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

        return cancelled

    async def verify_position_orders(self) -> dict:
        """Check that every open position has active TP/SL orders.

        If a position is missing TP and/or SL, attempt to re-place them
        using default risk parameters (SL at 1% adverse, TP at 1.5% favorable).

        Returns a dict with:
            positions_checked: int
            unprotected: list of markets missing TP and/or SL
            reprotected: list of markets where TP/SL were re-placed
        """
        result = {"positions_checked": 0, "unprotected": [], "reprotected": []}

        try:
            portfolio = await self.dydx.get_portfolio_state()
            open_orders = await self.dydx.get_open_orders()
        except Exception as e:
            log.error("Failed to fetch state for position monitoring: %s", e)
            return result

        positions = portfolio.get("positions", [])
        result["positions_checked"] = len(positions)

        if not positions:
            log.info("Position monitor: no open positions")
            return result

        # Build a map: market -> set of order types (TAKE_PROFIT, STOP_LIMIT, etc.)
        orders_by_market: dict[str, set[str]] = {}
        for o in open_orders:
            mkt = o["market"]
            if mkt not in orders_by_market:
                orders_by_market[mkt] = set()
            orders_by_market[mkt].add(o["type"])

        for pos in positions:
            market = pos["market"]
            order_types = orders_by_market.get(market, set())

            has_tp = any("PROFIT" in t.upper() for t in order_types)
            has_sl = any("STOP" in t.upper() for t in order_types)

            if has_tp and has_sl:
                log.info("Position monitor: %s has TP + SL — OK", market)
                continue

            missing = []
            if not has_tp:
                missing.append("TP")
            if not has_sl:
                missing.append("SL")

            log.warning(
                "Position monitor: %s MISSING %s — position is %s %s @ %s",
                market, "+".join(missing),
                pos["side"], pos["size"], pos["entry_price"],
            )

            self._append_jsonl("trades.jsonl", {
                "timestamp": _ts(),
                "action": "POSITION_UNPROTECTED",
                "market": market,
                "side": pos["side"],
                "size": pos["size"],
                "entry_price": pos["entry_price"],
                "missing_orders": missing,
                "mode": "live",
                "status": "ALERT",
            })

            # Attempt to re-place missing protective orders
            reprotected = await self._reprotect_position(pos, has_tp, has_sl)
            if reprotected:
                result["reprotected"].append({
                    "market": market,
                    "side": pos["side"],
                    "orders_placed": reprotected,
                })
            else:
                result["unprotected"].append({
                    "market": market,
                    "side": pos["side"],
                    "size": pos["size"],
                    "entry_price": pos["entry_price"],
                    "missing": missing,
                })

        return result

    async def _reprotect_position(
        self, pos: dict, has_tp: bool, has_sl: bool
    ) -> list[str]:
        """Place missing TP/SL orders for an unprotected position.

        Uses default risk parameters:
          - SL: 1% adverse from entry price
          - TP: 1.5% favorable from entry price
          - GTT: 24 hours

        Returns list of order types successfully placed, e.g. ["TP", "SL"].
        """
        from dydx4.clients.helpers.chain_helpers import (
            OrderType, OrderSide, OrderTimeInForce, OrderExecution,
        )

        entry_price = float(pos["entry_price"])
        size = float(pos["size"])
        side = pos["side"]  # "LONG" or "SHORT"
        market = pos["market"]

        sl_pct = self.cfg.get("reprotect_sl_pct", 0.01)
        tp_pct = self.cfg.get("reprotect_tp_pct", 0.015)
        gtt_seconds = 86400  # 24h max

        if side == "LONG":
            sl_price = round(entry_price * (1 - sl_pct), 0)
            tp_price = round(entry_price * (1 + tp_pct), 0)
            close_side = OrderSide.SELL
        else:
            sl_price = round(entry_price * (1 + sl_pct), 0)
            tp_price = round(entry_price * (1 - tp_pct), 0)
            close_side = OrderSide.BUY

        placed = []

        # Place TP if missing
        if not has_tp:
            try:
                self.dydx.client.place_order(
                    self.dydx.subaccount,
                    market=market,
                    type=OrderType.TAKE_PROFIT_MARKET,
                    side=close_side,
                    price=tp_price,
                    size=size,
                    client_id=random.randint(0, MAX_CLIENT_ID),
                    time_in_force=OrderTimeInForce.GTT,
                    good_til_block=0,
                    good_til_time_in_seconds=gtt_seconds,
                    execution=OrderExecution.IOC,
                    post_only=False,
                    reduce_only=True,
                    trigger_price=tp_price,
                )
                log.info("Re-placed TP at $%.0f for %s %s", tp_price, side, market)
                placed.append("TP")
                self._append_jsonl("trades.jsonl", {
                    "timestamp": _ts(),
                    "action": "REPROTECT_TP",
                    "market": market,
                    "side": side,
                    "size": size,
                    "tp_price": tp_price,
                    "entry_price": entry_price,
                    "mode": "live",
                    "status": "PLACED",
                })
            except Exception as e:
                log.error("Failed to re-place TP for %s: %s", market, e)

        # Place SL if missing
        if not has_sl:
            try:
                # Add 1% slippage tolerance so SL fills in fast drops
                if close_side == OrderSide.SELL:
                    sl_limit = round(sl_price * 0.99)
                else:
                    sl_limit = round(sl_price * 1.01)
                self.dydx.client.place_order(
                    self.dydx.subaccount,
                    market=market,
                    type=OrderType.STOP_MARKET,
                    side=close_side,
                    price=sl_limit,
                    size=size,
                    client_id=random.randint(0, MAX_CLIENT_ID),
                    time_in_force=OrderTimeInForce.GTT,
                    good_til_block=0,
                    good_til_time_in_seconds=gtt_seconds,
                    execution=OrderExecution.IOC,
                    post_only=False,
                    reduce_only=True,
                    trigger_price=sl_price,
                )
                log.info("Re-placed SL at $%.0f (limit $%.0f) for %s %s", sl_price, sl_limit, side, market)
                placed.append("SL")
                self._append_jsonl("trades.jsonl", {
                    "timestamp": _ts(),
                    "action": "REPROTECT_SL",
                    "market": market,
                    "side": side,
                    "size": size,
                    "sl_price": sl_price,
                    "entry_price": entry_price,
                    "mode": "live",
                    "status": "PLACED",
                })
            except Exception as e:
                log.error("Failed to re-place SL for %s: %s", market, e)
                # SL is critical — emergency-close regardless of whether TP
                # was placed.  A position with TP but no SL has unlimited
                # downside on a leveraged perpetual.
                if "SL" not in placed:
                    log.warning("SL re-placement failed — attempting emergency close")
                    from dydx4.clients.helpers.chain_helpers import OrderSide as OS
                    await self._emergency_close(
                        {"size_btc": size, "market_price": entry_price},
                        close_side,
                    )

        if placed:
            log.info("Re-protected %s position with %s", market, "+".join(placed))

        return placed

    # ------------------------------------------------------------------
    # Order building
    # ------------------------------------------------------------------

    def _build_order_params(
        self, decision: dict, portfolio: dict, market_price: float
    ) -> dict:
        direction = decision["direction"]
        equity = portfolio["equity"]
        max_btc = self.cfg.get("max_position_size_btc", 0.05)

        # USD-denominated sizing (preferred); legacy pct fallback for old decision files
        size_usd = decision.get("position_size_usd")
        if size_usd is None:
            size_pct = decision.get("position_size_pct", 0.05)
            size_usd = equity * size_pct

        size_btc = size_usd / market_price if market_price > 0 else 0
        size_btc = min(size_btc, max_btc)
        size_btc = round(size_btc, 4)  # dYdX BTC-USD step size (0.0001)
        if size_btc < 0.0001 and equity > 0 and market_price > 0:
            size_btc = 0.0001  # floor to dYdX minimum for small accounts

        from dydx4.clients.helpers.chain_helpers import OrderSide
        side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL

        return {
            "direction": direction,
            "side": side,
            "size_btc": size_btc,
            "market_price": market_price,
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
        from dydx4.clients.helpers.chain_helpers import OrderType, OrderSide, Order_TimeInForce

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

        # Compute order side and slippage-adjusted limit price before the try block
        # so both the initial attempt AND the sequence-mismatch retry can use them
        # without risking a NameError if an exception fires before these lines.
        order_side = OrderSide.BUY if params["direction"] == "LONG" else OrderSide.SELL
        # Add 0.5% slippage tolerance for IOC orders
        if order_side == OrderSide.BUY:
            limit_price = round(params["market_price"] * 1.005)  # willing to pay up to 0.5% above
        else:
            limit_price = round(params["market_price"] * 0.995)  # willing to sell down to 0.5% below

        try:
            block_height = await self.dydx.get_latest_block_height()
            good_til_block = block_height + self.cfg.get("short_term_block_offset", 10)

            market = self.dydx.market
            subaccount = self.dydx.subaccount

            client_id = random.randint(0, MAX_CLIENT_ID)

            tx = self.dydx.client.place_short_term_order(
                subaccount,
                market=self.cfg.get("market", "BTC-USD"),
                side=order_side,
                type=OrderType.MARKET,
                price=limit_price,
                size=params["size_btc"],
                client_id=client_id,
                good_til_block=good_til_block,
                time_in_force=Order_TimeInForce.TIME_IN_FORCE_IOC,
                reduce_only=False,
            )
            log.info("Entry order TX submitted (price=%s, slippage limit=%s): %s",
                     params["market_price"], limit_price, tx)

        except Exception as e:
            # Retry once on sequence mismatch
            if "sequence" in str(e).lower():
                log.warning("Wallet sequence mismatch, retrying: %s", e)
                try:
                    # Use a new client_id and capture it so fill polling
                    # tracks the actual order that was submitted, not the
                    # rejected one from the first attempt.
                    retry_client_id = random.randint(0, MAX_CLIENT_ID)
                    tx = self.dydx.client.place_short_term_order(
                        subaccount,
                        market=self.cfg.get("market", "BTC-USD"),
                        side=order_side,
                        type=OrderType.MARKET,
                        price=limit_price,
                        size=params["size_btc"],
                        client_id=retry_client_id,
                        good_til_block=good_til_block,
                        time_in_force=Order_TimeInForce.TIME_IN_FORCE_IOC,
                        reduce_only=False,
                    )
                    client_id = retry_client_id  # update for fill poll below
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
            record_base["fill_price"] = float(fill.get("price", params["market_price"]))
            record_base["fee_usd"] = float(fill.get("fee", 0))
            record_base["status"] = "FILLED"
        else:
            record_base["fill_price"] = params["market_price"]
            record_base["status"] = "UNVERIFIED"
            log.warning("Fill not confirmed within timeout — logged as UNVERIFIED")

        return record_base

    async def _wait_for_fill(self, client_id: int) -> dict | None:
        """Poll Indexer for fill confirmation with retries."""
        max_attempts = self.cfg.get("fill_poll_max_attempts", 10)
        poll_interval_s = self.cfg.get("fill_poll_interval_s", 2)
        # Record when the order was submitted so the fallback path cannot
        # accidentally match a fill from a previous pipeline run.
        poll_start = datetime.now(timezone.utc)

        for attempt in range(max_attempts):
            await asyncio.sleep(poll_interval_s)
            try:
                address = self.dydx.subaccount.address
                fills_raw = self.dydx.client.indexer_client.account.get_subaccount_fills(
                    address, self.dydx.subaccount_number, limit=10
                )
                fills_resp = self.dydx._unwrap(fills_raw)
                fills = fills_resp.get("fills", [])
                # Primary: match by clientId (exact, safe across runs)
                for fill in fills:
                    if fill.get("clientId") == str(client_id):
                        log.info("Fill confirmed (matched clientId) on attempt %d", attempt + 1)
                        return fill
                # Fallback: accept most recent fill only if it occurred AFTER this
                # order was submitted (with 10s grace for indexer latency / clock
                # drift).  The old 120s window was wide enough to match fills from
                # the previous pipeline cycle, recording the wrong fill price.
                if fills:
                    fill_ts = fills[0].get("createdAt", "")
                    if fill_ts:
                        try:
                            fill_time = datetime.fromisoformat(fill_ts.replace("Z", "+00:00"))
                            cutoff = poll_start - timedelta(seconds=10)
                            if fill_time >= cutoff:
                                age_s = (datetime.now(timezone.utc) - fill_time).total_seconds()
                                log.info("Fill found (%.0fs old, no clientId match) on attempt %d",
                                         age_s, attempt + 1)
                                return fills[0]
                            else:
                                log.debug("Most recent fill predates poll start by %.0fs — skipping stale fill",
                                          (poll_start - fill_time).total_seconds())
                        except (ValueError, TypeError):
                            pass
            except Exception as e:
                log.warning("Fill poll attempt %d failed: %s", attempt + 1, e)
        return None

    async def _place_tp_sl_orders(
        self, decision: dict, params: dict, entry_record: dict
    ):
        """Place take-profit and stop-loss conditional orders.

        If the critical SL order fails, attempt to close the position
        via a market order to avoid an unprotected position.
        """
        from dydx4.clients.helpers.chain_helpers import OrderType, OrderSide, OrderTimeInForce, OrderExecution

        # Closing side is opposite of entry
        close_side = OrderSide.SELL if params["direction"] == "LONG" else OrderSide.BUY
        tp_ok = False
        sl_ok = False

        # Use actual position size (may differ from ordered size on testnet)
        order_size = params["size_btc"]
        try:
            portfolio = await self.dydx.get_portfolio_state()
            market = self.cfg.get("market", "BTC-USD")
            for pos in portfolio.get("positions", []):
                if pos["market"] == market:
                    actual = float(pos["size"])
                    if actual > 0 and actual != order_size:
                        log.info("TP/SL using actual position size %.4f (ordered %.3f)",
                                 actual, order_size)
                        order_size = actual
                    break
        except Exception as e:
            log.warning("Could not fetch actual position size: %s — using ordered size", e)

        # Expiry: trade duration + 1 hour buffer, clamped to [1h, 24h]
        duration_min = decision.get("duration_minutes", 60)
        gtt_seconds = min(max((duration_min + 60) * 60, 3600), 86400)

        # Take profit
        tp_price = params["take_profit"]
        if tp_price > 0:
            try:
                from dydx4.clients.helpers.chain_helpers import OrderExecution
                self.dydx.client.place_order(
                    self.dydx.subaccount,
                    market=self.cfg.get("market", "BTC-USD"),
                    type=OrderType.TAKE_PROFIT_MARKET,
                    side=close_side,
                    price=tp_price,
                    size=order_size,
                    client_id=random.randint(0, MAX_CLIENT_ID),
                    time_in_force=OrderTimeInForce.GTT,
                    good_til_block=0,
                    good_til_time_in_seconds=gtt_seconds,
                    execution=OrderExecution.IOC,
                    post_only=False,
                    reduce_only=True,
                    trigger_price=tp_price,
                )
                log.info("TP order placed at $%.2f", tp_price)
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
        sl_price = params["stop_loss"]
        if sl_price > 0:
            try:
                from dydx4.clients.helpers.chain_helpers import OrderExecution
                # Add 1% slippage tolerance so SL fills even in fast drops
                if close_side == OrderSide.SELL:
                    sl_limit = round(sl_price * 0.99)  # accept up to 1% worse
                else:
                    sl_limit = round(sl_price * 1.01)
                self.dydx.client.place_order(
                    self.dydx.subaccount,
                    market=self.cfg.get("market", "BTC-USD"),
                    type=OrderType.STOP_MARKET,
                    side=close_side,
                    price=sl_limit,
                    size=order_size,
                    client_id=random.randint(0, MAX_CLIENT_ID),
                    time_in_force=OrderTimeInForce.GTT,
                    good_til_block=0,
                    good_til_time_in_seconds=gtt_seconds,
                    execution=OrderExecution.IOC,
                    post_only=False,
                    reduce_only=True,
                    trigger_price=sl_price,
                )
                log.info("SL order placed: trigger=$%.2f limit=$%.2f", sl_price, sl_limit)
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

        Fetches the actual position size from the chain to avoid
        reduce-only size mismatch errors (e.g. on testnet partial fills).
        """
        from dydx4.clients.helpers.chain_helpers import OrderType, Order_TimeInForce

        try:
            # Get actual position size to avoid reduce-only mismatch
            portfolio = await self.dydx.get_portfolio_state()
            positions = portfolio.get("positions", [])
            market = self.cfg.get("market", "BTC-USD")
            actual_size = params["size_btc"]
            for pos in positions:
                if pos["market"] == market:
                    actual_size = float(pos["size"])
                    break

            if actual_size <= 0:
                log.info("Emergency close: no open position found — skipping")
                return

            price = await self.dydx.get_current_price()
            # Add 0.5% slippage buffer so the IOC order fills even if the market
            # has moved since the price quote.  Without this the order is rejected
            # by the matching engine if the fill price crosses the limit, leaving
            # the position unprotected at 5x leverage.
            from dydx4.clients.helpers.chain_helpers import OrderSide as _OS
            if close_side == _OS.SELL:
                limit_price = round(price * 0.995)  # accept up to 0.5% below mid
            else:
                limit_price = round(price * 1.005)  # accept up to 0.5% above mid
            block_height = await self.dydx.get_latest_block_height()
            good_til_block = block_height + self.cfg.get("short_term_block_offset", 10)
            self.dydx.client.place_short_term_order(
                self.dydx.subaccount,
                market=market,
                side=close_side,
                type=OrderType.MARKET,
                price=limit_price,
                size=actual_size,
                client_id=random.randint(0, MAX_CLIENT_ID),
                good_til_block=good_til_block,
                time_in_force=Order_TimeInForce.TIME_IN_FORCE_IOC,
                reduce_only=True,
            )
            log.info("Emergency close order submitted for %.4f BTC @ limit $%.0f",
                     actual_size, limit_price)
            self._append_jsonl("trades.jsonl", {
                "timestamp": _ts(),
                "action": "EMERGENCY_CLOSE",
                "size_btc": actual_size,
                "mode": "live",
                "status": "SUBMITTED",
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
