"""Pre-trade risk validation and circuit breaker.

Pure validation — no side effects, no external deps beyond stdlib + yaml.
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")


def _load_exec_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f).get("execution", {})


class RiskManager:
    """Validates trading decisions against risk limits before execution."""

    def __init__(self, config: dict | None = None):
        self.cfg = config or _load_exec_config()

    def validate_decision(
        self, decision: dict, portfolio: dict
    ) -> tuple[bool, str]:
        """Run all pre-trade checks.  Returns (passed, reason)."""
        checks = [
            self._check_direction,
            self._check_confidence,
            self._check_price_ordering,
            self._check_risk_reward,
            self._check_existing_positions,
            self._check_free_collateral,
            self._check_position_size,
            self._check_daily_loss,
        ]
        for check in checks:
            passed, reason = check(decision, portfolio)
            if not passed:
                log.info("Risk check failed: %s", reason)
                return False, reason
        return True, "all checks passed"

    # ------------------------------------------------------------------
    # Individual checks — return (True, "") on pass, (False, reason) on fail
    # ------------------------------------------------------------------

    def _check_direction(self, decision: dict, portfolio: dict) -> tuple[bool, str]:
        direction = decision.get("direction", "NO_TRADE")
        if direction == "NO_TRADE":
            return False, "direction is NO_TRADE"
        if direction not in ("LONG", "SHORT"):
            return False, f"invalid direction: {direction}"
        return True, ""

    def _check_confidence(self, decision: dict, portfolio: dict) -> tuple[bool, str]:
        threshold = self.cfg.get("confidence_threshold", 0.6)
        confidence = decision.get("confidence", 0)
        if confidence < threshold:
            return False, f"confidence {confidence:.2f} below {threshold:.2f}"
        return True, ""

    def _check_price_ordering(self, decision: dict, portfolio: dict) -> tuple[bool, str]:
        direction = decision.get("direction", "")
        entry = decision.get("entry_price", 0)
        tp = decision.get("take_profit", 0)
        sl = decision.get("stop_loss", 0)
        if not entry or not tp or not sl:
            return True, ""  # _check_risk_reward will catch missing prices

        if direction == "LONG":
            if not (tp > entry > sl):
                return False, (
                    f"LONG price ordering invalid: need TP({tp}) > entry({entry}) > SL({sl})"
                )
        elif direction == "SHORT":
            if not (sl > entry > tp):
                return False, (
                    f"SHORT price ordering invalid: need SL({sl}) > entry({entry}) > TP({tp})"
                )
        return True, ""

    def _check_risk_reward(self, decision: dict, portfolio: dict) -> tuple[bool, str]:
        entry = decision.get("entry_price", 0)
        tp = decision.get("take_profit", 0)
        sl = decision.get("stop_loss", 0)
        if not entry or not tp or not sl:
            return False, "missing entry_price, take_profit, or stop_loss"

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0:
            return False, "risk is zero (entry == stop_loss)"
        rr = reward / risk
        if rr < 1.5:
            return False, f"risk:reward {rr:.2f}:1 below 1.5:1 minimum"
        return True, ""

    def _check_existing_positions(
        self, decision: dict, portfolio: dict
    ) -> tuple[bool, str]:
        max_open = self.cfg.get("max_open_positions", 1)
        positions = portfolio.get("positions", [])
        open_count = len(positions)
        if open_count >= max_open:
            return False, f"{open_count} open position(s), max is {max_open}"
        return True, ""

    def _check_free_collateral(
        self, decision: dict, portfolio: dict
    ) -> tuple[bool, str]:
        min_pct = self.cfg.get("min_free_collateral_pct", 20.0)
        equity = portfolio.get("equity", 0)
        free = portfolio.get("free_collateral", 0)
        if equity <= 0:
            return False, "equity is zero or negative"
        free_pct = (free / equity) * 100
        if free_pct < min_pct:
            return (
                False,
                f"free collateral {free_pct:.1f}% below {min_pct:.1f}% minimum",
            )
        return True, ""

    def _check_position_size(
        self, decision: dict, portfolio: dict
    ) -> tuple[bool, str]:
        max_btc = self.cfg.get("max_position_size_btc", 0.05)
        max_pct = self.cfg.get("max_position_pct", 0.25)
        size_pct = decision.get("position_size_pct", 0)

        if size_pct > max_pct:
            return (
                False,
                f"position_size_pct {size_pct:.2%} exceeds max {max_pct:.2%}",
            )

        # Check absolute BTC size if we can compute it
        equity = portfolio.get("equity", 0)
        entry = decision.get("entry_price", 0)
        if equity > 0 and entry > 0:
            btc_size = (equity * size_pct) / entry
            if btc_size > max_btc:
                return (
                    False,
                    f"BTC size {btc_size:.4f} exceeds max {max_btc} BTC",
                )
        return True, ""

    def _check_daily_loss(
        self, decision: dict, portfolio: dict
    ) -> tuple[bool, str]:
        max_loss_pct = self.cfg.get("max_daily_loss_pct", 2.0)
        equity = portfolio.get("equity", 0)
        if equity <= 0:
            return True, ""  # can't compute, let other checks catch it

        state_dir = self.cfg.get("state_data_dir", "state_data")

        # Strategy: compute daily PnL from portfolio equity snapshots.
        # The executor writes snapshots before and after each trade, so
        # we compare today's earliest equity to current equity.
        portfolio_path = os.path.join(state_dir, "portfolio.jsonl")
        if not os.path.exists(portfolio_path):
            return True, ""

        today = datetime.now(timezone.utc).date()
        today_start_equity = None
        try:
            with open(portfolio_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        snap = json.loads(line)
                    except json.JSONDecodeError:
                        log.debug("Skipping corrupt JSONL line in portfolio snapshots")
                        continue
                    ts = snap.get("timestamp", "")
                    if not ts:
                        continue
                    try:
                        snap_date = datetime.fromisoformat(ts).date()
                    except ValueError:
                        continue
                    if snap_date == today:
                        snap_equity = snap.get("equity", 0)
                        if snap_equity > 0 and today_start_equity is None:
                            today_start_equity = snap_equity
        except OSError as e:
            log.warning("Could not read portfolio snapshots for daily loss check: %s", e)
            return False, f"daily loss check failed: could not read portfolio history ({e})"

        if today_start_equity is None or today_start_equity <= 0:
            return True, ""

        daily_pnl_pct = (equity - today_start_equity) / today_start_equity * 100
        if daily_pnl_pct < 0:
            loss_pct = abs(daily_pnl_pct)
            if loss_pct >= max_loss_pct:
                return (
                    False,
                    f"daily loss {loss_pct:.2f}% hit circuit breaker ({max_loss_pct}%)",
                )
        return True, ""
