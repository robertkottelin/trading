"""Trade History Reader — summarises executed trades and portfolio equity curve.

Reads state_data/trades.jsonl and state_data/portfolio.jsonl to build a
concise text section for the LLM prompt so the agent can learn from real
execution outcomes (fills, rejections, PnL, fees, equity trend).
"""

import json
import logging
import os
from datetime import datetime, timezone, timedelta

import yaml

log = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")


def _load_exec_config() -> dict:
    try:
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f).get("execution", {})
    except Exception:
        return {}


def _read_jsonl(path: str, max_lines: int = 500) -> list[dict]:
    """Read a JSONL file, returning the last *max_lines* records."""
    if not os.path.exists(path):
        return []
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records[-max_lines:]
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Could not read %s: %s", path, e)
        return []


def get_trade_history(n_trades: int = 20, n_snapshots: int = 10) -> str:
    """Build a formatted trade-history section for the LLM prompt.

    Includes:
      1. Recent executed trades (entries, rejections) with PnL & fees
      2. Aggregate stats: win rate, total PnL, avg trade, max win/loss
      3. Portfolio equity curve (last *n_snapshots* snapshots)

    Args:
        n_trades: Max number of recent trades to show.
        n_snapshots: Max number of portfolio snapshots to show.

    Returns:
        Multi-line formatted text (empty-safe — returns a short notice if
        no history exists yet).
    """
    cfg = _load_exec_config()
    state_dir = cfg.get("state_data_dir", "state_data")

    trades = _read_jsonl(os.path.join(state_dir, "trades.jsonl"))
    snapshots = _read_jsonl(os.path.join(state_dir, "portfolio.jsonl"))

    if not trades and not snapshots:
        return "TRADE HISTORY: No executed trades yet."

    lines = ["TRADE HISTORY (executed orders):"]

    # --- Trade log ---
    if trades:
        recent = trades[-n_trades:]
        entries = [t for t in recent if t.get("action") == "ENTRY"]
        rejections = [t for t in recent if t.get("action") == "REJECTED"]

        # Aggregate stats over ALL trades (not just the recent window)
        all_entries = [t for t in trades if t.get("action") == "ENTRY"]
        _append_aggregate_stats(lines, all_entries)

        # Recent entries
        if entries:
            lines.append(f"  Recent fills ({len(entries)}):")
            for t in entries[-10:]:
                ts = t.get("timestamp", "?")[:16]
                direction = t.get("direction", "?")
                size = float(t.get("size_btc", 0))
                fill = float(t.get("fill_price", 0) or 0)
                tp = float(t.get("take_profit", 0) or 0)
                sl = float(t.get("stop_loss", 0) or 0)
                fee = float(t.get("fee_usd", 0) or 0)
                notional = float(t.get("notional_usd", 0) or 0)
                status = t.get("status", "?")
                mode = t.get("mode", "?")
                lines.append(
                    f"    {ts} {direction} {size} BTC @ ${fill:,.2f} "
                    f"(TP ${tp:,.2f} / SL ${sl:,.2f}) "
                    f"notional=${notional:,.2f} fee=${fee:.3f} "
                    f"[{status}] [{mode}]"
                )

        # Recent rejections
        if rejections:
            lines.append(f"  Recent rejections ({len(rejections)}):")
            for t in rejections[-5:]:
                ts = t.get("timestamp", "?")[:16]
                direction = t.get("direction", "?")
                reason = t.get("rejection_reason", "?")
                conf = t.get("confidence", 0)
                lines.append(
                    f"    {ts} {direction} conf={conf:.2f} — {reason}"
                )
    else:
        lines.append("  No executed trades yet.")

    # --- Equity curve ---
    if snapshots:
        recent_snaps = snapshots[-n_snapshots:]
        lines.append(f"  Portfolio equity curve (last {len(recent_snaps)} snapshots):")
        for s in recent_snaps:
            ts = s.get("timestamp", "?")[:16]
            eq = s.get("equity", 0)
            free = s.get("free_collateral", 0)
            margin = s.get("margin_pct", 0)
            n_pos = len(s.get("positions", []))
            lines.append(
                f"    {ts} equity=${eq:,.2f} free=${free:,.2f} "
                f"margin={margin:.1f}% positions={n_pos}"
            )

        # Equity change
        if len(snapshots) >= 2:
            first_eq = snapshots[0].get("equity", 0)
            last_eq = snapshots[-1].get("equity", 0)
            if first_eq > 0:
                eq_change = (last_eq - first_eq) / first_eq * 100
                lines.append(
                    f"  Equity change since first snapshot: "
                    f"${first_eq:,.2f} -> ${last_eq:,.2f} ({eq_change:+.2f}%)"
                )

    return "\n".join(lines)


def _append_aggregate_stats(lines: list[str], entries: list[dict]):
    """Compute and append aggregate trade statistics."""
    if not entries:
        return

    filled = [t for t in entries if t.get("status") == "FILLED"]
    total_count = len(entries)
    filled_count = len(filled)
    failed_count = sum(1 for t in entries if t.get("status") == "FAILED")
    unverified_count = sum(1 for t in entries if t.get("status") == "UNVERIFIED")

    total_fees = sum(t.get("fee_usd", 0) for t in filled)
    total_notional = sum(t.get("notional_usd", 0) for t in filled)

    # Direction breakdown
    longs = sum(1 for t in filled if t.get("direction") == "LONG")
    shorts = sum(1 for t in filled if t.get("direction") == "SHORT")

    # Paper vs live
    paper = sum(1 for t in filled if t.get("mode") == "paper")
    live = sum(1 for t in filled if t.get("mode") == "live")

    lines.append(
        f"  Total entries: {total_count} "
        f"(filled={filled_count}, failed={failed_count}, unverified={unverified_count})"
    )
    lines.append(
        f"  Direction split: {longs} long, {shorts} short"
    )
    if paper or live:
        lines.append(f"  Mode split: {paper} paper, {live} live")
    lines.append(
        f"  Total notional: ${total_notional:,.2f}, total fees: ${total_fees:,.2f}"
    )
