"""Performance analysis — reads state_data/ JSONL and decision_history.json.

Usage:
    python scripts/analyze_performance.py
    python scripts/analyze_performance.py --state-dir state_data
    python scripts/analyze_performance.py --json          # machine-readable output
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, returning a list of dicts."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_json(path: str) -> list | dict:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def analyze_trades(trades: list[dict]) -> dict:
    """Analyze trade records from trades.jsonl."""
    entries = [t for t in trades if t.get("action") == "ENTRY"]
    rejections = [t for t in trades if t.get("action") == "REJECTED"]

    stats = {
        "total_trades": len(entries),
        "total_rejections": len(rejections),
        "rejection_rate": len(rejections) / max(len(trades), 1),
    }

    if not entries:
        return stats

    # Per-direction breakdown
    by_dir = defaultdict(list)
    for t in entries:
        by_dir[t.get("direction", "UNKNOWN")].append(t)

    stats["per_direction"] = {}
    for direction, dir_trades in sorted(by_dir.items()):
        notionals = [t.get("notional_usd", 0) for t in dir_trades]
        fees = [t.get("fee_usd", 0) for t in dir_trades]
        sizes = [t.get("size_btc", 0) for t in dir_trades]
        confidences = [t.get("confidence", 0) for t in dir_trades]
        stats["per_direction"][direction] = {
            "count": len(dir_trades),
            "avg_size_btc": sum(sizes) / len(sizes),
            "total_notional_usd": sum(notionals),
            "total_fees_usd": sum(fees),
            "avg_confidence": sum(confidences) / len(confidences),
        }

    # Overall stats
    all_notional = sum(t.get("notional_usd", 0) for t in entries)
    all_fees = sum(t.get("fee_usd", 0) for t in entries)
    all_sizes = [t.get("size_btc", 0) for t in entries]
    all_confidences = [t.get("confidence", 0) for t in entries]
    stats["total_notional_usd"] = all_notional
    stats["total_fees_usd"] = all_fees
    stats["avg_size_btc"] = sum(all_sizes) / len(all_sizes)
    stats["avg_confidence"] = sum(all_confidences) / len(all_confidences)

    # Average duration
    durations = [t.get("duration_minutes", 0) for t in entries if t.get("duration_minutes")]
    if durations:
        stats["avg_duration_minutes"] = sum(durations) / len(durations)

    # Risk/reward analysis
    rr_ratios = []
    for t in entries:
        try:
            entry = float(t.get("fill_price") or t.get("entry_price", 0))
            tp = float(t.get("take_profit", 0))
            sl = float(t.get("stop_loss", 0))
        except (ValueError, TypeError):
            continue
        if entry and tp and sl:
            risk = abs(entry - sl)
            reward = abs(tp - entry)
            if risk > 0:
                rr_ratios.append(reward / risk)
    if rr_ratios:
        stats["avg_risk_reward"] = sum(rr_ratios) / len(rr_ratios)

    # Rejection reasons
    if rejections:
        reasons = defaultdict(int)
        for r in rejections:
            reasons[r.get("rejection_reason", "unknown")] += 1
        stats["rejection_reasons"] = dict(reasons)

    return stats


def analyze_decisions(decisions: list[dict]) -> dict:
    """Analyze decision history from decision_history.json."""
    if not decisions:
        return {}

    stats = {"total_decisions": len(decisions)}

    # Direction distribution
    dir_counts = defaultdict(int)
    for d in decisions:
        dir_counts[d.get("direction", "UNKNOWN")] += 1
    stats["direction_distribution"] = dict(dir_counts)

    # Confidence stats
    confidences = [d.get("confidence", 0) for d in decisions]
    stats["avg_confidence"] = sum(confidences) / len(confidences)
    stats["min_confidence"] = min(confidences)
    stats["max_confidence"] = max(confidences)

    # Outcome stats
    outcomes = [d.get("outcome", {}).get("status", "UNKNOWN") for d in decisions]
    outcome_counts = defaultdict(int)
    for o in outcomes:
        outcome_counts[o] += 1
    stats["outcome_distribution"] = dict(outcome_counts)

    # Win rate from resolved decisions
    resolved = [d for d in decisions if d.get("outcome", {}).get("status") in ("WIN", "LOSS")]
    if resolved:
        wins = sum(1 for d in resolved if d["outcome"]["status"] == "WIN")
        stats["win_rate"] = wins / len(resolved)
        stats["resolved_count"] = len(resolved)

        # PnL from resolved
        pnls = [d["outcome"].get("pnl_usd", 0) for d in resolved if "pnl_usd" in d.get("outcome", {})]
        if pnls:
            stats["total_pnl_usd"] = sum(pnls)
            stats["avg_pnl_usd"] = sum(pnls) / len(pnls)

        # Confidence correlation: avg confidence for wins vs losses
        win_conf = [d.get("confidence", 0) for d in resolved if d["outcome"]["status"] == "WIN"]
        loss_conf = [d.get("confidence", 0) for d in resolved if d["outcome"]["status"] == "LOSS"]
        if win_conf:
            stats["avg_win_confidence"] = sum(win_conf) / len(win_conf)
        if loss_conf:
            stats["avg_loss_confidence"] = sum(loss_conf) / len(loss_conf)

    return stats


def analyze_portfolio(snapshots: list[dict]) -> dict:
    """Analyze portfolio snapshots for equity curve."""
    if not snapshots:
        return {}

    stats = {"snapshot_count": len(snapshots)}
    equities = [(s.get("timestamp", ""), s.get("equity", 0)) for s in snapshots]
    equity_vals = [e for _, e in equities]

    if equity_vals:
        stats["first_equity"] = equity_vals[0]
        stats["last_equity"] = equity_vals[-1]
        stats["min_equity"] = min(equity_vals)
        stats["max_equity"] = max(equity_vals)
        if equity_vals[0] > 0:
            stats["total_return_pct"] = (equity_vals[-1] - equity_vals[0]) / equity_vals[0] * 100

        # Equity curve data points (for external plotting)
        stats["equity_curve"] = [
            {"ts": ts, "equity": eq} for ts, eq in equities
        ]

    return stats


def print_report(trade_stats: dict, decision_stats: dict, portfolio_stats: dict):
    """Print formatted performance report."""
    print("=" * 60)
    print("  TRADING PERFORMANCE REPORT")
    print("=" * 60)

    # Trade summary
    print("\n--- Trade Summary ---")
    print(f"  Total entries:    {trade_stats.get('total_trades', 0)}")
    print(f"  Total rejections: {trade_stats.get('total_rejections', 0)}")
    print(f"  Rejection rate:   {trade_stats.get('rejection_rate', 0):.1%}")

    if trade_stats.get("total_trades", 0) > 0:
        print(f"  Avg size:         {trade_stats.get('avg_size_btc', 0):.4f} BTC")
        print(f"  Total notional:   ${trade_stats.get('total_notional_usd', 0):,.2f}")
        print(f"  Total fees:       ${trade_stats.get('total_fees_usd', 0):,.2f}")
        print(f"  Avg confidence:   {trade_stats.get('avg_confidence', 0):.1%}")
        if "avg_duration_minutes" in trade_stats:
            print(f"  Avg duration:     {trade_stats['avg_duration_minutes']:.0f} min")
        if "avg_risk_reward" in trade_stats:
            print(f"  Avg R:R (planned):1:{trade_stats['avg_risk_reward']:.2f}")

    # Per-direction
    if "per_direction" in trade_stats:
        print("\n--- Per Direction ---")
        for direction, ds in trade_stats["per_direction"].items():
            print(f"  {direction}: {ds['count']} trades, "
                  f"avg {ds['avg_size_btc']:.4f} BTC, "
                  f"conf {ds['avg_confidence']:.1%}")

    # Decision quality
    if decision_stats:
        print("\n--- Decision Quality ---")
        print(f"  Total decisions:  {decision_stats.get('total_decisions', 0)}")
        print(f"  Avg confidence:   {decision_stats.get('avg_confidence', 0):.1%}")
        print(f"  Confidence range: {decision_stats.get('min_confidence', 0):.1%} — "
              f"{decision_stats.get('max_confidence', 0):.1%}")

        if "direction_distribution" in decision_stats:
            dist = decision_stats["direction_distribution"]
            parts = [f"{k}: {v}" for k, v in sorted(dist.items())]
            print(f"  Directions:       {', '.join(parts)}")

        if "outcome_distribution" in decision_stats:
            dist = decision_stats["outcome_distribution"]
            parts = [f"{k}: {v}" for k, v in sorted(dist.items())]
            print(f"  Outcomes:         {', '.join(parts)}")

        if "win_rate" in decision_stats:
            print(f"  Win rate:         {decision_stats['win_rate']:.1%} "
                  f"({decision_stats.get('resolved_count', 0)} resolved)")
        if "total_pnl_usd" in decision_stats:
            print(f"  Total PnL:        ${decision_stats['total_pnl_usd']:+,.2f}")
            print(f"  Avg PnL/trade:    ${decision_stats.get('avg_pnl_usd', 0):+,.2f}")
        if "avg_win_confidence" in decision_stats:
            print(f"  Avg win conf:     {decision_stats['avg_win_confidence']:.1%}")
        if "avg_loss_confidence" in decision_stats:
            print(f"  Avg loss conf:    {decision_stats['avg_loss_confidence']:.1%}")

    # Rejection reasons
    if "rejection_reasons" in trade_stats:
        print("\n--- Rejection Reasons ---")
        for reason, count in sorted(trade_stats["rejection_reasons"].items(),
                                     key=lambda x: -x[1]):
            print(f"  {count}x  {reason}")

    # Portfolio / equity curve
    if portfolio_stats:
        print("\n--- Equity Curve ---")
        print(f"  Snapshots:        {portfolio_stats.get('snapshot_count', 0)}")
        if "first_equity" in portfolio_stats:
            print(f"  First equity:     ${portfolio_stats['first_equity']:,.2f}")
            print(f"  Last equity:      ${portfolio_stats['last_equity']:,.2f}")
            print(f"  Min equity:       ${portfolio_stats['min_equity']:,.2f}")
            print(f"  Max equity:       ${portfolio_stats['max_equity']:,.2f}")
        if "total_return_pct" in portfolio_stats:
            print(f"  Total return:     {portfolio_stats['total_return_pct']:+.2f}%")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze trading performance")
    parser.add_argument("--state-dir", default="state_data",
                        help="Directory with JSONL state files (default: state_data)")
    parser.add_argument("--decision-history",
                        default="llm_agent/decision_history.json",
                        help="Path to decision history JSON")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of formatted text")
    args = parser.parse_args()

    # Load data
    trades = load_jsonl(os.path.join(args.state_dir, "trades.jsonl"))
    portfolio_snapshots = load_jsonl(os.path.join(args.state_dir, "portfolio.jsonl"))
    decisions = load_json(args.decision_history)
    if isinstance(decisions, dict):
        decisions = [decisions]

    # Analyze
    trade_stats = analyze_trades(trades)
    decision_stats = analyze_decisions(decisions)
    portfolio_stats = analyze_portfolio(portfolio_snapshots)

    if args.json:
        output = {
            "trades": trade_stats,
            "decisions": decision_stats,
            "portfolio": portfolio_stats,
        }
        # Remove equity_curve from JSON for brevity unless piped
        json.dump(output, sys.stdout, indent=2, default=str)
        print()
    else:
        print_report(trade_stats, decision_stats, portfolio_stats)


if __name__ == "__main__":
    main()
