"""Select a portfolio of N uncorrelated strategies meeting Sharpe threshold.

Greedy selector:
- Filter to strategies with Sharpe >= min_sharpe and num_trades >= min_trades
- Start with the highest Sharpe
- Iteratively add the next-highest-Sharpe strategy whose max correlation to
  any already-selected strategy is below max_corr
- Stop when target_n reached or no more candidates

Outputs the selected portfolio and writes selection to config/portfolio.yaml.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from strategies.engine_v2 import (
    SimConfig, backtest, load_5m_price, compute_signal_correlations,
)
from strategies.backtest_v2 import get_all_strategies, load_overlays, cfg_for_strategy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-sharpe", type=float, default=2.0)
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--max-corr", type=float, default=0.5)
    p.add_argument("--target-n", type=int, default=10)
    p.add_argument("--no-overlays", action="store_true")
    args = p.parse_args()

    print("Loading 5-min price...")
    price = load_5m_price()
    print(f"  {len(price):,} bars\n")

    overlays = {} if args.no_overlays else load_overlays()
    base = SimConfig(fee_per_side=0.0005)

    results = []
    for s in get_all_strategies():
        scfg = cfg_for_strategy(base, s, overlays)
        try:
            r = backtest(s, price, scfg)
            print(f"  {s.name:<34} Sharpe={r.metrics.get('sharpe', 0):>5.2f}  "
                  f"Trades={r.metrics.get('num_trades', 0):>4d}  "
                  f"TF={r.bar_seconds//3600}h")
            results.append(r)
        except Exception as e:
            print(f"  {s.name}: ERROR {e}")

    # Filter eligible
    eligible = [r for r in results if r.metrics.get("sharpe", 0) >= args.min_sharpe
                and r.metrics.get("num_trades", 0) >= args.min_trades]
    eligible.sort(key=lambda r: r.metrics["sharpe"], reverse=True)

    print(f"\n{len(eligible)} eligible strategies (Sharpe >= {args.min_sharpe}, "
          f"trades >= {args.min_trades}):")
    for r in eligible:
        print(f"  {r.strategy:<34} Sharpe={r.metrics['sharpe']:.2f}  "
              f"Trades={r.metrics['num_trades']}  TF={r.bar_seconds//3600}h")

    if not eligible:
        print("\nNo eligible strategies. Lower thresholds or run optimizer.")
        return

    # Compute correlation
    print("\nComputing daily-return correlations...")
    corr = compute_signal_correlations(eligible)

    # Greedy selection by Sharpe with correlation gate
    selected = []
    remaining = list(eligible)
    while remaining and len(selected) < args.target_n:
        # Take highest-Sharpe remaining
        cand = remaining.pop(0)
        if not selected:
            selected.append(cand)
            continue
        # Check max corr against current selection
        max_c = 0.0
        for sel in selected:
            try:
                c = abs(corr.loc[cand.strategy, sel.strategy])
            except KeyError:
                c = 0.0
            max_c = max(max_c, c)
        if max_c <= args.max_corr:
            selected.append(cand)
        # else skip

    print(f"\nSELECTED PORTFOLIO ({len(selected)} strategies, "
          f"max pairwise corr <= {args.max_corr}):")
    print("-" * 90)
    print(f"{'#':>3} {'Strategy':<34} {'Sharpe':>7} {'Trades':>7} {'TF':>4} {'MaxDD':>8}")
    print("-" * 90)
    for i, r in enumerate(selected, 1):
        m = r.metrics
        tf = f"{r.bar_seconds // 86400}d" if r.bar_seconds >= 86400 else f"{r.bar_seconds // 3600}h"
        print(f"{i:>3} {r.strategy:<34} {m['sharpe']:>7.2f} {m['num_trades']:>7d} "
              f"{tf:>4} {m['max_drawdown']:>7.1%}")

    # Show correlation matrix among selected
    sel_names = [r.strategy for r in selected]
    sub = corr.loc[sel_names, sel_names]
    print("\nCorrelation matrix:")
    print(f"{'':>34}" + "".join(f"{n[:11]:>12}" for n in sel_names))
    for n in sel_names:
        row = f"{n[:33]:>34}"
        for n2 in sel_names:
            row += f"{sub.loc[n, n2]:>12.2f}"
        print(row)

    # Save selection
    out = Path("config") / "portfolio.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategies": [
            {
                "name": r.strategy,
                "sharpe": float(r.metrics["sharpe"]),
                "trades": int(r.metrics["num_trades"]),
                "bar_seconds": int(r.bar_seconds),
                "max_drawdown": float(r.metrics["max_drawdown"]),
            }
            for r in selected
        ],
        "criteria": {
            "min_sharpe": args.min_sharpe,
            "min_trades": args.min_trades,
            "max_corr": args.max_corr,
        },
    }
    out.write_text(yaml.dump(payload, sort_keys=False))
    print(f"\nSaved portfolio to {out}")


if __name__ == "__main__":
    main()
