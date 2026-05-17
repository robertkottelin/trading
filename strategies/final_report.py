"""Compare walk-forward vs full-history params and select final portfolio.

Loads strategy_overlays.yaml (latest = full-history) and strategy_overlays_wf.yaml
(walk-forward backup). Runs each strategy under both configurations on the full
history and shows side-by-side metrics. Then greedy-selects an uncorrelated
portfolio.

Usage:
    python -m strategies.final_report
    python -m strategies.final_report --min-sharpe 2.0 --max-corr 0.5
"""

import argparse
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from strategies.engine_v2 import SimConfig, backtest, load_5m_price, compute_signal_correlations
from strategies.backtest_v2 import get_all_strategies, cfg_for_strategy


def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def _apply_strategy_params(strategy, params: dict):
    if not params:
        return
    for k, v in params.items():
        if hasattr(strategy, k):
            setattr(strategy, k, v)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-sharpe", type=float, default=2.0)
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--max-corr", type=float, default=0.5)
    p.add_argument("--target-n", type=int, default=10)
    p.add_argument("--mode", choices=("fh", "wf", "best"), default="best",
                   help="fh=full-history params, wf=walk-forward params, best=max of both")
    args = p.parse_args()

    print("Loading 5-min price...")
    price = load_5m_price()
    print(f"  {len(price):,} bars")

    # Load both param sets
    fh_strat = _load_yaml("config/strategy_params.yaml")
    fh_overlay = _load_yaml("config/strategy_overlays.yaml")
    wf_strat = _load_yaml("config/strategy_params_wf.yaml")
    wf_overlay = _load_yaml("config/strategy_overlays_wf.yaml")

    base = SimConfig(fee_per_side=0.0005)

    print("\n" + "=" * 130)
    print(f"{'Strategy':<32}{'FH Sharpe':>11}{'FH Trd':>8}{'WF Sharpe':>11}{'WF Trd':>8}"
          f"{'Best':>8}{'TF':>5}{'MaxDD':>8}")
    print("-" * 130)

    results = []  # list of (strategy_name, fh_metrics, wf_metrics, best_metrics_for_corr, sim_result)
    for s in get_all_strategies():
        klass_lc = type(s).__name__.lower()

        # FH run
        s_fh = type(s)()
        _apply_strategy_params(s_fh, fh_strat.get(klass_lc, {}))
        cfg_fh = cfg_for_strategy(base, s_fh, fh_overlay)
        try:
            r_fh = backtest(s_fh, price, cfg_fh)
            m_fh = r_fh.metrics
        except Exception as e:
            m_fh = {"sharpe": -99, "num_trades": 0, "error": str(e)}
            r_fh = None

        # WF run
        s_wf = type(s)()
        _apply_strategy_params(s_wf, wf_strat.get(klass_lc, {}))
        cfg_wf = cfg_for_strategy(base, s_wf, wf_overlay)
        try:
            r_wf = backtest(s_wf, price, cfg_wf)
            m_wf = r_wf.metrics
        except Exception as e:
            m_wf = {"sharpe": -99, "num_trades": 0, "error": str(e)}
            r_wf = None

        fh_s = m_fh.get("sharpe", 0)
        wf_s = m_wf.get("sharpe", 0)
        fh_n = m_fh.get("num_trades", 0)
        wf_n = m_wf.get("num_trades", 0)
        best_s = max(fh_s, wf_s)

        tf_sec = (r_fh or r_wf).bar_seconds if (r_fh or r_wf) else 86400
        tf = (f"{tf_sec // 86400}d" if tf_sec >= 86400
              else f"{tf_sec // 3600}h")

        max_dd = max(m_fh.get("max_drawdown", 0), m_wf.get("max_drawdown", 0))
        print(f"{s.name:<32}{fh_s:>11.2f}{fh_n:>8d}{wf_s:>11.2f}{wf_n:>8d}"
              f"{best_s:>8.2f}{tf:>5}{max_dd:>7.1%}")

        # Pick the better of FH/WF for correlation analysis
        if args.mode == "fh":
            chosen = (m_fh, r_fh, "fh")
        elif args.mode == "wf":
            chosen = (m_wf, r_wf, "wf")
        else:  # best
            if fh_s >= wf_s:
                chosen = (m_fh, r_fh, "fh")
            else:
                chosen = (m_wf, r_wf, "wf")
        results.append((s.name, m_fh, m_wf, chosen[0], chosen[1], chosen[2]))

    # Filter eligible
    eligible = []
    for name, m_fh, m_wf, m_best, r_best, src in results:
        if r_best is None:
            continue
        if m_best.get("sharpe", 0) < args.min_sharpe:
            continue
        if m_best.get("num_trades", 0) < args.min_trades:
            continue
        eligible.append((name, m_best, r_best, src))

    eligible.sort(key=lambda x: x[1]["sharpe"], reverse=True)

    print()
    print(f"{len(eligible)} strategies pass Sharpe >= {args.min_sharpe} "
          f"& Trades >= {args.min_trades}:")
    for name, m, r, src in eligible:
        print(f"  ✓ {name:<32} Sharpe={m['sharpe']:.2f} Trades={m['num_trades']:>4d} [{src}]")

    if len(eligible) < 2:
        print("\nNot enough eligible strategies for portfolio selection.")
        return

    # Correlation among eligible
    print("\nComputing daily-return correlations among eligible...")
    sim_results = [r for _, _, r, _ in eligible]
    corr = compute_signal_correlations(sim_results)

    # Greedy selection with correlation gate
    selected = []
    for name, m, r, src in eligible:
        if not selected:
            selected.append((name, m, r, src))
            continue
        max_c = 0.0
        for sname, _, _, _ in selected:
            try:
                c = abs(corr.loc[name, sname])
            except KeyError:
                c = 0.0
            max_c = max(max_c, c)
        if max_c <= args.max_corr:
            selected.append((name, m, r, src))
        if len(selected) >= args.target_n:
            break

    print(f"\nSELECTED PORTFOLIO ({len(selected)} of {args.target_n} target, "
          f"max corr <= {args.max_corr}):")
    print("-" * 100)
    print(f"{'#':>3} {'Strategy':<32}{'Sharpe':>8}{'Trades':>8}{'TF':>5}{'MaxDD':>8} {'Source':>7}")
    print("-" * 100)
    for i, (name, m, r, src) in enumerate(selected, 1):
        tf = (f"{r.bar_seconds // 86400}d" if r.bar_seconds >= 86400
              else f"{r.bar_seconds // 3600}h")
        print(f"{i:>3} {name:<32}{m['sharpe']:>8.2f}{m['num_trades']:>8d}{tf:>5}"
              f"{m['max_drawdown']:>7.1%} {src:>7}")

    if selected:
        sel_names = [n for n, _, _, _ in selected]
        sub = corr.loc[sel_names, sel_names]
        print("\nCorrelation matrix (selected):")
        print(f"{'':>32}" + "".join(f"{n[:11]:>12}" for n in sel_names))
        for n in sel_names:
            row = f"{n[:31]:>32}"
            for n2 in sel_names:
                row += f"{sub.loc[n, n2]:>12.2f}"
            print(row)

    # Save selection
    out = Path("config/portfolio.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategies": [
            {
                "name": n,
                "sharpe": float(m["sharpe"]),
                "trades": int(m["num_trades"]),
                "bar_seconds": int(r.bar_seconds),
                "max_drawdown": float(m["max_drawdown"]),
                "source": src,
            }
            for n, m, r, src in selected
        ],
        "criteria": {
            "min_sharpe": args.min_sharpe,
            "min_trades": args.min_trades,
            "max_corr": args.max_corr,
        },
    }
    out.write_text(yaml.dump(payload, sort_keys=False))
    print(f"\nSaved selection → {out}")


if __name__ == "__main__":
    main()
