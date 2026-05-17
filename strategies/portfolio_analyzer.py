"""Portfolio-level analysis: equal-weight combine N strategies and compute
the combined daily-returns Sharpe, drawdown, etc.

Useful when individual strategies have Sharpe ~1.0-1.5 but uncorrelated
combination delivers portfolio Sharpe ≥ 2.0.

Usage:
    python -m strategies.portfolio_analyzer
"""

import argparse

import numpy as np
import pandas as pd

from strategies.engine_v2 import SimConfig, backtest, load_5m_price, compute_signal_correlations
from strategies.backtest_v2 import get_all_strategies, load_overlays, cfg_for_strategy


def to_daily_returns(r) -> pd.Series:
    if len(r.daily_returns) == 0:
        return pd.Series(dtype=float)
    ts = pd.to_datetime(r.timestamps, unit="ms", utc=True)
    s = pd.Series(r.daily_returns, index=ts)
    return (1 + s).resample("1D").prod() - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-sharpe", type=float, default=0.8,
                   help="Minimum individual Sharpe to include")
    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--max-corr", type=float, default=0.6)
    p.add_argument("--target-n", type=int, default=10)
    args = p.parse_args()

    print("Loading 5-min price...")
    price = load_5m_price()
    overlays = load_overlays()
    base = SimConfig(fee_per_side=0.0005)

    print("Backtesting all strategies...\n")
    results = []
    for s in get_all_strategies():
        cfg = cfg_for_strategy(base, s, overlays)
        try:
            r = backtest(s, price, cfg)
            results.append(r)
        except Exception as e:
            print(f"  ERROR {s.name}: {e}")

    # Sort by Sharpe
    results.sort(key=lambda r: r.metrics.get("sharpe", -99), reverse=True)

    # Filter
    eligible = [r for r in results
                 if r.metrics.get("sharpe", 0) >= args.min_sharpe
                 and r.metrics.get("num_trades", 0) >= args.min_trades]
    print(f"{len(eligible)} eligible strategies (Sharpe>={args.min_sharpe}, "
          f"Trades>={args.min_trades})")

    if len(eligible) < 2:
        print("Not enough strategies.")
        return

    # Correlation matrix
    corr = compute_signal_correlations(eligible)

    # Greedy selection
    selected = []
    for r in eligible:
        if not selected:
            selected.append(r)
            continue
        max_c = max(abs(corr.loc[r.strategy, s.strategy]) for s in selected
                    if r.strategy in corr.index and s.strategy in corr.columns)
        if max_c <= args.max_corr:
            selected.append(r)
        if len(selected) >= args.target_n:
            break

    print(f"\n=== INDIVIDUAL STRATEGIES IN PORTFOLIO ({len(selected)}) ===")
    print(f"{'#':>3}  {'Strategy':<32}{'Sharpe':>8}{'Sortino':>8}{'Trades':>8}"
          f"{'TF':>5}{'MaxDD':>8}")
    print("-" * 90)
    for i, r in enumerate(selected, 1):
        m = r.metrics
        tf = (f"{r.bar_seconds // 86400}d" if r.bar_seconds >= 86400
              else f"{r.bar_seconds // 3600}h")
        print(f"{i:>3}  {r.strategy:<32}{m['sharpe']:>8.2f}{m['sortino']:>8.2f}"
              f"{m['num_trades']:>8d}{tf:>5}{m['max_drawdown']:>7.1%}")

    # Equal-weight portfolio: combine daily returns
    print("\n=== EQUAL-WEIGHT PORTFOLIO (1/N allocation across selected) ===")
    if not selected:
        return
    daily_series = []
    for r in selected:
        daily_series.append(to_daily_returns(r))
    if not daily_series:
        return

    # Align on common date index
    all_dates = sorted(set().union(*[s.index for s in daily_series]))
    common_index = pd.DatetimeIndex(all_dates)
    df = pd.DataFrame(index=common_index)
    for r, s in zip(selected, daily_series):
        df[r.strategy] = s.reindex(common_index).fillna(0.0)

    n = len(selected)
    cols = [r.strategy for r in selected]

    # Equal weight
    df["eq_portfolio"] = df[cols].sum(axis=1) / n

    # Inverse-vol weight (risk parity): w_i ~ 1/sigma_i, normalized
    vols = df[cols].std(ddof=1)
    inv_vol = 1.0 / vols.replace(0, np.nan)
    weights = inv_vol / inv_vol.sum()
    df["rp_portfolio"] = (df[cols] * weights).sum(axis=1)

    for label, pcol in (("EQUAL-WEIGHT", "eq_portfolio"),
                        ("RISK-PARITY (1/vol)", "rp_portfolio")):
        pr = df[pcol]
        mu = pr.mean()
        sd = pr.std(ddof=1)
        sharpe = mu / sd * np.sqrt(365.25) if sd > 0 else 0
        downside = pr[pr < 0]
        dsd = downside.std(ddof=1)
        sortino = mu / dsd * np.sqrt(365.25) if dsd > 0 else 0

        equity = (1 + pr).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / peak
        max_dd = abs(dd.min())

        years = (df.index[-1] - df.index[0]).days / 365.25 if len(df) > 1 else 0.1
        total_return = equity.iloc[-1] - 1
        ann_return = (1 + total_return) ** (1 / years) - 1

        print(f"\n  --- {label} ---")
        print(f"  Days in series:    {len(pr)}")
        print(f"  Portfolio Sharpe:  {sharpe:.2f}")
        print(f"  Portfolio Sortino: {sortino:.2f}")
        print(f"  Total return:      {total_return:.1%}")
        print(f"  Annual return:     {ann_return:.1%}")
        print(f"  Max drawdown:      {max_dd:.1%}")
        print(f"  Calmar:            {ann_return / max_dd if max_dd > 0 else 0:.2f}")

    # Correlations among selected
    print("\nCorrelation matrix (selected):")
    sel_names = [r.strategy for r in selected]
    sub = corr.loc[sel_names, sel_names]
    print(f"{'':>32}" + "".join(f"{n[:11]:>12}" for n in sel_names))
    for n in sel_names:
        row = f"{n[:31]:>32}"
        for n2 in sel_names:
            row += f"{sub.loc[n, n2]:>12.2f}"
        print(row)


if __name__ == "__main__":
    main()
