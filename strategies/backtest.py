"""Backtesting engine for conventional strategies.

Runs walk-forward backtests on all strategies using historical data from raw_data/.
Computes performance metrics and generates comparison reports.

Usage:
    python -m strategies.backtest              # Backtest all strategies
    python -m strategies.backtest --strategy 1  # Backtest single strategy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from strategies.funding_rate import FundingRateReversion
from strategies.volatility_regime import VolatilityRegime
from strategies.macro_regime import MacroRegime
from strategies.liquidation_flow import LiquidationFlow
from strategies.sentiment_flow import SentimentFlow
from strategies.basis_reversion import BasisReversion
from strategies.trend_following import TrendFollowing
from strategies.momentum_composite import MomentumComposite

DATA_DIR = "raw_data"
FEE_PER_SIDE = 0.0013  # 10bps fee + 3bps slippage


def get_all_strategies() -> list[BaseStrategy]:
    return [
        FundingRateReversion(),
        VolatilityRegime(),
        MacroRegime(),
        LiquidationFlow(),
        SentimentFlow(),
        BasisReversion(),
        TrendFollowing(),
        MomentumComposite(),
    ]


def load_price_reference() -> pd.DataFrame:
    """Load Binance futures daily OHLC as price reference."""
    path = Path(DATA_DIR) / "binance_futures_klines_5m.csv"
    df = pd.read_csv(path)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df["ts_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df["date"] = df["datetime"].dt.date

    daily = df.groupby("date").agg(
        open=("open", "first"),
        close=("close", "last"),
        high=("close", "max"),
        low=("close", "min"),
    ).reset_index()
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def backtest_strategy(strategy: BaseStrategy, price_df: pd.DataFrame,
                       start_date=None, end_date=None) -> dict:
    """Run backtest on a single strategy.

    Returns dict with trades, equity curve, and metrics.
    """
    # Load data and compute signals
    data = strategy.load_data(DATA_DIR)
    signal_df = strategy.compute_signal_series(data)

    if signal_df.empty:
        return {"error": "No signals generated", "strategy": strategy.name}

    # Ensure date columns are compatible
    signal_df["date"] = pd.to_datetime(signal_df["date"]).dt.date
    price_df = price_df.copy()

    # Merge signals with price
    merged = price_df.merge(signal_df[["date", "signal", "confidence"]], on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    if start_date:
        merged = merged[merged["date"] >= start_date]
    if end_date:
        merged = merged[merged["date"] <= end_date]

    if len(merged) < 10:
        return {"error": "Insufficient data after merge", "strategy": strategy.name}

    # Simulate trades
    trades = []
    equity = [1.0]  # Start with $1 normalized
    daily_returns = []

    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_date = None

    for i in range(1, len(merged)):
        sig = int(merged["signal"].iloc[i])
        prev_sig = int(merged["signal"].iloc[i - 1])
        price = merged["close"].iloc[i]
        next_open = merged["open"].iloc[i]  # Enter at this bar's open

        # Daily mark-to-market return
        if position != 0:
            prev_close = merged["close"].iloc[i - 1]
            daily_ret = position * (price - prev_close) / prev_close
        else:
            daily_ret = 0.0

        daily_returns.append(daily_ret)
        equity.append(equity[-1] * (1 + daily_ret))

        # Signal changed → close old position, open new one
        if sig != prev_sig:
            # Close existing position
            if position != 0:
                exit_price = next_open
                raw_ret = position * (exit_price - entry_price) / entry_price
                net_ret = raw_ret - 2 * FEE_PER_SIDE  # Round-trip fees
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": merged["date"].iloc[i],
                    "direction": "LONG" if position == 1 else "SHORT",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "raw_return": raw_ret,
                    "net_return": net_ret,
                    "hold_days": (merged["date"].iloc[i] - entry_date).days
                        if hasattr(merged["date"].iloc[i], '__sub__') else 0,
                })
                position = 0

            # Open new position
            if sig != 0:
                position = sig
                entry_price = next_open
                entry_date = merged["date"].iloc[i]

    # Close any remaining position at end
    if position != 0:
        exit_price = merged["close"].iloc[-1]
        raw_ret = position * (exit_price - entry_price) / entry_price
        net_ret = raw_ret - 2 * FEE_PER_SIDE
        trades.append({
            "entry_date": entry_date,
            "exit_date": merged["date"].iloc[-1],
            "direction": "LONG" if position == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "raw_return": raw_ret,
            "net_return": net_ret,
            "hold_days": (merged["date"].iloc[-1] - entry_date).days
                if hasattr(merged["date"].iloc[-1], '__sub__') else 0,
        })

    # Compute metrics
    metrics = compute_metrics(trades, daily_returns, merged)
    metrics["strategy"] = strategy.name
    metrics["num_signals"] = len(merged)
    metrics["date_range"] = f"{merged['date'].iloc[0]} to {merged['date'].iloc[-1]}"

    return {
        "metrics": metrics,
        "trades": trades,
        "equity": equity,
        "daily_returns": daily_returns,
        "strategy": strategy.name,
    }


def compute_metrics(trades: list, daily_returns: list, merged_df: pd.DataFrame) -> dict:
    """Compute comprehensive backtest metrics."""
    if not trades:
        return {
            "total_return": 0, "annual_return": 0, "sharpe": 0,
            "max_drawdown": 0, "win_rate": 0, "num_trades": 0,
            "avg_return": 0, "avg_hold_days": 0, "profit_factor": 0,
            "calmar": 0, "trades_per_year": 0,
        }

    net_returns = [t["net_return"] for t in trades]
    winners = [r for r in net_returns if r > 0]
    losers = [r for r in net_returns if r <= 0]

    # Total return (compounded)
    total_return = 1.0
    for r in net_returns:
        total_return *= (1 + r)
    total_return -= 1

    # Time span
    first_date = merged_df["date"].iloc[0]
    last_date = merged_df["date"].iloc[-1]
    try:
        days = (last_date - first_date).days
    except (TypeError, AttributeError):
        days = len(merged_df)
    years = max(days / 365.25, 0.1)

    # Annualized return
    annual_return = (1 + total_return) ** (1 / years) - 1

    # Sharpe ratio (from daily returns)
    dr = np.array(daily_returns)
    dr = dr[~np.isnan(dr)]
    if len(dr) > 10 and dr.std() > 0:
        sharpe = (dr.mean() / dr.std()) * np.sqrt(365)
    else:
        sharpe = 0

    # Max drawdown
    cumulative = np.cumprod(1 + dr)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

    # Win rate
    win_rate = len(winners) / len(trades) if trades else 0

    # Profit factor
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0.001
    profit_factor = gross_profit / gross_loss

    # Average metrics
    avg_return = np.mean(net_returns) if net_returns else 0
    avg_hold = np.mean([t["hold_days"] for t in trades]) if trades else 0

    # Calmar ratio
    calmar = annual_return / max_dd if max_dd > 0 else 0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "num_trades": len(trades),
        "avg_return": avg_return,
        "avg_hold_days": avg_hold,
        "profit_factor": profit_factor,
        "calmar": calmar,
        "trades_per_year": len(trades) / years,
    }


def run_period_analysis(strategy: BaseStrategy, price_df: pd.DataFrame) -> dict:
    """Run backtests on multiple time periods for robustness check."""
    from datetime import date

    periods = [
        ("2020-2022", date(2020, 1, 1), date(2021, 12, 31)),
        ("2022-2024", date(2022, 1, 1), date(2023, 12, 31)),
        ("2024-2026", date(2024, 1, 1), date(2026, 3, 1)),
        ("Full", None, None),
    ]

    results = {}
    for name, start, end in periods:
        result = backtest_strategy(strategy, price_df, start, end)
        if "error" in result:
            results[name] = {"sharpe": 0, "error": result["error"]}
        else:
            results[name] = result["metrics"]

    return results


def print_results(results: list[dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 120)
    print("STRATEGY BACKTEST RESULTS")
    print("=" * 120)
    print(f"{'Strategy':<30} {'Return':>8} {'Annual':>8} {'Sharpe':>7} {'MaxDD':>7} "
          f"{'WinRate':>8} {'Trades':>7} {'Trades/Y':>9} {'AvgRet':>8} {'PF':>6} {'Calmar':>7}")
    print("-" * 120)

    for r in results:
        if "error" in r:
            print(f"{r['strategy']:<30} ERROR: {r['error']}")
            continue

        m = r["metrics"]
        print(f"{m['strategy']:<30} "
              f"{m['total_return']:>7.1%} "
              f"{m['annual_return']:>7.1%} "
              f"{m['sharpe']:>7.2f} "
              f"{m['max_drawdown']:>6.1%} "
              f"{m['win_rate']:>7.1%} "
              f"{m['num_trades']:>7d} "
              f"{m['trades_per_year']:>8.1f} "
              f"{m['avg_return']:>7.2%} "
              f"{m['profit_factor']:>5.2f} "
              f"{m['calmar']:>7.2f}")

    print("=" * 120)


def print_period_analysis(all_periods: dict):
    """Print period-by-period robustness analysis."""
    print("\n" + "=" * 100)
    print("PERIOD ROBUSTNESS ANALYSIS")
    print("=" * 100)

    for strat_name, periods in all_periods.items():
        print(f"\n  {strat_name}:")
        sharpes = []
        for period_name, metrics in periods.items():
            if "error" in metrics:
                print(f"    {period_name:<12} ERROR: {metrics['error']}")
                continue
            s = metrics.get("sharpe", 0)
            sharpes.append(s)
            print(f"    {period_name:<12} Sharpe={s:>6.2f}  "
                  f"Return={metrics.get('total_return', 0):>7.1%}  "
                  f"Trades={metrics.get('num_trades', 0):>4d}  "
                  f"WinRate={metrics.get('win_rate', 0):>5.1%}")

        if sharpes:
            min_s = min(sharpes)
            avg_s = np.mean(sharpes)
            print(f"    {'→ Min Sharpe:':<12} {min_s:.2f}  Avg: {avg_s:.2f}")


def compute_signal_correlations(strategies: list[BaseStrategy], price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise signal correlations across strategies."""
    signal_series = {}
    for strat in strategies:
        data = strat.load_data(DATA_DIR)
        sig = strat.compute_signal_series(data)
        if not sig.empty:
            sig["date"] = pd.to_datetime(sig["date"]).dt.date
            signal_series[strat.name] = sig.set_index("date")["signal"]

    if len(signal_series) < 2:
        return pd.DataFrame()

    # Align all signals on a common date index (deduplicated)
    all_dates = set()
    for s in signal_series.values():
        all_dates.update(s.index.tolist())
    all_dates = sorted(all_dates)
    date_index = pd.Index(all_dates)

    combined = pd.DataFrame(index=date_index)
    for name, s in signal_series.items():
        # Deduplicate: keep last value per date
        s = s[~s.index.duplicated(keep="last")]
        combined[name] = s.reindex(date_index)
    combined = combined.fillna(0)

    return combined.corr()


def rank_strategies(results: list[dict], period_results: dict,
                     corr_matrix: pd.DataFrame) -> list[tuple[str, float]]:
    """Rank strategies by composite score for final selection."""
    scores = []

    for r in results:
        if "error" in r:
            continue

        m = r["metrics"]
        name = m["strategy"]

        # 40% Sharpe
        sharpe_score = max(m["sharpe"], 0) * 0.4

        # 30% Robustness (minimum Sharpe across periods)
        periods = period_results.get(name, {})
        period_sharpes = [p.get("sharpe", 0) for p in periods.values()
                          if isinstance(p, dict) and "sharpe" in p]
        min_sharpe = min(period_sharpes) if period_sharpes else 0
        robustness_score = max(min_sharpe, 0) * 0.3

        # 15% Trade frequency (target ~100+ trades per year)
        freq = m["trades_per_year"]
        freq_score = min(freq / 100, 1.0) * 0.15

        # 15% Uncorrelation (average absolute correlation with other strategies)
        if not corr_matrix.empty and name in corr_matrix.columns:
            other_corrs = corr_matrix[name].drop(name, errors="ignore").abs()
            avg_corr = other_corrs.mean() if len(other_corrs) > 0 else 0
            uncorr_score = (1 - avg_corr) * 0.15
        else:
            uncorr_score = 0.1

        total = sharpe_score + robustness_score + freq_score + uncorr_score
        scores.append((name, total, m["sharpe"], min_sharpe, freq))

    scores.sort(key=lambda x: -x[1])
    return scores


def main():
    parser = argparse.ArgumentParser(description="Backtest conventional strategies")
    parser.add_argument("--strategy", type=int, help="Strategy index (1-7)")
    args = parser.parse_args()

    print("Loading price reference...")
    price_df = load_price_reference()
    print(f"Price data: {len(price_df)} daily bars, "
          f"{price_df['date'].iloc[0]} to {price_df['date'].iloc[-1]}")

    strategies = get_all_strategies()

    if args.strategy:
        strategies = [strategies[args.strategy - 1]]

    # Run full backtests
    print(f"\nRunning backtests on {len(strategies)} strategies...")
    results = []
    for strat in strategies:
        print(f"  Backtesting: {strat.name}...")
        result = backtest_strategy(strat, price_df)
        results.append(result)

    print_results(results)

    # Period analysis
    print("\nRunning period robustness analysis...")
    all_periods = {}
    for strat in strategies:
        print(f"  Analyzing: {strat.name}...")
        all_periods[strat.name] = run_period_analysis(strat, price_df)

    print_period_analysis(all_periods)

    # Signal correlations
    if len(strategies) > 1:
        print("\nComputing signal correlations...")
        corr = compute_signal_correlations(strategies, price_df)
        if not corr.empty:
            print("\nSIGNAL CORRELATION MATRIX:")
            # Format correlation matrix
            names = list(corr.columns)
            header = f"{'':>30}" + "".join(f"{n[:12]:>14}" for n in names)
            print(header)
            for i, name in enumerate(names):
                row = f"{name:>30}"
                for j, name2 in enumerate(names):
                    val = corr.iloc[i, j]
                    row += f"{val:>14.3f}"
                print(row)

        # Rank strategies
        print("\n" + "=" * 80)
        print("STRATEGY RANKING (composite score)")
        print("=" * 80)
        rankings = rank_strategies(results, all_periods, corr)
        for i, (name, score, sharpe, min_s, freq) in enumerate(rankings):
            marker = " ✓ SELECTED" if i < 5 else "   backup"
            print(f"  {i+1}. {name:<30} Score={score:.3f}  "
                  f"Sharpe={sharpe:.2f}  MinSharpe={min_s:.2f}  "
                  f"Freq={freq:.0f}/yr{marker}")


if __name__ == "__main__":
    main()
