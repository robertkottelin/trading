"""Position-vector backtest engine (v2) — realistic PnL with fees, stops, vol targeting.

Design principles:
- Daily/intra-bar PnL accounts for the BAR-OPEN handoff at signal-flip bars
- Fees deducted from the daily PnL on the bar where position size changes
- Optional stop-loss / take-profit triggered intra-bar from bar high/low
- Optional vol-targeted position sizing (annualized vol cap, capped leverage)
- Optional confidence-weighted sizing; confidence-min gate
- Multi-timeframe: any equal-spaced bar grid (5m, 1h, 4h, 1d, 1w)
- Total return = compounded daily_returns (consistent with Sharpe)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy

log = logging.getLogger(__name__)

DATA_DIR = "raw_data"


@dataclass
class SimConfig:
    """Universal simulation parameters applied across strategies."""
    fee_per_side: float = 0.0005           # 5 bps each side — realistic for dYdX maker+slip
    stop_loss_pct: Optional[float] = None  # e.g. 0.05 = stop at 5% adverse intra-bar
    take_profit_pct: Optional[float] = None  # e.g. 0.10 = take 10% intra-bar
    max_hold_bars: Optional[int] = None    # force exit after N bars
    confidence_min: float = 0.0            # signals below this confidence are flat
    sizing: str = "binary"                 # "binary" | "confidence" | "vol_target"
    vol_target_annual: float = 0.40        # used if sizing == "vol_target"
    vol_lookback_bars: int = 30            # realized-vol window for vol targeting
    leverage_max: float = 1.0              # cap on |position|


@dataclass
class SimResult:
    daily_returns: np.ndarray
    equity: np.ndarray
    position: np.ndarray
    trades: list
    metrics: dict
    bar_seconds: int
    timestamps: np.ndarray
    strategy: str
    signal_series: pd.DataFrame = field(default_factory=pd.DataFrame)


# --------------------------------------------------------------------------- #
#  Price loaders
# --------------------------------------------------------------------------- #


def load_5m_price(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Load Binance 5-min futures klines, sorted, no duplicates."""
    path = Path(data_dir) / "binance_futures_klines_5m.csv"
    df = pd.read_csv(path)
    for c in ["open", "high", "low", "close", "volume", "open_time_ms"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open_time_ms", "close"])
    df = df.sort_values("open_time_ms").drop_duplicates("open_time_ms").reset_index(drop=True)
    df["ts_ms"] = df["open_time_ms"].astype(np.int64)
    return df[["ts_ms", "open", "high", "low", "close", "volume"]]


def resample_to_bars(df_5m: pd.DataFrame, bar_seconds: int) -> pd.DataFrame:
    """Aggregate 5-min OHLCV to any equal-spaced bar grid."""
    if bar_seconds == 300:
        return df_5m.copy()
    bar_ms = bar_seconds * 1000
    df = df_5m.copy()
    df["bucket"] = (df["ts_ms"] // bar_ms) * bar_ms
    agg = df.groupby("bucket").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    agg = agg.rename(columns={"bucket": "ts_ms"}).sort_values("ts_ms").reset_index(drop=True)
    return agg


# --------------------------------------------------------------------------- #
#  Signal alignment
# --------------------------------------------------------------------------- #


def _signal_to_ts_ms(signal_df: pd.DataFrame, bar_seconds: int) -> pd.DataFrame:
    """Normalize a strategy signal_df to have a ts_ms column at the bar boundary.

    Accepts:
    - ts_ms column directly
    - date column (YYYY-MM-DD or datetime.date) — converted to UTC midnight
    """
    if signal_df.empty:
        return pd.DataFrame(columns=["ts_ms", "signal", "confidence"])

    df = signal_df.copy()
    if "ts_ms" not in df.columns:
        if "date" not in df.columns:
            raise ValueError("signal_df must have ts_ms or date column")
        ts = pd.to_datetime(df["date"], utc=True)
        df["ts_ms"] = (ts.astype(np.int64) // 1_000_000).astype(np.int64)

    if "confidence" not in df.columns:
        df["confidence"] = 1.0

    # Snap signal timestamps to bar boundaries
    bar_ms = bar_seconds * 1000
    df["ts_ms"] = (df["ts_ms"] // bar_ms) * bar_ms

    df = df[["ts_ms", "signal", "confidence"]].sort_values("ts_ms")
    df = df.drop_duplicates("ts_ms", keep="last").reset_index(drop=True)
    df["signal"] = df["signal"].fillna(0).astype(int).clip(-1, 1)
    df["confidence"] = df["confidence"].fillna(0.0).clip(0.0, 1.0).astype(float)
    return df


def align_signal_to_price(price_df: pd.DataFrame, signal_df: pd.DataFrame,
                          bar_seconds: int) -> pd.DataFrame:
    """Merge signals onto price grid; forward-fill so positions persist between signals.

    IMPORTANT: signals are SHIFTED FORWARD BY ONE BAR to prevent look-ahead bias.
    A signal computed using OHLC data of bar t is only known after t closes,
    so it can only drive position changes from the OPEN of bar t+1 onwards.
    """
    sigs = _signal_to_ts_ms(signal_df, bar_seconds)
    px = price_df[["ts_ms", "open", "high", "low", "close"]].copy()
    merged = px.merge(sigs, on="ts_ms", how="left")
    # Forward-fill signal & confidence (no signal yet => 0 / 0)
    merged["signal"] = merged["signal"].ffill().fillna(0).astype(int)
    merged["confidence"] = merged["confidence"].ffill().fillna(0.0)
    # Shift by +1 bar so we trade on the NEXT bar after the signal was generated
    merged["signal"] = merged["signal"].shift(1).fillna(0).astype(int)
    merged["confidence"] = merged["confidence"].shift(1).fillna(0.0)
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Position simulator (vectorized core, scalar fallback for stops)
# --------------------------------------------------------------------------- #


def _compute_target_position(merged: pd.DataFrame, cfg: SimConfig,
                              bar_seconds: int) -> np.ndarray:
    """Compute the *target* position at each bar from signal + sizing rules.

    This is the position you'd want to be holding at the bar's open BEFORE
    intra-bar stop-loss/take-profit considerations.
    """
    sig = merged["signal"].to_numpy(dtype=float)
    conf = merged["confidence"].to_numpy(dtype=float)

    # Confidence gate
    if cfg.confidence_min > 0:
        sig = np.where(conf >= cfg.confidence_min, sig, 0.0)

    # Sizing
    if cfg.sizing == "binary":
        size = np.ones_like(sig)
    elif cfg.sizing == "confidence":
        size = np.clip(conf, 0.0, 1.0)
    elif cfg.sizing == "vol_target":
        close = merged["close"].to_numpy(dtype=float)
        log_ret = np.zeros_like(close)
        log_ret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-12))
        # Realized vol over lookback window, annualized
        bars_per_year = 365.25 * 86400 / bar_seconds
        rv = pd.Series(log_ret).rolling(cfg.vol_lookback_bars, min_periods=5).std()
        rv = rv.bfill().fillna(0.0).to_numpy()
        annual_vol = rv * np.sqrt(bars_per_year)
        size = np.where(annual_vol > 1e-6, cfg.vol_target_annual / annual_vol, 0.0)
        size = np.clip(size, 0.0, cfg.leverage_max)
    else:
        raise ValueError(f"Unknown sizing mode: {cfg.sizing}")

    target = sig * size
    target = np.clip(target, -cfg.leverage_max, cfg.leverage_max)
    return target


def simulate(merged: pd.DataFrame, cfg: SimConfig, bar_seconds: int) -> SimResult:
    """Run the position simulator.

    Bar semantics:
    - At bar t, OPEN price is observed. New target position is established at OPEN.
    - During bar t: position moves from prev held to new target → fee paid on |delta|.
    - Intra-bar: high/low may trigger stop-loss or take-profit at the stop price.
    - At bar t close: mark-to-market PnL realized.
    """
    n = len(merged)
    opens = merged["open"].to_numpy(dtype=float)
    highs = merged["high"].to_numpy(dtype=float)
    lows = merged["low"].to_numpy(dtype=float)
    closes = merged["close"].to_numpy(dtype=float)
    ts = merged["ts_ms"].to_numpy(dtype=np.int64)

    target = _compute_target_position(merged, cfg, bar_seconds)

    # Position actually held during each bar (after considering stops & max hold)
    position = np.zeros(n)
    daily_returns = np.zeros(n)

    # Trade tracking
    trades: list[dict] = []
    cur_pos = 0.0
    cur_entry_price = 0.0
    cur_entry_idx = -1

    for i in range(1, n):
        prev_pos = cur_pos
        new_target = target[i]

        # Apply max-hold forced exit
        if cfg.max_hold_bars is not None and cur_pos != 0 and \
                (i - cur_entry_idx) >= cfg.max_hold_bars:
            new_target = 0.0

        # Position changes at this bar's OPEN
        delta = new_target - prev_pos
        fee_cost = abs(delta) * cfg.fee_per_side
        # Trade event accounting at position change
        if delta != 0.0:
            # If closing all or part of an existing position, log the trade
            if prev_pos != 0 and (np.sign(new_target) != np.sign(prev_pos) or new_target == 0.0):
                exit_price = opens[i]
                raw_ret = prev_pos * (exit_price - cur_entry_price) / cur_entry_price
                trades.append({
                    "entry_idx": cur_entry_idx,
                    "exit_idx": i,
                    "entry_ts_ms": int(ts[cur_entry_idx]) if cur_entry_idx >= 0 else 0,
                    "exit_ts_ms": int(ts[i]),
                    "direction": "LONG" if prev_pos > 0 else "SHORT",
                    "size": float(abs(prev_pos)),
                    "entry_price": float(cur_entry_price),
                    "exit_price": float(exit_price),
                    "raw_return": float(raw_ret),
                    "bars_held": i - cur_entry_idx,
                    "exit_reason": "signal",
                })

            # Update entry state if opening a fresh non-zero target
            if new_target != 0.0 and (np.sign(new_target) != np.sign(prev_pos) or prev_pos == 0):
                cur_entry_price = opens[i]
                cur_entry_idx = i
            cur_pos = new_target

        # Check intra-bar stop / take-profit on the position held during this bar
        held_pos = cur_pos
        exit_triggered = False
        exit_price_intrabar = closes[i]
        exit_reason = None
        if held_pos != 0.0 and cur_entry_price > 0:
            sl = cfg.stop_loss_pct
            tp = cfg.take_profit_pct
            if held_pos > 0:
                # LONG: stop at entry*(1-sl), take profit at entry*(1+tp)
                stop_price = cur_entry_price * (1 - sl) if sl else -np.inf
                tp_price = cur_entry_price * (1 + tp) if tp else np.inf
                # Pessimistic: if both could trigger, assume stop hits first
                if lows[i] <= stop_price and stop_price > 0:
                    exit_triggered = True
                    exit_price_intrabar = stop_price
                    exit_reason = "stop_loss"
                elif highs[i] >= tp_price:
                    exit_triggered = True
                    exit_price_intrabar = tp_price
                    exit_reason = "take_profit"
            else:
                # SHORT: stop at entry*(1+sl), take at entry*(1-tp)
                stop_price = cur_entry_price * (1 + sl) if sl else np.inf
                tp_price = cur_entry_price * (1 - tp) if tp else -np.inf
                if highs[i] >= stop_price and stop_price < np.inf:
                    exit_triggered = True
                    exit_price_intrabar = stop_price
                    exit_reason = "stop_loss"
                elif lows[i] <= tp_price and tp_price > 0:
                    exit_triggered = True
                    exit_price_intrabar = tp_price
                    exit_reason = "take_profit"

        # Bar PnL — split at OPEN handoff if position changed at this bar
        if delta != 0.0:
            # Old pos earned prev_close → open; new pos earns open → close (or intra-bar exit)
            if i == 0 or closes[i - 1] <= 0:
                pnl_old = 0.0
            else:
                pnl_old = prev_pos * (opens[i] - closes[i - 1]) / closes[i - 1]
            if exit_triggered:
                pnl_new = held_pos * (exit_price_intrabar - opens[i]) / opens[i]
            else:
                pnl_new = held_pos * (closes[i] - opens[i]) / opens[i]
            daily_returns[i] = pnl_old + pnl_new - fee_cost
        else:
            # No position change at open — full bar with held_pos
            if exit_triggered:
                pnl = held_pos * (exit_price_intrabar - closes[i - 1]) / closes[i - 1]
            elif closes[i - 1] > 0:
                pnl = held_pos * (closes[i] - closes[i - 1]) / closes[i - 1]
            else:
                pnl = 0.0
            daily_returns[i] = pnl - fee_cost  # fee_cost is 0 here since no delta

        # If stop/TP hit, close the position now and log trade
        if exit_triggered:
            raw_ret = held_pos * (exit_price_intrabar - cur_entry_price) / cur_entry_price
            trades.append({
                "entry_idx": cur_entry_idx,
                "exit_idx": i,
                "entry_ts_ms": int(ts[cur_entry_idx]) if cur_entry_idx >= 0 else 0,
                "exit_ts_ms": int(ts[i]),
                "direction": "LONG" if held_pos > 0 else "SHORT",
                "size": float(abs(held_pos)),
                "entry_price": float(cur_entry_price),
                "exit_price": float(exit_price_intrabar),
                "raw_return": float(raw_ret),
                "bars_held": i - cur_entry_idx,
                "exit_reason": exit_reason,
            })
            # Pay exit fee (model as a one-sided close fee)
            daily_returns[i] -= abs(held_pos) * cfg.fee_per_side
            cur_pos = 0.0
            cur_entry_idx = -1
            cur_entry_price = 0.0
            held_pos = 0.0

        position[i] = held_pos

    # Close any open position at last bar
    if cur_pos != 0.0:
        exit_price = closes[-1]
        raw_ret = cur_pos * (exit_price - cur_entry_price) / cur_entry_price
        trades.append({
            "entry_idx": cur_entry_idx,
            "exit_idx": n - 1,
            "entry_ts_ms": int(ts[cur_entry_idx]) if cur_entry_idx >= 0 else 0,
            "exit_ts_ms": int(ts[n - 1]),
            "direction": "LONG" if cur_pos > 0 else "SHORT",
            "size": float(abs(cur_pos)),
            "entry_price": float(cur_entry_price),
            "exit_price": float(exit_price),
            "raw_return": float(raw_ret),
            "bars_held": n - 1 - cur_entry_idx,
            "exit_reason": "end_of_data",
        })
        daily_returns[-1] -= abs(cur_pos) * cfg.fee_per_side

    # Equity curve from daily returns
    equity = np.cumprod(1.0 + daily_returns)

    # Metrics
    metrics = compute_metrics_v2(daily_returns, trades, ts, bar_seconds)

    return SimResult(
        daily_returns=daily_returns,
        equity=equity,
        position=position,
        trades=trades,
        metrics=metrics,
        bar_seconds=bar_seconds,
        timestamps=ts,
        strategy="",
    )


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #


def compute_metrics_v2(daily_returns: np.ndarray, trades: list, ts_ms: np.ndarray,
                       bar_seconds: int) -> dict:
    """Compute realistic backtest metrics from per-bar returns."""
    n = len(daily_returns)
    bars_per_year = 365.25 * 86400 / bar_seconds

    nz = daily_returns[~np.isnan(daily_returns)]
    if len(nz) < 10:
        return {
            "total_return": 0, "annual_return": 0, "sharpe": 0,
            "sortino": 0, "max_drawdown": 0, "calmar": 0,
            "num_trades": 0, "trades_per_year": 0,
            "win_rate": 0, "avg_return": 0, "profit_factor": 0,
            "avg_hold_bars": 0, "stops_pct": 0, "tps_pct": 0,
        }

    # Sharpe / Sortino — annualized
    mu = nz.mean()
    sd = nz.std(ddof=1) if len(nz) > 1 else 0.0
    sharpe = mu / sd * np.sqrt(bars_per_year) if sd > 0 else 0.0
    downside = nz[nz < 0]
    dsd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = mu / dsd * np.sqrt(bars_per_year) if dsd > 0 else 0.0

    # Compounded return
    equity = np.cumprod(1.0 + nz)
    total_return = equity[-1] - 1.0
    # Time span
    if len(ts_ms) >= 2:
        years = max((ts_ms[-1] - ts_ms[0]) / 1000 / (365.25 * 86400), 0.1)
    else:
        years = 0.1
    annual_return = (1 + total_return) ** (1 / years) - 1 if total_return > -0.999 else -1.0

    # Drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = abs(dd.min()) if len(dd) > 0 else 0.0
    calmar = annual_return / max_dd if max_dd > 0 else 0.0

    # Trade-level
    raw_rets = np.array([t["raw_return"] for t in trades]) if trades else np.array([])
    holds = np.array([t["bars_held"] for t in trades]) if trades else np.array([])
    wins = raw_rets[raw_rets > 0]
    losses = raw_rets[raw_rets <= 0]
    win_rate = len(wins) / len(raw_rets) if len(raw_rets) else 0.0
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() < 0 else 0.0
    avg_ret = raw_rets.mean() if len(raw_rets) else 0.0
    avg_hold = holds.mean() if len(holds) else 0.0
    stops = sum(1 for t in trades if t.get("exit_reason") == "stop_loss")
    tps = sum(1 for t in trades if t.get("exit_reason") == "take_profit")

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "num_trades": len(trades),
        "trades_per_year": float(len(trades) / years) if years > 0 else 0.0,
        "win_rate": float(win_rate),
        "avg_return": float(avg_ret),
        "profit_factor": float(pf),
        "avg_hold_bars": float(avg_hold),
        "stops_pct": float(stops / max(len(trades), 1)),
        "tps_pct": float(tps / max(len(trades), 1)),
    }


# --------------------------------------------------------------------------- #
#  Top-level: per-strategy backtest
# --------------------------------------------------------------------------- #


def get_strategy_bar_seconds(strategy: BaseStrategy) -> int:
    """Strategies may declare bar_seconds (default 86400 = daily)."""
    return int(getattr(strategy, "bar_seconds", 86400))


def backtest(strategy: BaseStrategy, price_5m: pd.DataFrame,
             cfg: SimConfig, data_dir: str = DATA_DIR,
             start_ms: int | None = None,
             end_ms: int | None = None,
             preloaded_data: dict | None = None,
             preloaded_price_bars: pd.DataFrame | None = None) -> SimResult:
    """Backtest a single strategy with the new engine.

    Passing ``preloaded_data`` (the dict returned by strategy.load_data) and
    ``preloaded_price_bars`` (the resampled OHLCV) lets the optimizer reuse
    these across trials — 10–50x speedup vs reloading every call.
    """
    bar_seconds = get_strategy_bar_seconds(strategy)
    if preloaded_price_bars is not None:
        price_bars = preloaded_price_bars
    else:
        price_bars = resample_to_bars(price_5m, bar_seconds)

    data = preloaded_data if preloaded_data is not None else strategy.load_data(data_dir)
    signal_df = strategy.compute_signal_series(data)
    if signal_df.empty:
        result = SimResult(np.array([]), np.array([]), np.array([]), [], {}, bar_seconds, np.array([]), strategy.name)
        result.metrics = {"strategy": strategy.name, "error": "no signals"}
        return result

    merged = align_signal_to_price(price_bars, signal_df, bar_seconds)

    if start_ms is not None:
        merged = merged[merged["ts_ms"] >= start_ms].reset_index(drop=True)
    if end_ms is not None:
        merged = merged[merged["ts_ms"] <= end_ms].reset_index(drop=True)

    if len(merged) < 20:
        result = SimResult(np.array([]), np.array([]), np.array([]), [], {}, bar_seconds, np.array([]), strategy.name)
        result.metrics = {"strategy": strategy.name, "error": "insufficient data"}
        return result

    res = simulate(merged, cfg, bar_seconds)
    res.strategy = strategy.name
    res.metrics["strategy"] = strategy.name
    res.signal_series = merged[["ts_ms", "signal", "confidence"]].copy()
    return res


def print_results(results: list[SimResult]) -> None:
    print("\n" + "=" * 130)
    print(f"{'Strategy':<32}{'TF':>5}{'Return':>10}{'AnnRet':>9}{'Sharpe':>8}"
          f"{'Sortino':>9}{'MaxDD':>8}{'Calmar':>8}{'Trades':>8}{'T/Yr':>7}"
          f"{'Win%':>7}{'AvgRet':>9}{'PF':>6}")
    print("-" * 130)
    for r in results:
        m = r.metrics
        if "error" in m:
            print(f"{m.get('strategy', '?'):<32} ERROR: {m['error']}")
            continue
        tf = _tf_label(r.bar_seconds)
        print(f"{m['strategy']:<32}{tf:>5}"
              f"{m['total_return']:>9.1%}"
              f"{m['annual_return']:>8.1%}"
              f"{m['sharpe']:>8.2f}"
              f"{m['sortino']:>9.2f}"
              f"{m['max_drawdown']:>7.1%}"
              f"{m['calmar']:>8.2f}"
              f"{m['num_trades']:>8d}"
              f"{m['trades_per_year']:>7.1f}"
              f"{m['win_rate']:>6.1%}"
              f"{m['avg_return']:>8.2%}"
              f"{m['profit_factor']:>6.2f}")
    print("=" * 130)


def _tf_label(bar_seconds: int) -> str:
    if bar_seconds < 3600:
        return f"{bar_seconds // 60}m"
    if bar_seconds < 86400:
        return f"{bar_seconds // 3600}h"
    if bar_seconds < 7 * 86400:
        return f"{bar_seconds // 86400}d"
    return f"{bar_seconds // (7 * 86400)}w"


def compute_signal_correlations(results: list[SimResult]) -> pd.DataFrame:
    """Correlation between strategy *daily-return* series (more meaningful than signal corr)."""
    series = {}
    for r in results:
        if len(r.daily_returns) == 0 or "error" in r.metrics:
            continue
        # Re-time series indexed by ts_ms
        s = pd.Series(r.daily_returns, index=pd.to_datetime(r.timestamps, unit="ms", utc=True))
        # Resample to daily for cross-strategy comparability
        daily = (1 + s).resample("1D").prod() - 1
        series[r.strategy] = daily

    if len(series) < 2:
        return pd.DataFrame()

    combined = pd.concat(series.values(), axis=1, keys=series.keys()).fillna(0.0)
    return combined.corr()
