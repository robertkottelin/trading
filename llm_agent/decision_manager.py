"""Decision Manager — read/write decision.json + decision_history.json.

Tracks current decision and historical outcomes. On each run, resolves
pending decisions by checking if TP/SL were hit or duration expired.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

AGENT_DIR = Path("llm_agent")
DECISION_FILE = AGENT_DIR / "decision.json"
HISTORY_FILE = AGENT_DIR / "decision_history.json"
CONTEXT_DIR = Path("market_context_data")


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log.warning("Failed to load %s: %s", path, e)
        return None


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _load_history() -> list:
    data = _load_json(HISTORY_FILE)
    return data if isinstance(data, list) else []


def _load_klines_since(ts_iso: str) -> pd.DataFrame | None:
    """Load dYdX 5m candles since a given ISO timestamp for outcome tracking."""
    path = CONTEXT_DIR / "dydx_candles_5m.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["_ts"] = pd.to_datetime(df["timestamp"], utc=True)
            entry_ts = pd.Timestamp(ts_iso, tz="UTC")
            df = df[df["_ts"] >= entry_ts].sort_values("_ts")
        for col in ["high", "low", "close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        log.warning("Failed to load klines for outcome tracking: %s", e)
        return None


def resolve_pending() -> int:
    """Check pending decisions and resolve outcomes.

    For each pending decision, scans kline data since entry to determine
    if TP/SL were hit or duration expired.

    Returns:
        Number of decisions resolved.
    """
    history = _load_history()
    if not history:
        return 0

    resolved_count = 0
    for entry in history:
        if entry.get("outcome", {}).get("status") != "PENDING":
            continue

        direction = entry.get("direction", "")
        if direction == "NO_TRADE":
            entry["outcome"] = {"status": "NO_TRADE", "pnl_pct": 0.0}
            resolved_count += 1
            continue

        entry_price = entry.get("entry_price")
        tp = entry.get("take_profit")
        sl = entry.get("stop_loss")
        duration_min = entry.get("duration_minutes", 120)
        ts = entry.get("timestamp")

        if not all([entry_price, tp, sl, ts]):
            continue

        klines = _load_klines_since(ts)
        if klines is None or klines.empty:
            continue

        # Check candle by candle
        entry_ts = pd.Timestamp(ts, tz="UTC")
        expiry_ts = entry_ts + pd.Timedelta(minutes=duration_min)

        # Only check candles strictly after entry and up to expiry + 1 candle
        klines = klines[klines["_ts"] > entry_ts]
        klines = klines[klines["_ts"] <= expiry_ts + pd.Timedelta(minutes=5)]
        if klines.empty:
            continue

        outcome = None
        for _, candle in klines.iterrows():
            candle_ts = candle["_ts"]
            c_high = candle["high"]
            c_low = candle["low"]
            c_close = candle["close"]

            # Skip candles with NaN price data — pd.to_numeric(errors="coerce")
            # produces NaN for bad values, and NaN comparisons always return
            # False, which would silently miss TP/SL hits.
            if pd.isna(c_high) or pd.isna(c_low) or pd.isna(c_close):
                log.debug("Skipping candle at %s with NaN price data", candle_ts)
                continue

            if direction == "LONG":
                # Check SL first (conservative — SL before TP if both hit in same candle)
                if c_low <= sl:
                    pnl = (sl - entry_price) / entry_price * 100
                    outcome = {
                        "status": "SL_HIT",
                        "exit_price": sl,
                        "pnl_pct": round(pnl, 4),
                        "duration_actual_minutes": int((candle_ts - entry_ts).total_seconds() / 60),
                        "resolved_at": candle_ts.isoformat(),
                    }
                    break
                if c_high >= tp:
                    pnl = (tp - entry_price) / entry_price * 100
                    outcome = {
                        "status": "TP_HIT",
                        "exit_price": tp,
                        "pnl_pct": round(pnl, 4),
                        "duration_actual_minutes": int((candle_ts - entry_ts).total_seconds() / 60),
                        "resolved_at": candle_ts.isoformat(),
                    }
                    break
            elif direction == "SHORT":
                if c_high >= sl:
                    pnl = (entry_price - sl) / entry_price * 100
                    outcome = {
                        "status": "SL_HIT",
                        "exit_price": sl,
                        "pnl_pct": round(pnl, 4),
                        "duration_actual_minutes": int((candle_ts - entry_ts).total_seconds() / 60),
                        "resolved_at": candle_ts.isoformat(),
                    }
                    break
                if c_low <= tp:
                    pnl = (entry_price - tp) / entry_price * 100
                    outcome = {
                        "status": "TP_HIT",
                        "exit_price": tp,
                        "pnl_pct": round(pnl, 4),
                        "duration_actual_minutes": int((candle_ts - entry_ts).total_seconds() / 60),
                        "resolved_at": candle_ts.isoformat(),
                    }
                    break

            # Check expiry
            if candle_ts >= expiry_ts:
                if direction == "LONG":
                    pnl = (c_close - entry_price) / entry_price * 100
                else:
                    pnl = (entry_price - c_close) / entry_price * 100
                outcome = {
                    "status": "EXPIRED",
                    "exit_price": c_close,
                    "pnl_pct": round(pnl, 4),
                    "duration_actual_minutes": duration_min,
                    "resolved_at": candle_ts.isoformat(),
                }
                break

        if outcome:
            entry["outcome"] = outcome
            resolved_count += 1
            log.info("Resolved decision %s: %s (PnL: %.4f%%)",
                     ts, outcome["status"], outcome["pnl_pct"])

    if resolved_count > 0:
        _save_json(HISTORY_FILE, history)

    return resolved_count


def get_recent_summary(n: int = 15) -> str:
    """Get a formatted summary of the last n decisions for the LLM prompt.

    Includes win-rate trends (last 5 vs last 10), consecutive SL streak detection,
    and a RISK_LEVEL / RECOMMENDATION block the LLM uses to self-adjust thresholds.
    """
    history = _load_history()
    if not history:
        return "RECENT DECISIONS: No previous decisions."

    recent = history[-n:]
    lines = [f"RECENT DECISIONS (last {len(recent)}):"]

    wins = 0
    losses = 0
    total_pnl = 0.0
    trades = 0

    # Collect resolved trade statuses in order (oldest first) for trend analysis
    resolved_in_order: list[str] = []

    for i, entry in enumerate(recent, 1):
        ts = entry.get("timestamp", "?")[:16]
        direction = entry.get("direction", "?")
        outcome = entry.get("outcome", {})
        status = outcome.get("status", "PENDING")

        if direction == "NO_TRADE":
            lines.append(f"  #{i}: {ts} NO_TRADE")
            continue

        pnl = outcome.get("pnl_pct", 0)
        dur = outcome.get("duration_actual_minutes", "?")

        if status == "PENDING":
            lines.append(f"  #{i}: {ts} {direction} -> PENDING")
        else:
            lines.append(f"  #{i}: {ts} {direction} -> {status} {pnl:+.2f}% in {dur}min")
            trades += 1
            total_pnl += pnl
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            resolved_in_order.append(status)

    # --- Overall window stats ---
    if trades > 0:
        win_rate = wins / trades * 100
        avg_pnl = total_pnl / trades
        lines.append(
            f"  Window stats ({trades} resolved): "
            f"win rate {wins}/{trades} ({win_rate:.0f}%), avg PnL {avg_pnl:+.2f}%"
        )

    # --- Recent trend: last 5 vs last 10 ---
    if len(resolved_in_order) >= 5:
        last5 = resolved_in_order[-5:]
        last10 = resolved_in_order[-10:] if len(resolved_in_order) >= 10 else resolved_in_order
        l5_tp = sum(1 for s in last5 if s == "TP_HIT")
        l5_sl = sum(1 for s in last5 if s == "SL_HIT")
        l10_tp = sum(1 for s in last10 if s == "TP_HIT")
        l10_sl = sum(1 for s in last10 if s == "SL_HIT")
        lines.append(
            f"  Trend — last 5 resolved: {l5_tp} TP / {l5_sl} SL | "
            f"last {len(last10)} resolved: {l10_tp} TP / {l10_sl} SL"
        )

    # --- Consecutive SL streak (walk backwards through resolved outcomes) ---
    consecutive_sl = 0
    for s in reversed(resolved_in_order):
        if s == "SL_HIT":
            consecutive_sl += 1
        else:
            break

    # --- RISK_LEVEL determination ---
    last5_sl_count = sum(1 for s in resolved_in_order[-5:] if s == "SL_HIT") if resolved_in_order else 0

    if consecutive_sl >= 4:
        risk_level = "HIGH"
        risk_rec = (
            f"CONSECUTIVE_SL_STREAK={consecutive_sl}. "
            "Raise effective confidence threshold to 0.80+. "
            "Require ML models AND strategy signals AND confirming price action to all agree. "
            "Strongly prefer NO_TRADE — waiting for a clearly superior setup has positive EV."
        )
    elif consecutive_sl >= 2 or last5_sl_count >= 3:
        risk_level = "ELEVATED"
        risk_rec = (
            f"CONSECUTIVE_SL_STREAK={consecutive_sl}, recent SL rate elevated. "
            "Raise effective confidence threshold to 0.70+. "
            "Require 3 independent signal types (e.g., ML + strategy + market context) before entering."
        )
    else:
        risk_level = "NORMAL"
        risk_rec = (
            "Standard thresholds apply: confidence >= 0.65, 2+ independent signal types aligned."
        )

    lines.append(f"  RISK_LEVEL: {risk_level}")
    lines.append(f"  RECOMMENDATION: {risk_rec}")

    return "\n".join(lines)


def save_decision(decision: dict):
    """Save a new decision to decision.json and append to history.

    Skips appending to history if there is already a PENDING decision
    with the same direction that hasn't expired yet (prevents decision
    spam when the pipeline runs every 5 minutes).

    Args:
        decision: Dict with direction, confidence, entry_price, take_profit,
                  stop_loss, duration_minutes, position_size_pct, rationale,
                  market_conditions, model_consensus.
    """
    # Ensure timestamp
    if "timestamp" not in decision:
        decision["timestamp"] = datetime.now(timezone.utc).isoformat()

    # Save as current decision
    _save_json(DECISION_FILE, decision)
    log.info("Saved decision to %s", DECISION_FILE)

    direction = decision.get("direction", "")

    # Check for existing unexpired PENDING decision with same direction
    history = _load_history()
    if direction not in ("NO_TRADE", ""):
        now = datetime.now(timezone.utc)
        for entry in reversed(history):
            if entry.get("outcome", {}).get("status") != "PENDING":
                continue
            if entry.get("direction") != direction:
                continue
            # Check if the pending decision has expired
            entry_ts = entry.get("timestamp")
            entry_dur = entry.get("duration_minutes", 120)
            if entry_ts:
                try:
                    entry_time = datetime.fromisoformat(entry_ts)
                    if entry_time.tzinfo is None:
                        entry_time = entry_time.replace(tzinfo=timezone.utc)
                    expiry = entry_time + pd.Timedelta(minutes=entry_dur)
                    if now < expiry:
                        log.info(
                            "Skipping history append — active PENDING %s decision "
                            "from %s (expires in %d min)",
                            direction, entry_ts[:16],
                            int((expiry - now).total_seconds() / 60),
                        )
                        return
                except (ValueError, TypeError):
                    pass

    # Append to history with PENDING outcome (unless NO_TRADE)
    history_entry = dict(decision)
    if direction == "NO_TRADE":
        history_entry["outcome"] = {"status": "NO_TRADE", "pnl_pct": 0.0}
    else:
        history_entry["outcome"] = {"status": "PENDING"}
    history.append(history_entry)
    _save_json(HISTORY_FILE, history)
    log.info("Appended decision to history (%d total)", len(history))


def get_current_decision() -> dict | None:
    """Load the current (latest) decision."""
    return _load_json(DECISION_FILE)
