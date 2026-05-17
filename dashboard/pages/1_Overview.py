"""Overview — live snapshot of bot state, latest decision, current positions."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import state  # noqa: E402

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
st.title("📊 Overview")


# ---------------- controls ----------------
with st.sidebar:
    st.markdown("### Refresh")
    auto = st.checkbox("Auto-refresh", value=True)
    interval_s = st.slider("Interval (s)", min_value=2, max_value=60, value=5)
    if auto:
        st_autorefresh(interval=interval_s * 1000, key="overview_autorefresh")


# ---------------- top metrics ----------------
hb = state.heartbeat()
pf = state.latest_portfolio()
dec = state.latest_decision()
rs = state.retrain_state()
paused = state.is_paused()
age = state.heartbeat_age_seconds()


def _format_age(s):
    if s is None:
        return "—"
    if s < 60:
        return f"{int(s)}s ago"
    if s < 3600:
        return f"{int(s/60)}m ago"
    return f"{s/3600:.1f}h ago"


c1, c2, c3, c4 = st.columns(4)
with c1:
    status = (hb or {}).get("status", "—")
    color = {"running": "🟢", "completed": "🟢", "failed": "🔴"}.get(status, "⚪")
    st.metric(f"{color} Bot status", status,
              delta=f"run #{(hb or {}).get('run_number', '?')}" if hb else None)
with c2:
    equity = (pf or {}).get("equity")
    margin = (pf or {}).get("margin_pct")
    st.metric("Equity (USD)",
              f"${equity:,.2f}" if isinstance(equity, (int, float)) else "—",
              delta=f"margin {margin}%" if margin is not None else None)
with c3:
    direction = (dec or {}).get("direction", "—")
    conf = (dec or {}).get("confidence")
    emoji = {"LONG": "🟢", "SHORT": "🔴", "NO_TRADE": "⚪"}.get(direction, "⚪")
    st.metric(f"{emoji} Latest decision", direction,
              delta=f"conf {conf:.2f}" if isinstance(conf, (int, float)) else None)
with c4:
    st.metric("Heartbeat age", _format_age(age),
              delta=f"interval {state.heartbeat()['next_run_at'][11:19] if hb and 'next_run_at' in hb else ''}"
              if hb and 'next_run_at' in hb else None)

if paused:
    st.warning("⏸️ Pause flag is active — bot is in position-management-only mode (no new entries).")


# ---------------- decision panel ----------------
st.markdown("---")
st.markdown("### Latest decision")

if not dec:
    st.info("No decision recorded yet.")
else:
    a, b = st.columns([1, 2])
    with a:
        direction = dec.get("direction", "—")
        emoji = {"LONG": "🟢", "SHORT": "🔴", "NO_TRADE": "⚪"}.get(direction, "⚪")
        st.markdown(f"**Direction:** {emoji} `{direction}`")
        conf = dec.get("confidence")
        st.markdown(f"**Confidence:** `{conf:.2f}`" if conf is not None else "**Confidence:** —")
        st.markdown(f"**Entry:** `{dec.get('entry_price', '—')}`")
        st.markdown(f"**Take profit:** `{dec.get('take_profit', '—')}`")
        st.markdown(f"**Stop loss:** `{dec.get('stop_loss', '—')}`")
        st.markdown(f"**Position size USD:** `{dec.get('position_size_usd', '—')}`")
        st.markdown(f"**Duration:** `{dec.get('duration_minutes', '—')}` min")
        st.markdown(f"**Timestamp:** `{dec.get('timestamp', '—')}`")
    with b:
        st.markdown("**Rationale:**")
        st.write(dec.get("rationale", "—"))

        mc = dec.get("model_consensus", {}) or {}
        cond = dec.get("market_conditions", {}) or {}
        st.markdown("**Model consensus:**")
        st.json({
            "bullish": mc.get("bullish_count"),
            "bearish": mc.get("bearish_count"),
            "neutral": mc.get("neutral_count"),
            "weighted_score": mc.get("weighted_score"),
            "avg_raw_score": mc.get("avg_raw_score"),
        })
        st.markdown("**Market conditions:**")
        st.json({
            "btc_price": cond.get("btc_price"),
            "funding_rate": cond.get("funding_rate"),
            "fng_value": cond.get("fng_value"),
        })


# ---------------- positions ----------------
st.markdown("---")
st.markdown("### Open positions")

positions = (pf or {}).get("positions") or []
if not positions:
    st.info("No open positions.")
else:
    import pandas as pd
    df = pd.DataFrame(positions)
    st.dataframe(state.arrow_safe(df), use_container_width=True, hide_index=True)


# ---------------- retrain status ----------------
st.markdown("---")
st.markdown("### Retraining")
if rs:
    cols = st.columns(4)
    cols[0].metric("Status", rs.get("status", "—"))
    cols[1].metric("Last deployed", (rs.get("last_deployed") or "—")[:19])
    cols[2].metric("Last started", (rs.get("last_retrain_started") or "—")[:19])
    cols[3].metric("Last PID", rs.get("pid") or "—")
    if rs.get("failure_reason"):
        st.error(f"Last failure: {rs['failure_reason']}")
else:
    st.caption("No retrain_state.json yet.")
