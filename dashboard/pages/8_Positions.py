"""Positions — equity curve, drawdown, win-rate, PnL by direction, recent fills."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import state, plots  # noqa: E402

st.set_page_config(page_title="Positions", page_icon="💰", layout="wide")
st.title("💰 Positions, P&L, Performance")

with st.sidebar:
    st.markdown("### Refresh")
    auto = st.checkbox("Auto-refresh", value=True)
    if auto:
        st_autorefresh(interval=10_000, key="positions_autorefresh")
    history_n = st.number_input("History points", min_value=50, max_value=10_000,
                                value=500, step=50)


# ---------------- top metrics ----------------
pf = state.latest_portfolio()
pf_df = state.portfolio_df(tail=int(history_n))

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    eq = (pf or {}).get("equity")
    st.metric("Current equity", f"${eq:,.2f}" if isinstance(eq, (int, float)) else "—")
with c2:
    fc = (pf or {}).get("free_collateral")
    st.metric("Free collateral", f"${fc:,.2f}" if isinstance(fc, (int, float)) else "—")
with c3:
    mp = (pf or {}).get("margin_pct")
    st.metric("Margin %", f"{mp}%" if mp is not None else "—")
with c4:
    n_pos = len((pf or {}).get("positions") or [])
    st.metric("Open positions", n_pos)
with c5:
    if not pf_df.empty and "equity" in pf_df.columns:
        first_eq = pf_df["equity"].iloc[0]
        last_eq = pf_df["equity"].iloc[-1]
        pct = (last_eq / first_eq - 1) * 100 if first_eq else 0
        st.metric("Total return (window)", f"{pct:+.2f}%")
    else:
        st.metric("Total return (window)", "—")


# ---------------- equity + drawdown ----------------
st.markdown("---")
st.markdown("### Equity & drawdown")
fig = plots.equity_with_drawdown(pf_df)
st.plotly_chart(fig, use_container_width=True)


# ---------------- decision outcomes ----------------
st.markdown("---")
st.markdown("### Decision outcomes (from llm_agent/decision_history.json)")

hist = state.llm_decision_history()
if hist:
    rows = []
    for h in hist:
        rows.append({
            "timestamp": h.get("timestamp"),
            "direction": h.get("direction"),
            "confidence": h.get("confidence"),
            "entry": h.get("entry_price"),
            "tp": h.get("take_profit"),
            "sl": h.get("stop_loss"),
            "outcome": h.get("outcome"),
            "realized_pnl_pct": h.get("realized_pnl_pct"),
            "resolved_at": h.get("resolved_at"),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # Outcome rollup
    outcomes = df["outcome"].dropna().astype(str)
    counts = outcomes.value_counts()
    rollup_cols = st.columns(max(4, len(counts)))
    for i, (k, v) in enumerate(counts.items()):
        rollup_cols[i % len(rollup_cols)].metric(k, int(v))

    # PnL by direction
    pnl_col = "realized_pnl_pct"
    if pnl_col in df.columns:
        by_dir = df.dropna(subset=[pnl_col, "direction"]).groupby("direction")[pnl_col].agg(
            ["count", "mean", "sum"]
        ).round(4)
        if not by_dir.empty:
            st.markdown("**P&L (%) by direction**")
            st.dataframe(by_dir, use_container_width=True)

        # Cumulative-PnL chart
        chrono = df.dropna(subset=[pnl_col]).sort_values("timestamp")
        if not chrono.empty:
            chrono["cum_pnl_pct"] = chrono[pnl_col].cumsum()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=chrono["timestamp"], y=chrono["cum_pnl_pct"],
                                       mode="lines+markers", name="Cumulative realized %",
                                       line=dict(color="#2ca02c", width=2)))
            fig2.update_layout(height=330, hovermode="x unified",
                               yaxis_title="Cum. realized P&L %",
                               margin=dict(l=40, r=20, t=30, b=40))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Recent decisions**")
    st.dataframe(state.arrow_safe(df.head(50)), use_container_width=True, hide_index=True)
else:
    st.info("No decision history available yet.")


# ---------------- trades ledger ----------------
st.markdown("---")
st.markdown("### Trades ledger (state_data/trades.jsonl)")
tdf = state.trades_df(tail=200)
if tdf.empty:
    st.info("No trades recorded yet.")
else:
    if "timestamp" in tdf.columns:
        tdf = tdf.sort_values("timestamp", ascending=False).reset_index(drop=True)
    st.dataframe(state.arrow_safe(tdf.head(200)), use_container_width=True, hide_index=True)
