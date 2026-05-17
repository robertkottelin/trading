"""Strategies — live status of all 18 conventional strategies."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import state, strategy_runner  # noqa: E402

st.set_page_config(page_title="Strategies", page_icon="⚡", layout="wide")
st.title("⚡ Conventional strategies (18)")

with st.sidebar:
    st.markdown("### Source data")
    data_dir = st.radio(
        "Data directory", ["market_context_data", "raw_data"], index=0,
        help="market_context_data is the live pipeline source; raw_data is backtest-style"
    )
    abs_dir = str(state.REPO_ROOT / data_dir)

    if st.button("🔄 Reload params from YAML"):
        # Force a fresh run that re-instantiates strategies (picks up new params)
        st.cache_data.clear()
        st.success("Cleared cache — next run will reload params.")

    rerun = st.button("▶️ Run strategies now", type="primary")


@st.cache_data(ttl=30, show_spinner=False)
def _run(data_dir: str):
    return strategy_runner.run_strategies(data_dir=data_dir)


# Pull a cached result (or fresh if rerun pressed)
if rerun:
    st.cache_data.clear()

with st.spinner("Running all strategies..."):
    result = _run(abs_dir)

if result["error"]:
    st.error(f"Strategy engine error: {result['error']}")
    st.stop()

signals = result["signals"]
consensus = result.get("consensus", {})


# ---------------- consensus summary ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("🟢 LONG", consensus.get("long_count", 0))
c2.metric("🔴 SHORT", consensus.get("short_count", 0))
c3.metric("⚪ INACTIVE", consensus.get("inactive_count", 0))
c4.metric("Total", len(signals))

# ---------------- strategy grid ----------------
st.markdown("---")

DIRECTION_COLORS = {"LONG": "#2ca02c", "SHORT": "#d62728", "INACTIVE": "#7f7f7f"}
DIRECTION_EMOJI = {"LONG": "🟢", "SHORT": "🔴", "INACTIVE": "⚪"}

names = list(signals.keys())
cols_per_row = 3
for row_start in range(0, len(names), cols_per_row):
    cols = st.columns(cols_per_row)
    for i, col in enumerate(cols):
        idx = row_start + i
        if idx >= len(names):
            break
        name = names[idx]
        s = signals[name]
        direction = s.get("direction", "INACTIVE")
        conf = s.get("confidence", 0.0)
        explanation = s.get("explanation", "")
        details = s.get("details") or {}
        color = DIRECTION_COLORS.get(direction, "#7f7f7f")
        emoji = DIRECTION_EMOJI.get(direction, "⚪")
        # Card
        with col:
            st.markdown(
                f'<div style="border:1px solid #ddd; border-radius:8px; '
                f'padding:12px; background:#fafafa; height:100%;">'
                f'<div style="display:flex; justify-content:space-between; '
                f'align-items:baseline;">'
                f'<b style="font-size:14px;">{name}</b>'
                f'<span style="color:{color}; font-weight:600;">'
                f'{emoji} {direction}</span></div>'
                f'<div style="margin-top:8px; height:6px; background:#eee; '
                f'border-radius:3px; overflow:hidden;">'
                f'<div style="width:{conf*100:.0f}%; height:100%; '
                f'background:{color};"></div></div>'
                f'<div style="font-size:11px; color:#666; margin-top:4px;">'
                f'conf {conf:.2f}</div>'
                f'<div style="font-size:11px; color:#333; margin-top:8px; '
                f'min-height:36px;">{explanation[:200]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if details:
                with st.expander("details", expanded=False):
                    st.json(details)


# ---------------- text summary (sent to Grok) ----------------
st.markdown("---")
with st.expander("Text summary (sent to Grok)", expanded=False):
    st.text_area("text_summary", value=result.get("text_summary", ""),
                 height=300, label_visibility="collapsed")


# ---------------- as a sortable table ----------------
st.markdown("---")
st.markdown("### Table view")
rows = []
for name, s in signals.items():
    rows.append({
        "strategy": name,
        "direction": s.get("direction"),
        "confidence": s.get("confidence"),
        "explanation": (s.get("explanation") or "")[:200],
    })
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)
