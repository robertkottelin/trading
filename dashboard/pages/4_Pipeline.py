"""Pipeline — DAG of the 7 reasoning agent stages + market data refresh."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import state  # noqa: E402
from dashboard.lib.pipeline_parser import parse_latest_cycle  # noqa: E402

st.set_page_config(page_title="Pipeline", page_icon="🔁", layout="wide")
st.title("🔁 Pipeline progression")

with st.sidebar:
    st.markdown("### Refresh")
    auto = st.checkbox("Auto-refresh", value=True)
    if auto:
        st_autorefresh(interval=3000, key="pipeline_autorefresh")


hb = state.heartbeat()
log_path = state.latest_log_path()
if not log_path:
    st.warning("No pipeline log file found. Start the bot to generate one.")
    st.stop()

st.caption(f"Reading: `{log_path}`")
parsed = parse_latest_cycle(log_path)
stages = parsed["stages"]


# ---------------- countdown ----------------
def _now_utc():
    return datetime.now(timezone.utc)


col1, col2, col3, col4 = st.columns(4)
with col1:
    started = parsed["cycle_started"]
    st.metric("Cycle started", started.strftime("%H:%M:%S UTC") if started else "—")
with col2:
    finished = parsed["cycle_finished"]
    st.metric("Cycle finished", finished.strftime("%H:%M:%S UTC") if finished else "in progress")
with col3:
    if started:
        elapsed = ((finished or _now_utc()) - started).total_seconds()
        st.metric("Cycle elapsed", f"{elapsed:.1f}s")
    else:
        st.metric("Cycle elapsed", "—")
with col4:
    if hb and hb.get("next_run_at"):
        try:
            next_at = datetime.fromisoformat(hb["next_run_at"])
            if next_at.tzinfo is None:
                next_at = next_at.replace(tzinfo=timezone.utc)
            remaining = (next_at - _now_utc()).total_seconds()
            if remaining > 0:
                st.metric("Next run in", f"{int(remaining)}s")
            else:
                st.metric("Next run", "due now")
        except Exception:
            st.metric("Next run", hb.get("next_run_at"))
    else:
        st.metric("Next run", "—")

if parsed["paused"]:
    st.warning("⏸️ PAUSE FLAG active for this cycle — only Stage 0 ran.")

# ---------------- DAG ----------------
st.markdown("---")
st.markdown("### Stage progression")


def _build_dag(stages):
    colors = {"done": "#2ca02c", "running": "#1f77b4", "pending": "#cccccc",
              "failed": "#d62728", "paused": "#e6a700"}
    fig = go.Figure()
    n = len(stages)
    for i, s in enumerate(stages):
        c = colors.get(s["status"], "#cccccc")
        hover = f"<b>{s['name']}</b><br>status: {s['status']}"
        if s.get("duration_s") is not None:
            hover += f"<br>duration: {s['duration_s']:.1f}s"
        if s.get("started_at"):
            hover += f"<br>started: {s['started_at'].strftime('%H:%M:%S UTC')}"
        # Node
        fig.add_trace(go.Scatter(
            x=[i], y=[0], mode="markers+text",
            marker=dict(size=58, color=c, line=dict(color="black", width=1.5),
                        symbol="circle"),
            text=[str(i + 1)], textposition="middle center",
            textfont=dict(size=18, color="white"),
            hovertext=hover, hoverinfo="text", showlegend=False,
        ))
        # Label below
        fig.add_annotation(x=i, y=-0.6, text=s["name"], showarrow=False,
                            font=dict(size=11), align="center", yanchor="top")
        # Duration label (small) above
        if s.get("duration_s") is not None:
            fig.add_annotation(x=i, y=0.5,
                               text=f"{s['duration_s']:.1f}s",
                               showarrow=False, font=dict(size=10, color="#555"))
        # Arrow to next
        if i < n - 1:
            fig.add_annotation(x=i + 0.5, y=0, ax=i + 0.05, ay=0,
                                xref="x", yref="y", axref="x", ayref="y",
                                showarrow=True, arrowhead=3, arrowsize=1.6,
                                arrowwidth=2, arrowcolor="#888")
    fig.update_layout(
        height=260, showlegend=False,
        xaxis=dict(visible=False, range=[-0.5, n - 0.5]),
        yaxis=dict(visible=False, range=[-1.5, 1]),
        margin=dict(l=10, r=10, t=10, b=80),
        plot_bgcolor="#ffffff",
    )
    return fig


st.plotly_chart(_build_dag(stages), use_container_width=True)


# ---------------- stage table ----------------
import pandas as pd  # noqa: E402

rows = []
for s in stages:
    rows.append({
        "Stage": s["name"],
        "Status": s["status"],
        "Started": s["started_at"].strftime("%H:%M:%S UTC") if s.get("started_at") else "—",
        "Duration (s)": f"{s['duration_s']:.1f}" if s.get("duration_s") is not None else "—",
    })
df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------- raw tail ----------------
st.markdown("---")
st.markdown("### Cycle log (last 60 lines)")
tail = parsed.get("raw_tail", [])[-60:]
st.code("\n".join(tail) or "(empty)", language="log")
