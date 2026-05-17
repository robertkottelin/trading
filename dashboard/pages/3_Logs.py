"""Logs — live tail of pipeline log files with level and regex filtering."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import state  # noqa: E402
from dashboard.lib.log_tail import tail_lines, filter_lines, colorize_level  # noqa: E402

st.set_page_config(page_title="Logs", page_icon="📜", layout="wide")
st.title("📜 Logs")

# ---------------- controls ----------------
all_logs = state.list_log_files()
live = state.LOG_DIR / "pipeline_live.log"
latest = state.latest_log_path()

with st.sidebar:
    st.markdown("### Source")
    options = []
    labels = []
    if live.exists():
        options.append(str(live))
        labels.append(f"pipeline_live.log ({live.stat().st_size//1024} KB)")
    for p in all_logs:
        if p == live:
            continue
        options.append(str(p))
        labels.append(f"{p.name} ({p.stat().st_size//1024} KB)")
    default_idx = 0 if options else None
    selected_idx = st.selectbox("Log file", options=list(range(len(options))),
                                format_func=lambda i: labels[i],
                                index=default_idx) if options else None
    selected = Path(options[selected_idx]) if options and selected_idx is not None else None

    st.markdown("### Filters")
    n_lines = st.number_input("Tail N lines", min_value=50, max_value=10_000,
                              value=500, step=50)
    level = st.selectbox("Min level", ["", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        index=0)
    regex = st.text_input("Regex / substring", value="")
    colorize = st.checkbox("Colorize levels", value=True)
    auto = st.checkbox("Auto-refresh", value=True)
    interval_s = st.slider("Interval (s)", min_value=2, max_value=60, value=5)
    if auto:
        st_autorefresh(interval=interval_s * 1000, key="logs_autorefresh")


# ---------------- render ----------------
if not selected:
    st.info(f"No log files found in {state.LOG_DIR}.")
    st.stop()

st.caption(f"Reading: `{selected}`")
lines = tail_lines(selected, n=int(n_lines))
filtered = filter_lines(lines, level=level or None, regex=regex or None)

st.caption(f"{len(filtered)} of {len(lines)} lines match filters.")

if colorize:
    rendered = "<br>".join(colorize_level(ln) for ln in filtered)
    st.markdown(
        f'<div style="font-family:monospace; white-space:pre-wrap; '
        f'font-size:12px; background:#0e1117; color:#fafafa; padding:12px; '
        f'border-radius:6px; max-height:75vh; overflow-y:auto;">{rendered}</div>',
        unsafe_allow_html=True,
    )
else:
    st.code("\n".join(filtered), language="log")
