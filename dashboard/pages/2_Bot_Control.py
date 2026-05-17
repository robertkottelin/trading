"""Bot Control — start / stop / pause / resume / restart the trading bot."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import bot_supervisor, state  # noqa: E402
from dashboard.lib.log_tail import tail_lines  # noqa: E402

st.set_page_config(page_title="Bot Control", page_icon="🤖", layout="wide")
st.title("🤖 Bot Control")

st_autorefresh(interval=3000, key="bot_control_autorefresh")


# ---------------- status panel ----------------
status = bot_supervisor.status()
state_label = status["state"]


def _format_uptime(s):
    if s is None:
        return "—"
    if s < 60:
        return f"{int(s)}s"
    if s < 3600:
        return f"{int(s/60)}m {int(s%60)}s"
    return f"{s/3600:.1f}h"


cols = st.columns([1, 1, 1, 1])
status_emoji = {"running": "🟢", "stopped": "⚪"}.get(state_label, "❓")
cols[0].metric(f"{status_emoji} State", state_label.upper())
cols[1].metric("PID", status.get("pid") or "—")
cols[2].metric("Uptime", _format_uptime(status.get("uptime_s")))
cols[3].metric("Mode / Network",
               f"{status.get('mode') or '—'} · {status.get('network') or '—'}")

if status["paused"]:
    st.warning("⏸️ PAUSE FLAG active — bot will skip new entries on next cycle. "
               "Position management (Stage 0) still runs.")

if status.get("command"):
    with st.expander("Full command", expanded=False):
        st.code(" ".join(status["command"]), language="bash")

if status.get("heartbeat"):
    hb = status["heartbeat"]
    st.caption(f"Last heartbeat: `{hb.get('timestamp')}` · "
               f"run #{hb.get('run_number')} · status `{hb.get('status')}` · "
               f"next run `{hb.get('next_run_at', '—')}`")


st.markdown("---")

# ---------------- start controls ----------------
st.markdown("### Start")
running = state_label == "running"
with st.form("start_form", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        mode = st.radio("Execution mode", ["paper", "live"],
                        index=0, horizontal=True,
                        help="paper = no on-chain orders, live = real orders")
        network = st.radio("Network", ["testnet", "mainnet"],
                           index=0, horizontal=True,
                           help="CAUTION: mainnet uses real funds")
        interval = st.number_input("Loop interval (s)", min_value=30,
                                   max_value=3600, value=300, step=30)
    with c2:
        skip_signals = st.checkbox("--skip-signals (no ML inference)", value=False)
        skip_web_search = st.checkbox("--skip-web-search (no Grok web/X tools)", value=False)
        no_execute = st.checkbox("--no-execute (run pipeline, don't trade)", value=False)
        verbose = st.checkbox("--verbose (DEBUG logs)", value=False)
        full = st.checkbox("--full (run ML retrain on startup)", value=False)

    if mode == "live" and network == "mainnet":
        st.error("⚠️ LIVE × MAINNET selected — this will execute real trades with real funds.")

    submitted = st.form_submit_button("▶️ Start bot", disabled=running, type="primary")
    if submitted:
        try:
            payload = bot_supervisor.start_bot(
                mode=mode, network=network, interval=int(interval),
                skip_signals=skip_signals, skip_web_search=skip_web_search,
                no_execute=no_execute, verbose=verbose, full=full,
            )
            st.success(f"Bot started — PID {payload['pid']}, log: {payload['log_path']}")
            st.rerun()
        except RuntimeError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Failed to start: {e}")

st.markdown("---")

# ---------------- stop / pause / restart ----------------
st.markdown("### Stop / Pause / Restart")
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("⏹️ Stop", disabled=not running, use_container_width=True):
        result = bot_supervisor.stop_bot(grace_seconds=10.0)
        if result["killed"]:
            st.warning("Bot did not exit within grace period — SIGKILL sent.")
        elif result["signalled"]:
            st.success(f"Bot stopped (PID {result['pid']}).")
        else:
            st.info("Nothing to stop.")
        st.rerun()
with c2:
    if not status["paused"]:
        if st.button("⏸️ Pause", use_container_width=True):
            bot_supervisor.pause()
            st.success("Pause flag set. Next cycle will skip Stages 1-7.")
            st.rerun()
    else:
        if st.button("▶️ Resume", use_container_width=True, type="primary"):
            bot_supervisor.resume()
            st.success("Pause flag cleared. Next cycle resumes full pipeline.")
            st.rerun()
with c3:
    if st.button("🔁 Restart (paper · testnet · 300s)",
                 disabled=not running, use_container_width=True):
        try:
            payload = bot_supervisor.restart_bot(mode="paper", network="testnet",
                                                 interval=300)
            st.success(f"Bot restarted — PID {payload['pid']}.")
            st.rerun()
        except Exception as e:
            st.error(f"Restart failed: {e}")
with c4:
    if st.button("🧹 Clear stale PID file", use_container_width=True):
        bot_supervisor._clear_pid_file()
        st.success("PID file cleared.")
        st.rerun()


# ---------------- live log tail ----------------
st.markdown("---")
st.markdown("### Latest bot log (last 100 lines)")
log_path = None
if status.get("log_path") and Path(status["log_path"]).exists():
    log_path = Path(status["log_path"])
else:
    log_path = state.latest_log_path()

if log_path and log_path.exists():
    st.caption(f"`{log_path}`")
    lines = tail_lines(log_path, n=100)
    st.code("\n".join(lines) or "(empty)", language="log")
else:
    st.info("No log file available yet.")
