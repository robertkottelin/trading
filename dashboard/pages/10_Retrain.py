"""Retrain — view retrain_state and trigger background retraining."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit_autorefresh import st_autorefresh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import job_runner, state  # noqa: E402
from dashboard.lib.log_tail import tail_lines  # noqa: E402

st.set_page_config(page_title="Retrain", page_icon="🔧", layout="wide")
st.title("🔧 ML Retraining")

st_autorefresh(interval=5000, key="retrain_autorefresh")


# ---------------- state ----------------
rs = state.retrain_state() or {}
status = rs.get("status", "—")

c1, c2, c3, c4 = st.columns(4)
status_emoji = {"deployed": "🟢", "running": "🔵",
                "failed": "🔴", "—": "⚪"}.get(status, "⚪")
c1.metric(f"{status_emoji} Status", status)
c2.metric("Last deployed", (rs.get("last_deployed") or "—")[:19])
c3.metric("Last started", (rs.get("last_retrain_started") or "—")[:19])
c4.metric("Last PID", rs.get("pid") or "—")

if rs.get("failure_reason"):
    st.error(f"Last failure: {rs['failure_reason']}")

with st.expander("Full retrain_state.json", expanded=False):
    st.json(rs)

# ---------------- detect external runner ----------------
external = job_runner.find_other_running("retrain_manager") or \
           job_runner.find_other_running("train_v2_staging") or \
           job_runner.find_other_running("train_bearish_staging")
if external:
    st.info("External retrain process(es) detected:")
    for e in external:
        st.caption(f"PID {e['pid']} · started {e['started_at']}")
        st.code(e['cmd'], language="bash")


# ---------------- trigger ----------------
st.markdown("---")
st.markdown("### Trigger retraining")

if "retrain_handle" not in st.session_state:
    st.session_state.retrain_handle = None
handle = st.session_state.retrain_handle
running = bool(handle and handle.is_running())

t1, t2 = st.columns(2)
with t1:
    st.markdown("**Full chain (bullish + bearish + strategy tuning)**")
    if st.button("▶️ Run retrain chain (background)", type="primary",
                 disabled=running):
        log_path = state.LOG_DIR / f"dashboard_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        cmd = [sys.executable, "-u", "-c",
               "from retraining.retrain_manager import run_retrain_chain; "
               "run_retrain_chain(background=False)"]
        st.session_state.retrain_handle = job_runner.spawn_job(
            "retrain_chain", cmd=cmd, cwd=state.REPO_ROOT, log_path=log_path,
        )
        st.success(f"Spawned retrain chain, PID {st.session_state.retrain_handle.pid()}")
        st.rerun()

with t2:
    st.markdown("**Bearish only**")
    if st.button("▶️ Train bearish models", disabled=running):
        log_path = state.LOG_DIR / f"dashboard_retrain_bear_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        cmd = [sys.executable, "-u", "-m", "retraining.train_bearish_staging"]
        st.session_state.retrain_handle = job_runner.spawn_job(
            "retrain_bearish", cmd=cmd, cwd=state.REPO_ROOT, log_path=log_path,
        )
        st.success(f"Spawned bearish retrain, PID {st.session_state.retrain_handle.pid()}")
        st.rerun()


k1, k2 = st.columns(2)
with k1:
    if st.button("⏹️ Kill current job", disabled=not running):
        if handle:
            handle.kill()
            st.success("Kill signal sent.")
            st.rerun()
with k2:
    if st.button("🧹 Clear handle"):
        st.session_state.retrain_handle = None
        st.rerun()


# ---------------- active job ----------------
st.markdown("---")
if handle:
    st.markdown("### Active job")
    st.caption(f"`{ ' '.join(handle.cmd) }` · PID {handle.pid()} · {handle.status()}")
    st.caption(f"Log: `{handle.log_path}`")
    st.code("\n".join(handle.tail(n=300)) or "(no output yet)", language="log")
    if not handle.is_running():
        rc = handle.returncode
        (st.success if rc == 0 else st.error)(f"Job finished — exit code {rc}.")
else:
    st.caption("No retrain job currently running from the dashboard.")


# ---------------- recent retrain logs ----------------
st.markdown("---")
st.markdown("### Recent retrain logs")
retrain_logs = sorted(state.LOG_DIR.glob("retrain*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
retrain_logs += sorted(state.LOG_DIR.glob("dashboard_retrain*.log"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
if retrain_logs:
    options = [str(p.relative_to(state.REPO_ROOT)) for p in retrain_logs[:20]]
    picked = st.selectbox("Log file", options=options, index=0)
    full = state.REPO_ROOT / picked
    st.code("\n".join(tail_lines(full, n=300)) or "(empty)", language="log")
else:
    st.caption("No retrain logs found.")
