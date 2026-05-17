"""Trading Dashboard — main entry point.

Run with:
    bash dashboard/run_dashboard.sh
or
    streamlit run dashboard/app.py --server.port 8501 --server.address 127.0.0.1
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Ensure repo root is on sys.path so `from strategies.engine import ...` works
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import state  # noqa: E402

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _format_age(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    return f"{seconds / 3600:.1f}h ago"


def _sidebar_status():
    """Render bot status snapshot in the sidebar."""
    st.sidebar.markdown("### Bot status")
    hb = state.heartbeat()
    pid_info = state.pid_info()
    paused = state.is_paused()

    if hb:
        status = hb.get("status", "unknown")
        run_no = hb.get("run_number", "?")
        age = state.heartbeat_age_seconds()
        emoji = {"running": "🟢", "completed": "🟢", "failed": "🔴"}.get(status, "⚪")
        st.sidebar.markdown(f"{emoji} **{status}** · run #{run_no}")
        st.sidebar.caption(f"Last heartbeat: {_format_age(age)}")
    else:
        st.sidebar.markdown("⚪ **no heartbeat yet**")
        st.sidebar.caption("Bot has not written a heartbeat file.")

    if pid_info:
        st.sidebar.caption(f"PID: `{pid_info.get('pid')}` · "
                           f"mode: `{pid_info.get('mode', '?')}`")

    if paused:
        st.sidebar.warning("⏸️ Pause flag active — new entries blocked")


def main():
    st.title("📈 Trading Dashboard")
    st.caption(
        "Local control plane for the dYdX trading bot — pipeline state, "
        "ML/strategy signals, positions, configs, backtests, retraining."
    )

    _sidebar_status()
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pages** (use the navigator above)")
    st.sidebar.markdown(
        "1. Overview\n"
        "2. Bot Control\n"
        "3. Logs\n"
        "4. Pipeline\n"
        "5. Data Explorer\n"
        "6. ML Models\n"
        "7. Strategies\n"
        "8. Positions\n"
        "9. Backtests\n"
        "10. Retrain\n"
        "11. Config"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Repo: `{state.REPO_ROOT}`")
    st.sidebar.caption(f"Now: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    # Landing-page snapshot
    col1, col2, col3, col4 = st.columns(4)
    hb = state.heartbeat()
    pf = state.latest_portfolio()
    dec = state.latest_decision()
    rs = state.retrain_state()

    with col1:
        st.metric("Bot status", (hb or {}).get("status", "—"),
                  delta=f"run #{(hb or {}).get('run_number', '?')}" if hb else None)
    with col2:
        equity = (pf or {}).get("equity")
        st.metric("Equity (USD)",
                  f"${equity:,.2f}" if isinstance(equity, (int, float)) else "—",
                  delta=f"margin {(pf or {}).get('margin_pct', '—')}%" if pf else None)
    with col3:
        direction = (dec or {}).get("direction", "—")
        conf = (dec or {}).get("confidence")
        st.metric("Latest decision", direction,
                  delta=f"conf {conf:.2f}" if isinstance(conf, (int, float)) else None)
    with col4:
        last_dep = (rs or {}).get("last_deployed", "—")
        st.metric("Last ML deploy",
                  last_dep[:10] if isinstance(last_dep, str) else "—",
                  delta=(rs or {}).get("status"))

    st.markdown("---")
    st.markdown(
        "**Quick start**\n\n"
        "Navigate to **Bot Control** to start/stop the bot. "
        "**Overview** shows live state; **Pipeline** visualises stage progress; "
        "**ML Models** and **Strategies** show what is currently firing; "
        "**Positions** tracks PnL; **Backtests** and **Retrain** trigger jobs; "
        "**Config** edits YAML with safe backup."
    )


if __name__ == "__main__":
    main()
