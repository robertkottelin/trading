"""Backtests — view stored results and trigger backtest_v2 runs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import data_loaders, job_runner, state  # noqa: E402

st.set_page_config(page_title="Backtests", page_icon="🧪", layout="wide")
st.title("🧪 Backtests")

# ---------------- trigger ----------------
st.markdown("### Trigger a backtest")

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    strategy_id = st.text_input("--strategy (blank = all)", value="")
with c2:
    sl = st.number_input("--sl (stop loss)", min_value=0.0,
                         max_value=0.5, value=0.05, step=0.01,
                         format="%.3f")
with c3:
    tp = st.number_input("--tp (take profit)", min_value=0.0,
                         max_value=2.0, value=0.10, step=0.01,
                         format="%.3f")
with c4:
    sizing = st.selectbox("--sizing",
                          ["confidence", "fixed", "equal", "vol_target"],
                          index=0)
extra = st.text_input("Extra args (passed verbatim)", value="")

if "bt_handle" not in st.session_state:
    st.session_state.bt_handle = None

handle = st.session_state.bt_handle
running = bool(handle and handle.is_running())

bcols = st.columns(3)
with bcols[0]:
    if st.button("▶️ Run backtest_v2", type="primary",
                 disabled=running):
        from datetime import datetime
        log_path = state.LOG_DIR / f"dashboard_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        cmd = [sys.executable, "-u", "-m", "strategies.backtest_v2",
               "--sl", str(sl), "--tp", str(tp), "--sizing", sizing]
        if strategy_id.strip():
            cmd.extend(["--strategy", strategy_id.strip()])
        if extra.strip():
            cmd.extend(extra.strip().split())
        st.session_state.bt_handle = job_runner.spawn_job(
            "backtest_v2", cmd=cmd, cwd=state.REPO_ROOT, log_path=log_path,
        )
        st.success(f"Spawned backtest, PID {st.session_state.bt_handle.pid()}")
        st.rerun()
with bcols[1]:
    if st.button("⏹️ Kill backtest", disabled=not running):
        if handle:
            handle.kill()
            st.success("Kill signal sent.")
            st.rerun()
with bcols[2]:
    if st.button("🧹 Clear handle"):
        st.session_state.bt_handle = None
        st.rerun()


# ---------------- running job status ----------------
st.markdown("---")
if handle:
    st_autorefresh(interval=2000, key="bt_autorefresh")
    st.markdown("### Active job")
    st.caption(f"`{ ' '.join(handle.cmd) }` · PID {handle.pid()} · {handle.status()}")
    st.caption(f"Log: `{handle.log_path}`")
    st.code("\n".join(handle.tail(n=250)) or "(no output yet)", language="log")
    if not handle.is_running():
        rc = handle.returncode
        if rc == 0:
            st.success(f"Backtest finished — exit code {rc}.")
        else:
            st.error(f"Backtest failed — exit code {rc}.")
else:
    st.caption("No backtest currently running.")


# ---------------- result browser ----------------
st.markdown("---")
st.markdown("### Stored results")

# Look for any backtest output files in common locations
result_candidates: list[Path] = []
for root in [state.REPO_ROOT, state.REPO_ROOT / "model_training",
             state.REPO_ROOT / "strategies"]:
    if not root.exists():
        continue
    for pattern in ("backtest*.csv", "*_results.csv", "equity_*.csv",
                    "backtest*.parquet"):
        result_candidates.extend(root.glob(pattern))

# Also surface findings markdown
findings = list((state.REPO_ROOT / "model_training").glob("findings*.md"))

if not result_candidates and not findings:
    st.info("No stored backtest result files found. "
            "After a run finishes, output CSVs land in the repo root or "
            "model_training/.")
else:
    if result_candidates:
        st.markdown("**Result files**")
        rows = [{"file": str(p.relative_to(state.REPO_ROOT)),
                 "size_kb": p.stat().st_size // 1024,
                 "modified": pd.Timestamp(p.stat().st_mtime, unit="s")}
                for p in sorted(set(result_candidates), key=lambda p: p.stat().st_mtime,
                                reverse=True)]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        picked = st.selectbox("Open file",
                              options=[r["file"] for r in rows], index=0)
        try:
            df = data_loaders.load_dataframe(state.REPO_ROOT / picked,
                                              head_only=True, nrows=5000)
            st.dataframe(df.head(500), use_container_width=True)
            # If looks like an equity curve, render it
            ts_col = data_loaders.detect_timestamp_col(df)
            num = data_loaders.numeric_cols(df)
            if ts_col and "equity" in num:
                from dashboard.lib import plots
                df2 = data_loaders.coerce_timestamp(df, ts_col)
                df2 = df2.rename(columns={ts_col: "timestamp"})
                st.plotly_chart(plots.equity_with_drawdown(df2),
                                use_container_width=True)
        except Exception as e:
            st.error(f"Failed to preview: {e}")

    if findings:
        st.markdown("**Findings**")
        for f in findings:
            with st.expander(str(f.relative_to(state.REPO_ROOT))):
                st.markdown(f.read_text()[:20_000])
