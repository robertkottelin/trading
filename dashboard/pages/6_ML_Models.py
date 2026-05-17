"""ML Models — production config metadata, weights, firing history."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import ml_inspector, plots, state  # noqa: E402

st.set_page_config(page_title="ML Models", page_icon="🧠", layout="wide")
st.title("🧠 ML Models")


@st.cache_data(ttl=120)
def _list_configs():
    return [str(p) for p in ml_inspector.find_production_configs()]


@st.cache_data(ttl=120)
def _load_summary(path_str: str):
    cfg = ml_inspector.load_config(Path(path_str))
    return ml_inspector.summarize_config(cfg, Path(path_str))


configs = _list_configs()
if not configs:
    st.warning(f"No production_config_*.json files found under {state.MODELS_DIR}.")
    st.stop()


with st.sidebar:
    st.markdown("### Production config")
    labels = [str(Path(c).relative_to(state.REPO_ROOT)) for c in configs]
    cfg_idx = st.selectbox("File", options=list(range(len(configs))),
                           format_func=lambda i: labels[i])

selected_path = configs[cfg_idx]
summary = _load_summary(selected_path)

# ---------------- metadata ----------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Version", summary.get("version") or "—")
c2.metric("Generated", (summary.get("generated") or "—")[:19])
c3.metric("Weighted models", summary.get("num_weighted_models") or summary.get("num_listed_models"))
metrics = summary.get("metrics", {})
c4.metric("Sharpe",
          f"{metrics['sharpe']:.2f}" if isinstance(metrics.get("sharpe"), (int, float)) else "—")
c5.metric("Max DD",
          f"{metrics['max_dd']:.2%}" if isinstance(metrics.get("max_dd"), (int, float)) else "—")

with st.expander("Full config snapshot", expanded=False):
    st.json({k: v for k, v in summary.items() if k not in ("weights", "models_list")})

# ---------------- weights bar chart ----------------
st.markdown("---")
st.markdown("### Model weights")
weights = summary.get("weights") or {}
if not weights and summary.get("models_list"):
    # Bearish format — synthesize equal weights for visualization
    weights = {m.get("name", f"model_{i}"): m.get("prob_threshold") or 1.0
               for i, m in enumerate(summary["models_list"])}
if weights:
    wdf = ml_inspector.weights_dataframe(weights)
    with st.expander("Filter", expanded=False):
        directions = sorted(wdf["direction"].unique())
        picks = st.multiselect("Direction", directions, default=directions)
        top_n = st.slider("Show top N (by weight)", min_value=10,
                           max_value=max(20, len(wdf)),
                           value=min(50, len(wdf)))
    fdf = wdf[wdf["direction"].isin(picks)].head(top_n)
    fig = plots.model_weights_bar(dict(zip(fdf["name"], fdf["weight"])))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(wdf, use_container_width=True, hide_index=True)
else:
    st.info("No model weights present in this config.")

# ---------------- firing history ----------------
st.markdown("---")
st.markdown("### Firing history (per cycle)")
ddf = state.decisions_df(tail=500)
if ddf.empty:
    st.info("No decisions logged yet.")
else:
    fdf = ml_inspector.firing_history_df(ddf)
    if "bullish_count" in fdf.columns:
        st.plotly_chart(plots.signal_consensus_area(fdf), use_container_width=True)
    if "weighted_score" in fdf.columns:
        st.plotly_chart(plots.weighted_score_line(fdf), use_container_width=True)
    st.dataframe(fdf.tail(50)[::-1], use_container_width=True, hide_index=True)

# ---------------- model pickles ----------------
st.markdown("---")
st.markdown("### Model artifact files")
model_dir = Path(selected_path).parent
pkls = ml_inspector.model_pkl_files(model_dir)
st.caption(f"Directory: `{model_dir.relative_to(state.REPO_ROOT)}` — {len(pkls)} .pkl files")
if pkls:
    rows = [{"file": p.name, "size_kb": p.stat().st_size // 1024,
             "modified": pd.Timestamp(p.stat().st_mtime, unit="s")} for p in pkls]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------- live inference (optional) ----------------
st.markdown("---")
st.markdown("### Live inference")
st.caption("Runs `llm_agent.signal_generator.generate_signals()` against current `market_context_data/`.")
if st.button("Run live inference", type="primary"):
    try:
        from llm_agent import signal_generator
        with st.spinner("Loading models and generating signals..."):
            result = signal_generator.generate_signals()
        cons = result.get("consensus") or {}
        a, b, c, d = st.columns(4)
        a.metric("Bullish", cons.get("bullish_count", "—"))
        b.metric("Bearish", cons.get("bearish_count", "—"))
        c.metric("Neutral", cons.get("neutral_count", "—"))
        d.metric("Weighted score",
                 f"{cons.get('weighted_score', 0):+.3f}"
                 if isinstance(cons.get("weighted_score"), (int, float)) else "—")
        st.text_area("Text summary (sent to Grok)",
                     value=result.get("text_summary", ""), height=240)
        with st.expander("Raw output", expanded=False):
            # Try to JSON-serialize; fall back to repr
            try:
                import json
                st.code(json.dumps(result, indent=2, default=str)[:8000], language="json")
            except Exception:
                st.text(repr(result)[:4000])
    except Exception as e:
        st.error(f"Live inference failed: {e}")
