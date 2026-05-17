"""Data Explorer — browse CSV / Parquet / JSONL artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import data_loaders, plots, state  # noqa: E402

st.set_page_config(page_title="Data Explorer", page_icon="📈", layout="wide")
st.title("📈 Data Explorer")


@st.cache_data(ttl=30, show_spinner=False)
def _load(path_str: str, head_only: bool, nrows: int | None) -> pd.DataFrame:
    return data_loaders.load_dataframe(Path(path_str), head_only=head_only, nrows=nrows)


groups = data_loaders.list_data_files()
if not groups:
    st.warning("No data files found in market_context_data/, processed_data/, or raw_data/.")
    st.stop()

with st.sidebar:
    st.markdown("### File")
    group_label = st.selectbox("Source", list(groups.keys()))
    files = groups[group_label]
    labels = [f"{f.name}  ({data_loaders.file_size_kb(f)} KB)" for f in files]
    file_idx = st.selectbox("File", options=list(range(len(files))),
                            format_func=lambda i: labels[i])
    selected = files[file_idx]

    st.markdown("### Load")
    head_only = st.checkbox("Limit to first N rows (faster)", value=True)
    nrows = st.number_input("Row cap", min_value=100, max_value=200_000,
                            value=2000, step=100) if head_only else None


st.caption(f"`{selected}` · {data_loaders.file_size_kb(selected)} KB")

try:
    df = _load(str(selected), head_only=head_only, nrows=int(nrows) if nrows else None)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

if df.empty:
    st.info("File is empty.")
    st.stop()

# ---------------- metrics ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns)}")
mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
c3.metric("Memory", f"{mem_mb:.1f} MB")

# ---------------- head + dtypes ----------------
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Head", "Dtypes", "Summary stats"])
with tab1:
    st.dataframe(df.head(200), use_container_width=True)
with tab2:
    dt = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values,
                       "non_null": df.notna().sum().values,
                       "null": df.isna().sum().values})
    st.dataframe(dt, use_container_width=True, hide_index=True)
with tab3:
    num = data_loaders.numeric_cols(df)
    if num:
        st.dataframe(df[num].describe().T, use_container_width=True)
    else:
        st.caption("No numeric columns.")

# ---------------- plot ----------------
st.markdown("---")
st.markdown("### Plot")

ts_default = data_loaders.detect_timestamp_col(df)
all_cols = list(df.columns)
default_ts_idx = all_cols.index(ts_default) if ts_default in all_cols else 0
ts_col = st.selectbox("Timestamp column", all_cols, index=default_ts_idx)

if ts_col:
    work = data_loaders.coerce_timestamp(df, ts_col)
else:
    work = df

num_cols = [c for c in data_loaders.numeric_cols(work) if c != ts_col]
default_pick = [c for c in ("close", "Close", "price", "value", "open_interest", "funding_rate")
                if c in num_cols][:1] or num_cols[:1]
picked = st.multiselect("Value columns", num_cols, default=default_pick)

if not picked:
    st.info("Pick at least one numeric column to plot.")
else:
    # If picks look like OHLC, offer candlestick
    has_ohlc = all(c in num_cols for c in ("open", "high", "low", "close")) or \
               all(c in num_cols for c in ("Open", "High", "Low", "Close"))
    chart_kind = st.radio("Chart", ["Line"] + (["Candlestick (OHLC)"] if has_ohlc else []),
                          horizontal=True)
    if chart_kind == "Candlestick (OHLC)":
        fig = plots.ohlcv_candles(work.head(1500), ts_col=ts_col)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("OHLC columns not detected; falling back to line.")
            st.plotly_chart(plots.time_series_line(work.head(1500), ts_col, picked),
                            use_container_width=True)
    else:
        st.plotly_chart(plots.time_series_line(work.head(2000), ts_col, picked),
                        use_container_width=True)
