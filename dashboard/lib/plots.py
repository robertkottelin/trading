"""Reusable Plotly figures for the dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def equity_with_drawdown(df: pd.DataFrame, equity_col: str = "equity",
                         ts_col: str = "timestamp") -> go.Figure:
    """Two-panel chart: equity curve on top, drawdown % below."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05,
                        subplot_titles=("Equity", "Drawdown %"))
    if df.empty or equity_col not in df.columns:
        fig.update_layout(title="No portfolio data yet", height=500)
        return fig

    work = df[[ts_col, equity_col]].copy().dropna()
    if work.empty:
        fig.update_layout(title="No portfolio data yet", height=500)
        return fig

    work["peak"] = work[equity_col].cummax()
    work["dd_pct"] = (work[equity_col] / work["peak"] - 1.0) * 100

    fig.add_trace(go.Scatter(x=work[ts_col], y=work[equity_col],
                             mode="lines", name="Equity",
                             line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=work[ts_col], y=work["peak"],
                             mode="lines", name="Peak",
                             line=dict(color="rgba(31,119,180,0.3)", dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=work[ts_col], y=work["dd_pct"],
                             mode="lines", name="Drawdown",
                             fill="tozeroy",
                             line=dict(color="#d62728", width=1.5)), row=2, col=1)
    fig.update_layout(height=500, hovermode="x unified",
                      margin=dict(l=40, r=20, t=50, b=40),
                      showlegend=False)
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    return fig


def model_weights_bar(weights: dict[str, float], top_n: int | None = None) -> go.Figure:
    """Horizontal bar chart of ML model weights, sorted descending."""
    items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    if top_n:
        items = items[:top_n]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#2ca02c" if "up" in n or "fav" in n else "#d62728" if "down" in n or "bear" in n
              else "#7f7f7f" for n in names]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors))
    fig.update_layout(height=max(400, 18 * len(names)),
                      margin=dict(l=180, r=20, t=30, b=40),
                      xaxis_title="Weight", yaxis=dict(autorange="reversed"))
    return fig


def signal_consensus_area(df: pd.DataFrame) -> go.Figure:
    """Stacked-area chart of bullish_count / bearish_count / neutral_count over time."""
    fig = go.Figure()
    if df.empty or "timestamp" not in df.columns:
        fig.update_layout(title="No decision history yet", height=400)
        return fig
    if "bullish_count" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bullish_count"], name="Bullish",
                                 stackgroup="one", line=dict(color="#2ca02c")))
    if "bearish_count" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["bearish_count"], name="Bearish",
                                 stackgroup="one", line=dict(color="#d62728")))
    if "neutral_count" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["neutral_count"], name="Neutral",
                                 stackgroup="one", line=dict(color="#7f7f7f")))
    fig.update_layout(height=350, margin=dict(l=40, r=20, t=30, b=40),
                      yaxis_title="Model count", hovermode="x unified")
    return fig


def weighted_score_line(df: pd.DataFrame) -> go.Figure:
    """Line plot of weighted_score and avg_raw_score over time, with zero line."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title="No decision history yet", height=300)
        return fig
    if "weighted_score" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["weighted_score"],
                                 name="Weighted score", line=dict(color="#1f77b4")))
    if "avg_raw_score" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["avg_raw_score"],
                                 name="Avg raw score", line=dict(color="#ff7f0e", dash="dot")))
    fig.add_hline(y=0, line_color="gray", line_dash="dash")
    fig.update_layout(height=280, margin=dict(l=40, r=20, t=30, b=40),
                      hovermode="x unified", yaxis_title="Score")
    return fig


def ohlcv_candles(df: pd.DataFrame, ts_col: str = "timestamp") -> go.Figure:
    """Auto-detect open/high/low/close columns and render a candlestick chart."""
    cols_lower = {c.lower(): c for c in df.columns}
    o = cols_lower.get("open")
    h = cols_lower.get("high")
    l = cols_lower.get("low")
    c = cols_lower.get("close")
    if not all([o, h, l, c]):
        return None  # caller falls back to line chart
    if ts_col not in df.columns:
        # try common alternatives
        for alt in ("time", "datetime", "date", "ts"):
            if alt in df.columns:
                ts_col = alt
                break
    fig = go.Figure(go.Candlestick(x=df[ts_col], open=df[o], high=df[h],
                                    low=df[l], close=df[c]))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False,
                      margin=dict(l=40, r=20, t=30, b=40))
    return fig


def time_series_line(df: pd.DataFrame, ts_col: str, value_cols: list[str]) -> go.Figure:
    """Generic multi-series line chart."""
    fig = go.Figure()
    for col in value_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[ts_col], y=df[col], name=col, mode="lines"))
    fig.update_layout(height=450, hovermode="x unified",
                      margin=dict(l=40, r=20, t=30, b=40))
    return fig


def pipeline_dag(stage_status: list[dict[str, Any]]) -> go.Figure:
    """Render a horizontal pipeline DAG.

    Args:
        stage_status: list of {"name": str, "status": "done|running|pending|failed",
                                "duration_s": float|None}
    """
    colors = {"done": "#2ca02c", "running": "#1f77b4", "pending": "#cccccc",
              "failed": "#d62728"}
    n = len(stage_status)
    fig = go.Figure()
    for i, s in enumerate(stage_status):
        x = i
        color = colors.get(s.get("status"), "#cccccc")
        label = s["name"]
        dur = s.get("duration_s")
        text = f"<b>{label}</b><br>{s.get('status', 'pending')}"
        if dur is not None:
            text += f"<br>{dur:.1f}s"
        fig.add_trace(go.Scatter(x=[x], y=[0], mode="markers+text",
                                 marker=dict(size=60, color=color, line=dict(color="black", width=1)),
                                 text=[f"{label}"], textposition="bottom center",
                                 hovertext=text, hoverinfo="text",
                                 showlegend=False))
        if i < n - 1:
            fig.add_annotation(x=x + 0.5, y=0, ax=x, ay=0,
                               xref="x", yref="y", axref="x", ayref="y",
                               arrowhead=2, arrowsize=1.5, arrowwidth=2,
                               arrowcolor="#888")
    fig.update_layout(height=200, showlegend=False,
                      xaxis=dict(visible=False, range=[-0.5, n - 0.5]),
                      yaxis=dict(visible=False, range=[-1, 1]),
                      margin=dict(l=20, r=20, t=20, b=60))
    return fig
