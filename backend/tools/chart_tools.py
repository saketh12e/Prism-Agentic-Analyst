"""
PRISM — Chart Agent Tools
All chart tools return a serialised Plotly JSON string using the PRISM dark theme.
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_core.tools import tool

# ── PRISM Dark Theme ─────────────────────────────────────────────────────────

_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    title_font=dict(size=16, color="#f1f5f9"),
    margin=dict(t=60, b=50, l=50, r=30),
)
_ACCENT = "#6366f1"   # indigo-500


# ── Individual chart tools ───────────────────────────────────────────────────

@tool
def plot_histogram(df_path: str, col: str, bins: int = 40) -> str:
    """
    Plot the distribution of a numeric column as a histogram.
    Returns Plotly JSON string.
    """
    df = pd.read_csv(df_path)
    fig = px.histogram(
        df, x=col, nbins=bins,
        title=f"Distribution of {col}",
        color_discrete_sequence=[_ACCENT],
    )
    fig.update_layout(**_DARK)
    return fig.to_json()


@tool
def plot_correlation_heatmap(df_path: str) -> str:
    """
    Plot a correlation heatmap for all numeric columns.
    Returns Plotly JSON string.
    """
    df = pd.read_csv(df_path)
    corr = df.select_dtypes("number").corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
    )
    fig.update_layout(**_DARK)
    return fig.to_json()


@tool
def plot_scatter(df_path: str, x_col: str, y_col: str, color_col: str = None) -> str:
    """
    Scatter plot with OLS trendline between two numeric columns.
    Optionally colour points by a categorical column.
    Returns Plotly JSON string.
    """
    df = pd.read_csv(df_path)
    # trendline="ols" requires statsmodels; fall back gracefully if not installed
    try:
        import statsmodels  # noqa: F401
        trendline = "ols"
    except ImportError:
        trendline = None
    fig = px.scatter(
        df, x=x_col, y=y_col,
        color=color_col if color_col else None,
        title=f"{x_col} vs {y_col}",
        opacity=0.6,
        trendline=trendline,
    )
    fig.update_layout(**_DARK)
    return fig.to_json()


@tool
def plot_bar(df_path: str, cat_col: str, val_col: str, agg: str = "mean") -> str:
    """
    Grouped bar chart: aggregate val_col by cat_col.
    agg options: 'mean' | 'sum' | 'count' | 'median'
    Returns Plotly JSON string.
    """
    df = pd.read_csv(df_path)
    grouped = (
        df.groupby(cat_col)[val_col]
        .agg(agg)
        .reset_index()
        .sort_values(val_col, ascending=False)
    )
    fig = px.bar(
        grouped, x=cat_col, y=val_col,
        title=f"{agg.title()} {val_col} by {cat_col}",
        color=val_col,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(**_DARK)
    return fig.to_json()


@tool
def plot_box(df_path: str, num_col: str, group_col: str = None) -> str:
    """
    Box plot showing distribution and outliers for a numeric column.
    Optionally group by a categorical column.
    Returns Plotly JSON string.
    """
    df = pd.read_csv(df_path)
    title = f"Box Plot: {num_col}" + (f" by {group_col}" if group_col else "")
    fig = px.box(
        df, x=group_col if group_col else None, y=num_col,
        title=title,
        color=group_col if group_col else None,
        points="outliers",
    )
    fig.update_layout(**_DARK)
    return fig.to_json()


@tool
def plot_missing_heatmap(df_path: str) -> str:
    """
    Heatmap of missing values — rows as records, columns as features.
    Red = missing, dark = present.
    Returns Plotly JSON string.
    """
    df = pd.read_csv(df_path)
    null_df = df.isnull().astype(int)
    fig = px.imshow(
        null_df.T,
        aspect="auto",
        title="Missing Value Map",
        color_continuous_scale=["#1e293b", "#f43f5e"],
        labels=dict(color="Missing"),
    )
    fig.update_layout(**_DARK)
    return fig.to_json()


@tool
def plot_line_trend(trend_json: str, title: str) -> str:
    """
    Line chart from run_time_trend output.
    trend_json: JSON string of {period_str: value} dict (the 'trend' field from run_time_trend).
    Returns Plotly JSON string.
    """
    trend = json.loads(trend_json)
    periods = list(trend.keys())
    values = list(trend.values())
    fig = px.line(x=periods, y=values, title=title, markers=True)
    fig.update_traces(line_color=_ACCENT, line_width=2.5)
    fig.update_layout(**_DARK)
    return fig.to_json()


@tool
def plot_pairplot(df_path: str, cols_json: str) -> str:
    """
    Scatter matrix (pair plot) for a set of numeric columns.
    cols_json: JSON array of column names e.g. '["col1","col2","col3"]'
    Returns Plotly JSON string.
    """
    df = pd.read_csv(df_path)
    cols = json.loads(cols_json)[:5]  # cap at 5 to keep it readable
    fig = px.scatter_matrix(
        df[cols],
        title="Pair Plot — Key Numeric Columns",
        opacity=0.5,
    )
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(**_DARK)
    return fig.to_json()


@tool
def plot_violin(df_path: str, num_col: str, group_col: str = None) -> str:
    """
    Violin plot showing distribution shape for a numeric column.
    Optionally group by a categorical column.
    Returns Plotly JSON string.
    """
    df = pd.read_csv(df_path)
    title = f"Violin: {num_col}" + (f" by {group_col}" if group_col else "")
    fig = px.violin(
        df,
        x=group_col if group_col else None,
        y=num_col,
        box=True,
        points="outliers",
        title=title,
        color=group_col if group_col else None,
    )
    fig.update_layout(**_DARK)
    return fig.to_json()
