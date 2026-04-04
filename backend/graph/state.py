"""
PRISM — Shared Agent State
All dataclasses and the AgentState TypedDict that flow through the LangGraph.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Optional

from typing_extensions import TypedDict


# ── Typed outputs produced by the Profile Agent ─────────────────────────────

@dataclass
class ProfileReport:
    shape: tuple                        # (rows, cols)
    dtypes: dict                        # col -> dtype string
    null_counts: dict                   # col -> int
    null_pcts: dict                     # col -> float (percentage)
    duplicate_count: int
    duplicate_pct: float
    numeric_cols: list
    categorical_cols: list
    date_cols: list
    describe_stats: dict                # pandas describe() output
    outlier_flags: dict                 # col -> {"count": int, "pct": float}
    skewness: dict                      # col -> skew value
    dtype_issues: dict                  # col -> "likely_datetime" | "likely_numeric"
    clean_csv_path: str                 # absolute path to the cleaned CSV


@dataclass
class CleaningSummary:
    nulls_fixed: dict                   # col -> {"strategy": str, "count_fixed": int}
    duplicates_removed: int
    dtypes_fixed: dict                  # col -> {"from": str, "to": str}
    columns_created: list               # any derived columns added
    rows_before: int
    rows_after: int


# ── Typed outputs produced by the Stat Agent ────────────────────────────────

@dataclass
class StatResult:
    test_name: str                      # "pearson" | "chi_square" | "ttest" | "anova"
    col_a: str
    col_b: Optional[str]
    statistic: float
    p_value: float
    significant: bool
    interpretation: str                 # plain-English LLM-written finding
    effect_size: Optional[float]


@dataclass
class TimeSeriesResult:
    date_col: str
    value_col: str
    freq: str                           # M / W / D / Q
    trend_direction: str                # "up" | "down" | "flat"
    peak_period: str
    growth_rates: dict
    trend_values: dict


# ── Typed outputs produced by the Chart Agent ───────────────────────────────

@dataclass
class ChartSpec:
    chart_type: str                     # histogram|heatmap|scatter|bar|box|line|pairplot|violin|missing_heatmap
    title: str
    description: str                    # one-line caption shown in UI
    plotly_json: str                    # serialised Plotly figure (fig.to_json())


# ── Shared AgentState — the contract all agents read/write ──────────────────

class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    dataframe_path: str                 # absolute path to the uploaded raw CSV
    user_query: str                     # current user instruction / question

    # ── Profile Agent outputs ──────────────────────────────────────────────
    profile_report: Optional[ProfileReport]
    cleaning_summary: Optional[CleaningSummary]

    # ── Stat Agent outputs (list reducers — append across calls) ───────────
    stat_results: Annotated[list[StatResult], operator.add]
    time_series_results: Annotated[list[TimeSeriesResult], operator.add]

    # ── Chart Agent outputs ────────────────────────────────────────────────
    chart_specs: Annotated[list[ChartSpec], operator.add]

    # ── Supervisor output ──────────────────────────────────────────────────
    narrative_summary: Optional[str]

    # ── Chat Agent (query + export) ────────────────────────────────────────
    chat_history: Annotated[list[dict], operator.add]   # {"role": str, "content": str}
    export_paths: Annotated[list[dict], operator.add]   # {"type": str, "path": str, "filename": str}

    # ── Control flow ───────────────────────────────────────────────────────
    next_agent: str
    analysis_complete: bool
    iteration_count: int
    errors: Annotated[list[str], operator.add]
