"""
PRISM — Chart Agent (All Visualisations + Self-Review Loop)
Receives the clean CSV, ProfileReport, and StatResults.
Selects the most informative charts, generates them via tools,
then collects chart specs directly from tool results (bypasses LLM truncation).
"""

from __future__ import annotations

import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from graph.state import AgentState, ChartSpec
from graph.utils import extract_text
from tools.chart_tools import (
    plot_bar,
    plot_box,
    plot_correlation_heatmap,
    plot_histogram,
    plot_line_trend,
    plot_missing_heatmap,
    plot_pairplot,
    plot_scatter,
    plot_violin,
)

# ── LLM ─────────────────────────────────────────────────────────────────────

_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ── Tool → chart_type mapping ────────────────────────────────────────────────

_CHART_TOOLS = [
    plot_histogram, plot_correlation_heatmap, plot_scatter,
    plot_bar, plot_box, plot_missing_heatmap, plot_line_trend,
    plot_pairplot, plot_violin,
]

_TOOL_TO_CHART_TYPE: dict[str, str] = {
    "plot_histogram":           "histogram",
    "plot_correlation_heatmap": "heatmap",
    "plot_scatter":             "scatter",
    "plot_bar":                 "bar",
    "plot_box":                 "box",
    "plot_missing_heatmap":     "missing_heatmap",
    "plot_line_trend":          "line",
    "plot_pairplot":            "pairplot",
    "plot_violin":              "violin",
}

_CHART_TOOL_NAMES = set(_TOOL_TO_CHART_TYPE.keys())

# ── Prompt ───────────────────────────────────────────────────────────────────

CHART_AGENT_PROMPT = """
You are the PRISM Chart Agent — an expert data visualisation designer.

You receive:
1. The path to the CLEANED CSV.
2. A ProfileReport (column types, skewness, outlier flags, shape).
3. StatResults (significant findings from the Stat Agent).
4. TimeSeriesResults (time-trend data if date columns were found).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHART SELECTION RULES (generate all that apply)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. ALWAYS generate: plot_correlation_heatmap (if ≥ 2 numeric columns)
2. For EACH numeric column with |skewness| > 0.5: plot_histogram
3. For EACH numeric column: plot_box (no group_col)
4. For categorical vs numeric pairs where ANOVA/ttest was significant:
   → plot_bar(cat_col, num_col, agg="mean")
   → plot_violin(num_col, group_col=cat_col)
5. For pairs where Pearson |r| > 0.3:
   → plot_scatter(col_a, col_b)
6. If any time-series results exist:
   → plot_line_trend(trend_json, title) for each TimeSeriesResult
7. If the ProfileReport shows any nulls:
   → plot_missing_heatmap
8. If ≥ 4 numeric columns exist:
   → plot_pairplot(df_path, cols_json) using top-5 numeric cols

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELF-REVIEW (after each chart)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Is this the right chart type for the data?
• Is x-axis cardinality too high (> 20 categories for a bar chart)?
• Does it show the intended finding?
If any check fails: regenerate with adjusted parameters (max 2 retries).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call the tools. The system will collect their outputs automatically.
You do NOT need to reproduce the chart JSON in your final reply.
Just call all appropriate tools, then respond with "Charts generated."
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_title(tool_name: str, args: dict) -> str:
    """Reconstruct a human-readable title from tool call arguments."""
    if "title" in args:
        return args["title"]
    m = {
        "plot_histogram":           lambda a: f"Distribution of {a.get('col', '')}",
        "plot_correlation_heatmap": lambda a: "Correlation Heatmap",
        "plot_scatter":             lambda a: f"{a.get('x_col', '')} vs {a.get('y_col', '')}",
        "plot_bar":                 lambda a: f"{a.get('agg','mean').title()} {a.get('val_col','')} by {a.get('cat_col','')}",
        "plot_box":                 lambda a: f"Box Plot: {a.get('num_col', '')}",
        "plot_missing_heatmap":     lambda a: "Missing Value Map",
        "plot_line_trend":          lambda a: a.get("title", "Time Trend"),
        "plot_pairplot":            lambda a: "Pair Plot — Key Numeric Columns",
        "plot_violin":              lambda a: f"Violin: {a.get('num_col', '')}",
    }
    fn = m.get(tool_name)
    return fn(args) if fn else tool_name.replace("plot_", "").replace("_", " ").title()


def _extract_charts_from_messages(messages: list) -> list[ChartSpec]:
    """
    Scan the agent message history for ToolMessages produced by chart tools.
    This bypasses the LLM output entirely — no truncation, no JSON corruption.
    """
    # Build mapping: tool_call_id → {name, args}
    id_to_call: dict[str, dict] = {}
    for msg in messages:
        for tc in getattr(msg, "tool_calls", []) or []:
            id_to_call[tc["id"]] = tc

    charts: list[ChartSpec] = []
    for msg in messages:
        tc_id = getattr(msg, "tool_call_id", None)
        if not tc_id or tc_id not in id_to_call:
            continue
        call = id_to_call[tc_id]
        tool_name = call.get("name", "")
        if tool_name not in _CHART_TOOL_NAMES:
            continue
        content = extract_text(getattr(msg, "content", ""))
        if not content or not content.strip().startswith("{"):
            continue
        # Validate the JSON before storing
        try:
            json.loads(content)
        except json.JSONDecodeError:
            continue
        charts.append(ChartSpec(
            chart_type=_TOOL_TO_CHART_TYPE.get(tool_name, "custom"),
            title=_infer_title(tool_name, call.get("args", {})),
            description="",
            plotly_json=content,
        ))

    return charts


def _build_context(state: AgentState) -> str:
    pr = state.get("profile_report")
    stat_results = state.get("stat_results", [])
    ts_results = state.get("time_series_results", [])

    ctx: dict = {
        "profile": {
            "numeric_cols": pr.numeric_cols if pr else [],
            "categorical_cols": pr.categorical_cols if pr else [],
            "date_cols": pr.date_cols if pr else [],
            "shape": list(pr.shape) if pr else [0, 0],
            "skewness": pr.skewness if pr else {},
            "outlier_flags": pr.outlier_flags if pr else {},
            "null_any": any(v > 0 for v in (pr.null_counts.values() if pr else [])),
        },
        "significant_stats": [
            {
                "test_name": s.test_name,
                "col_a": s.col_a,
                "col_b": s.col_b,
                "r_or_stat": s.statistic,
                "p_value": s.p_value,
            }
            for s in stat_results
            if s.significant
        ],
        "time_series": [
            {
                "date_col": t.date_col,
                "value_col": t.value_col,
                "freq": t.freq,
                "trend_direction": t.trend_direction,
                "trend_values_json": json.dumps(t.trend_values),
            }
            for t in ts_results
        ],
    }
    return json.dumps(ctx, indent=2)


def chart_agent_node(state: AgentState) -> dict:
    """LangGraph node for the Chart Agent."""
    import os as _os
    pr = state.get("profile_report")
    clean_path = pr.clean_csv_path if pr else state["dataframe_path"]

    # Fallback: if clean CSV is missing, use original
    if not _os.path.exists(clean_path):
        clean_path = state["dataframe_path"]

    context = _build_context(state)

    agent = create_react_agent(_llm, _CHART_TOOLS, prompt=CHART_AGENT_PROMPT)
    result = agent.invoke({
        "messages": [
            ("user",
             f"Generate all charts for this cleaned CSV: {clean_path}\n\n"
             f"Dataset context:\n{context}")
        ]
    })

    messages = result.get("messages", [])

    # ── Primary: extract charts from tool messages (reliable, no LLM truncation)
    chart_specs = _extract_charts_from_messages(messages)

    updates: dict = {}
    if chart_specs:
        updates["chart_specs"] = chart_specs
    else:
        updates["errors"] = ["chart_agent: no valid charts extracted from tool results"]
        updates["chart_specs"] = []

    updates["next_agent"] = "supervisor"
    return updates
