"""
PRISM — Stat Agent (Statistics + Time Series)
Receives the clean CSV and ProfileReport, runs appropriate statistical tests,
and produces a list of StatResult and TimeSeriesResult objects.
"""

from __future__ import annotations

import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from graph.state import AgentState, StatResult, TimeSeriesResult
from graph.utils import last_ai_text, parse_json_from_text
from tools.stat_tools import (
    detect_date_columns,
    run_anova,
    run_chi_square,
    run_correlation_matrix,
    run_pearson,
    run_segment_compare,
    run_ttest,
    run_time_trend,
)

# ── LLM ─────────────────────────────────────────────────────────────────────

_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ── Prompt ───────────────────────────────────────────────────────────────────

STAT_AGENT_PROMPT = """
You are the PRISM Stat Agent — an expert statistician.

You receive the path to a CLEANED CSV and a structured profile of the dataset.
Your job is to extract every statistically meaningful finding from the data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — CORRELATION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Call run_correlation_matrix(df_path) to find the top correlated pairs.
• For each pair with |r| > 0.3:
  → Call run_pearson(df_path, col_a, col_b) for the exact test + p-value.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — CATEGORICAL × NUMERIC TESTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each pairing of a CATEGORICAL column (from the profile) with a NUMERIC column:
• If the categorical column has exactly 2 unique values:
  → Call run_ttest(df_path, num_col, group_col)
• If the categorical column has 3+ unique values AND cardinality ≤ 10:
  → Call run_anova(df_path, num_col, group_col)
  → Also call run_segment_compare(df_path, segment_col, value_col)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — CATEGORICAL × CATEGORICAL TESTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each pair of categorical columns where BOTH have cardinality ≤ 15:
• Call run_chi_square(df_path, col_a, col_b)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4 — TIME SERIES (if date columns exist)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Call detect_date_columns(df_path) to confirm date columns.
• For each date column found, pair it with EACH numeric column:
  → Call run_time_trend(df_path, date_col, value_col, freq="ME")
  → If the dataset spans > 5 years, also try freq="QE"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After completing all tests, respond with a JSON block:

```json
{{
  "stat_results": [
    {{
      "test_name": "pearson",
      "col_a": "column_name",
      "col_b": "other_column",
      "statistic": 0.0,
      "p_value": 0.0,
      "significant": true,
      "interpretation": "Plain English explanation of what this finding means for the data.",
      "effect_size": null
    }}
  ],
  "time_series_results": [
    {{
      "date_col": "date",
      "value_col": "revenue",
      "freq": "ME",
      "trend_direction": "up",
      "peak_period": "2023-12-01",
      "growth_rates": {{}},
      "trend_values": {{}}
    }}
  ]
}}
```

CRITICAL rules for the "interpretation" field:
- Write it as a business analyst would explain it to a non-technical stakeholder.
- Lead with what it means, not what the test did.
- Example: "Sales are significantly higher in Q4 than Q1 (p=0.001), suggesting a strong seasonal effect."
- If NOT significant: "No meaningful relationship was found between X and Y."
- Maximum 2 sentences.

Only include tests that you actually ran and have real numbers for. Do not invent values.
"""

_STAT_TOOLS = [
    run_correlation_matrix, run_pearson, run_chi_square,
    run_ttest, run_anova, run_segment_compare,
    detect_date_columns, run_time_trend,
]



def _build_profile_summary(state: AgentState) -> str:
    """Summarise ProfileReport for the agent's context."""
    pr = state.get("profile_report")
    if pr is None:
        return "No profile available."
    return json.dumps({
        "numeric_cols": pr.numeric_cols,
        "categorical_cols": pr.categorical_cols,
        "date_cols": pr.date_cols,
        "shape": list(pr.shape),
    }, indent=2)


def stat_agent_node(state: AgentState) -> dict:
    """LangGraph node for the Stat Agent."""
    import os
    pr = state.get("profile_report")
    clean_path = pr.clean_csv_path if pr else state["dataframe_path"]
    # Fallback: if clean CSV is missing (profile agent skipped writing it), use original
    if not os.path.exists(clean_path):
        clean_path = state["dataframe_path"]
    profile_summary = _build_profile_summary(state)

    agent = create_react_agent(_llm, _STAT_TOOLS, prompt=STAT_AGENT_PROMPT)
    result = agent.invoke({
        "messages": [
            ("user",
             f"Run all statistical tests on this cleaned CSV: {clean_path}\n\n"
             f"Dataset profile:\n{profile_summary}")
        ]
    })

    final_text = last_ai_text(result.get("messages", []))
    parsed = parse_json_from_text(final_text)
    updates: dict = {}

    if parsed:
        raw_stats = parsed.get("stat_results", [])
        updates["stat_results"] = [
            StatResult(
                test_name=s.get("test_name", "unknown"),
                col_a=s.get("col_a", ""),
                col_b=s.get("col_b"),
                statistic=float(s.get("statistic", 0)),
                p_value=float(s.get("p_value", 1)),
                significant=bool(s.get("significant", False)),
                interpretation=s.get("interpretation", ""),
                effect_size=float(s["effect_size"]) if s.get("effect_size") is not None else None,
            )
            for s in raw_stats
        ]

        raw_ts = parsed.get("time_series_results", [])
        updates["time_series_results"] = [
            TimeSeriesResult(
                date_col=t.get("date_col", ""),
                value_col=t.get("value_col", ""),
                freq=t.get("freq", "ME"),
                trend_direction=t.get("trend_direction", "flat"),
                peak_period=t.get("peak_period", ""),
                growth_rates=t.get("growth_rates", {}),
                trend_values=t.get("trend_values", {}),
            )
            for t in raw_ts
        ]
    else:
        updates["errors"] = [f"stat_agent: failed to parse output — {final_text[:300]}"]
        updates["stat_results"] = []
        updates["time_series_results"] = []

    updates["next_agent"] = "supervisor"
    return updates
