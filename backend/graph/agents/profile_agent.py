"""
PRISM — Profile Agent (Inspect + Clean)
Phase 1: Runs all inspection tools to understand the raw CSV.
Phase 2: Applies targeted cleaning based on Phase 1 findings.
Produces a ProfileReport and CleaningSummary in shared state.
"""

from __future__ import annotations

import json
import os
import shutil

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from graph.state import AgentState, CleaningSummary, ProfileReport
from graph.utils import extract_text, last_ai_text, parse_json_from_text
from tools.clean_tools import (
    cap_outliers,
    fix_dtype,
    fix_nulls,
    remove_duplicates,
    strip_whitespace,
)
from tools.inspect_tools import (
    detect_dtype_issues,
    get_describe,
    get_duplicates,
    get_dtypes,
    get_null_report,
    get_outliers,
    get_shape,
    get_skewness,
    get_value_counts,
)

# ── LLM ─────────────────────────────────────────────────────────────────────

_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ── Prompt ───────────────────────────────────────────────────────────────────

PROFILE_AGENT_PROMPT = """
You are the PRISM Profile Agent — an expert data inspector and cleaner.

You work in two phases. Complete EVERY step. Do NOT skip any tool call.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1 — INSPECT (run ALL tools below)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. get_shape(df_path)              → how many rows and columns?
2. get_dtypes(df_path)             → what type is each column?
3. get_null_report(df_path)        → null counts and percentages
4. get_duplicates(df_path)         → are there duplicate rows?
5. get_describe(df_path)           → summary statistics for numeric cols
6. get_skewness(df_path)           → skewness for numeric cols
7. detect_dtype_issues(df_path)    → object columns that are actually datetime/numeric?
8. For EACH numeric column found in step 2:
   → call get_outliers(df_path, col) to check for outliers
9. For EACH categorical column found in step 2 (up to 5 cols):
   → call get_value_counts(df_path, col) to check cardinality

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2 — CLEAN (apply fixes based on Phase 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use output_path = {output_path} for ALL cleaning operations.
After each operation, the next operation reads from output_path.

Apply these rules:
• Null values in NUMERIC columns:
  - If null_pct > 50%: strategy = "drop"  (column is too sparse)
  - If skewness |skew| > 1: strategy = "median" (skewed → median is robust)
  - Otherwise: strategy = "mean"
• Null values in CATEGORICAL columns:
  - strategy = "mode"
• Duplicate rows found:
  - call remove_duplicates(df_path=output_path, output_path=output_path)
• Dtype issues detected (likely_datetime / likely_numeric):
  - call fix_dtype for each flagged column
• Outlier pct > 2% in a numeric column:
  - call cap_outliers(df_path=output_path, col=col, output_path=output_path)
• Object columns with obvious whitespace issues:
  - call strip_whitespace

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After all cleaning is complete, respond with a JSON block in this exact format:

```json
{{
  "profile_report": {{
    "shape": [rows, cols],
    "dtypes": {{}},
    "null_counts": {{}},
    "null_pcts": {{}},
    "duplicate_count": 0,
    "duplicate_pct": 0.0,
    "numeric_cols": [],
    "categorical_cols": [],
    "date_cols": [],
    "describe_stats": {{}},
    "outlier_flags": {{}},
    "skewness": {{}},
    "dtype_issues": {{}},
    "clean_csv_path": "{output_path}"
  }},
  "cleaning_summary": {{
    "nulls_fixed": {{}},
    "duplicates_removed": 0,
    "dtypes_fixed": {{}},
    "columns_created": [],
    "rows_before": 0,
    "rows_after": 0
  }}
}}
```

Fill every field with real values from your tool calls. Do not leave placeholders.
"""

# ── Tool list ─────────────────────────────────────────────────────────────────

_PROFILE_TOOLS = [
    get_shape, get_dtypes, get_null_report, get_duplicates,
    get_describe, get_outliers, get_skewness, detect_dtype_issues, get_value_counts,
    fix_nulls, remove_duplicates, fix_dtype, cap_outliers, strip_whitespace,
]

# ── Agent ─────────────────────────────────────────────────────────────────────

def _build_agent(output_path: str):
    prompt = PROFILE_AGENT_PROMPT.format(output_path=output_path)
    return create_react_agent(_llm, _PROFILE_TOOLS, prompt=prompt)




def profile_agent_node(state: AgentState) -> dict:
    """
    LangGraph node for the Profile Agent.
    Runs inspection + cleaning; updates state with ProfileReport and CleaningSummary.
    """
    df_path = state["dataframe_path"]
    output_path = df_path.replace(".csv", "_clean.csv")
    # If output_path == df_path (no .csv ext), add suffix
    if output_path == df_path:
        output_path = df_path + "_clean.csv"

    # Pre-copy the original CSV to output_path so downstream agents always
    # have a valid file even if the LLM skips cleaning (e.g. data is already clean)
    if not os.path.exists(output_path) or output_path == df_path:
        shutil.copy2(df_path, output_path)

    agent = _build_agent(output_path)
    result = agent.invoke({
        "messages": [
            ("user",
             f"Inspect and clean this CSV: {df_path}\n"
             f"Save ALL cleaned versions to: {output_path}")
        ]
    })

    final_text = last_ai_text(result.get("messages", []))
    parsed = parse_json_from_text(final_text)
    updates: dict = {}

    if parsed:
        pr = parsed.get("profile_report", {})
        cs = parsed.get("cleaning_summary", {})

        updates["profile_report"] = ProfileReport(
            shape=tuple(pr.get("shape", [0, 0])),
            dtypes=pr.get("dtypes", {}),
            null_counts=pr.get("null_counts", {}),
            null_pcts=pr.get("null_pcts", {}),
            duplicate_count=pr.get("duplicate_count", 0),
            duplicate_pct=pr.get("duplicate_pct", 0.0),
            numeric_cols=pr.get("numeric_cols", []),
            categorical_cols=pr.get("categorical_cols", []),
            date_cols=pr.get("date_cols", []),
            describe_stats=pr.get("describe_stats", {}),
            outlier_flags=pr.get("outlier_flags", {}),
            skewness=pr.get("skewness", {}),
            dtype_issues=pr.get("dtype_issues", {}),
            clean_csv_path=pr.get("clean_csv_path", output_path),
        )
        updates["cleaning_summary"] = CleaningSummary(
            nulls_fixed=cs.get("nulls_fixed", {}),
            duplicates_removed=cs.get("duplicates_removed", 0),
            dtypes_fixed=cs.get("dtypes_fixed", {}),
            columns_created=cs.get("columns_created", []),
            rows_before=cs.get("rows_before", 0),
            rows_after=cs.get("rows_after", 0),
        )
    else:
        # Parsing failed — store the error and pass along
        updates["errors"] = [f"profile_agent: failed to parse output — {final_text[:300]}"]
        # Create minimal profile so pipeline can continue
        updates["profile_report"] = ProfileReport(
            shape=(0, 0), dtypes={}, null_counts={}, null_pcts={},
            duplicate_count=0, duplicate_pct=0.0,
            numeric_cols=[], categorical_cols=[], date_cols=[],
            describe_stats={}, outlier_flags={}, skewness={}, dtype_issues={},
            clean_csv_path=output_path,
        )

    updates["next_agent"] = "supervisor"
    return updates
