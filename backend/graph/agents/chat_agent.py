"""
PRISM — Chat Agent (On-Demand Analyst + Export Handler)
Always available. Handles two jobs:
1. Answer any user question about the data using execute_pandas or query_profile.
2. Trigger all export operations (clean CSV, charts ZIP, PDF report).
"""

from __future__ import annotations

import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from graph.state import AgentState
from graph.utils import extract_text, last_ai_text, parse_json_from_text
from tools.chat_tools import auto_chart_from_query, execute_pandas, query_profile
from tools.export_tools import export_charts_zip, export_clean_csv, generate_pdf_report

# ── LLM ─────────────────────────────────────────────────────────────────────

_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
    temperature=0.2,   # slight creativity for natural chat responses
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ── Prompt ───────────────────────────────────────────────────────────────────

CHAT_AGENT_PROMPT = """
You are the PRISM Chat Agent — an on-demand data analyst and export specialist.

You have access to:
- The CLEANED CSV (passed as df_path in tool calls)
- A JSON representation of the ProfileReport
- All statistical test results
- All generated chart specs

You have two modes:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE 1 — ANSWERING QUESTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the user asks a question about the data:

1. First, decide if query_profile can answer it directly (structural questions like
   "how many columns?" or "what are the numeric columns?").
   → If yes: call query_profile(profile_json, question) and return the answer.

2. If the question requires actual data computation:
   → Write pandas code that assigns the result to `output`
   → Call execute_pandas(df_path, code)
   → Interpret the result in plain English for the user.

3. If the answer would be better shown as a chart:
   → Write pandas code for the aggregation
   → Call auto_chart_from_query(df_path, code, title, chart_hint)
   → Include the chart in your response.

RESPONSE STYLE:
- Lead with the direct answer (1–2 sentences).
- Follow with supporting detail or a chart if relevant.
- Use plain English. No jargon.
- If the user asks about something not in the data, say so clearly.
- End with a follow-up question or suggestion when helpful.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE 2 — EXPORTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the user requests an export OR when the API calls you with export_{type}:

• "export csv" or "download clean data":
  → Call export_clean_csv(clean_csv_path, session_id)

• "export charts" or "download charts":
  → Call export_charts_zip(chart_specs_json, session_id)

• "export pdf" or "download report":
  → Call generate_pdf_report(profile_json, stats_json, narrative, session_id)

Always confirm what was exported and provide the filename.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Respond with a JSON block:

```json
{{
  "response": "Your plain-English answer here.",
  "new_chart": null,
  "export_path": null
}}
```

- response: always present, your natural-language reply
- new_chart: if you generated a chart, include the ChartSpec object here (or null)
- export_path: if you exported a file, include {"path": ..., "filename": ..., "type": ...} (or null)

Only fill the fields relevant to what you actually did. The other fields must be null.
"""

_CHAT_TOOLS = [
    execute_pandas,
    auto_chart_from_query,
    query_profile,
    export_clean_csv,
    export_charts_zip,
    generate_pdf_report,
]




def _build_context(state: AgentState) -> str:
    pr = state.get("profile_report")
    return json.dumps({
        "df_path": pr.clean_csv_path if pr else state.get("dataframe_path", ""),
        "profile_available": pr is not None,
        "numeric_cols": pr.numeric_cols if pr else [],
        "categorical_cols": pr.categorical_cols if pr else [],
        "stat_results_count": len(state.get("stat_results", [])),
        "chart_specs_count": len(state.get("chart_specs", [])),
        "session_id": state.get("session_id", "unknown"),
    }, indent=2)


def chat_agent_node(state: AgentState) -> dict:
    """LangGraph node for the Chat Agent."""
    pr = state.get("profile_report")
    clean_path = pr.clean_csv_path if pr else state.get("dataframe_path", "")
    user_query = state.get("user_query", "")
    context = _build_context(state)

    # Build profile JSON for tools that need it
    profile_json = "{}"
    if pr:
        try:
            profile_json = json.dumps({
                "shape": list(pr.shape),
                "dtypes": pr.dtypes,
                "null_counts": pr.null_counts,
                "null_pcts": pr.null_pcts,
                "duplicate_count": pr.duplicate_count,
                "duplicate_pct": pr.duplicate_pct,
                "numeric_cols": pr.numeric_cols,
                "categorical_cols": pr.categorical_cols,
                "date_cols": pr.date_cols,
                "skewness": pr.skewness,
                "outlier_flags": pr.outlier_flags,
            })
        except Exception:
            pass

    stats_json = json.dumps([
        {
            "test_name": s.test_name,
            "col_a": s.col_a,
            "col_b": s.col_b,
            "statistic": s.statistic,
            "p_value": s.p_value,
            "significant": s.significant,
            "interpretation": s.interpretation,
            "effect_size": s.effect_size,
        }
        for s in state.get("stat_results", [])
    ])

    agent = create_react_agent(_llm, _CHAT_TOOLS, prompt=CHAT_AGENT_PROMPT)
    result = agent.invoke({
        "messages": [
            ("user",
             f"User question / request: {user_query}\n\n"
             f"Cleaned CSV path: {clean_path}\n"
             f"Profile JSON: {profile_json}\n"
             f"Stats JSON (first 2000 chars): {stats_json[:2000]}\n"
             f"Context: {context}")
        ]
    })

    messages = result.get("messages", [])
    final_text = last_ai_text(messages)
    parsed = parse_json_from_text(final_text)
    updates: dict = {}

    # ── Extract response text ─────────────────────────────────────────────────
    if parsed:
        response_text = parsed.get("response", final_text)
    else:
        response_text = final_text
    updates["chat_history"] = [{"role": "assistant", "content": response_text}]

    # ── Extract chart from tool messages (avoids LLM truncation of plotly_json)
    id_to_call: dict = {}
    for msg in messages:
        for tc in getattr(msg, "tool_calls", []) or []:
            id_to_call[tc["id"]] = tc

    chart_tool_names = {"auto_chart_from_query"}
    for msg in messages:
        tc_id = getattr(msg, "tool_call_id", None)
        if not tc_id or tc_id not in id_to_call:
            continue
        call = id_to_call[tc_id]
        if call.get("name") not in chart_tool_names:
            continue
        content = extract_text(getattr(msg, "content", ""))
        if not content or not content.strip().startswith("{"):
            continue
        try:
            json.loads(content)
        except json.JSONDecodeError:
            continue
        from graph.state import ChartSpec
        args = call.get("args", {})
        updates["chart_specs"] = [ChartSpec(
            chart_type="bar",
            title=args.get("title", "Query Chart"),
            description="",
            plotly_json=content,
        )]
        break  # only one chart per chat turn

    # ── Extract export path from LLM output ──────────────────────────────────
    if parsed:
        export_path = parsed.get("export_path")
        if export_path:
            updates["export_paths"] = [export_path]

    updates["next_agent"] = "supervisor"
    return updates
