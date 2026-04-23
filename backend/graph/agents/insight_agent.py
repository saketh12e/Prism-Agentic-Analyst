"""
PRISM — Insight Agent
Phase 4 (autonomous): Reviews all prior EDA output and independently generates
3–5 data hypotheses, tests each one by executing pandas code, evaluates the
evidence, and produces structured DataInsight objects.

This agent is the most agentic component in PRISM:
  • It creates its OWN questions (emergent hypothesis generation)
  • It executes code to gather evidence (tool-driven ReAct loop)
  • It self-evaluates findings and assigns a verdict + confidence score
"""

from __future__ import annotations

import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from graph.state import AgentState, DataInsight
from graph.utils import last_ai_text, parse_json_from_text
from tools.insight_tools import test_hypothesis

# ── LLM ──────────────────────────────────────────────────────────────────────

_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
    temperature=0.4,    # slight creativity for novel hypothesis generation
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ── Prompt ────────────────────────────────────────────────────────────────────

INSIGHT_AGENT_PROMPT = """
You are the PRISM Insight Agent — an autonomous data scientist.

You have already seen the full exploratory analysis of this dataset:
profile statistics, statistical test results, and generated charts.

Your job is to go further. You independently generate testable hypotheses
about the data and then PROVE OR DISPROVE each one using pandas code.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1 — GENERATE HYPOTHESES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Based on the context provided, generate exactly 3–5 SPECIFIC, TESTABLE
hypotheses. A good hypothesis:
  • Names concrete columns ("Revenue is higher when Region = 'West'")
  • Is falsifiable with pandas code
  • Is genuinely interesting — not just restating what the stats already showed
  • Covers different aspects: distributions, segments, trends, anomalies, ratios

DO NOT test things already covered by standard statistical tests.
Aim for actionable business insights.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2 — TEST EACH HYPOTHESIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each hypothesis, call:
  test_hypothesis(df_path=<path>, hypothesis=<text>, code=<pandas code>)

Rules for your pandas code:
  • `df` is pre-loaded — do NOT call pd.read_csv()
  • Assign the key result to a variable named `output`
  • Keep code concise — one logical test per call
  • Handle edge cases (empty groups, division by zero) with try/except

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3 — EVALUATE AND REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After testing ALL hypotheses, respond with a JSON block:

```json
{
  "insights": [
    {
      "hypothesis": "Revenue is significantly higher in Q4 than other quarters",
      "test_code": "output = df.assign(quarter=pd.to_datetime(df['date']).dt.quarter).groupby('quarter')['revenue'].mean()",
      "finding": "Q4 average revenue ($142k) is 2.3x higher than Q1–Q3 average ($61k).",
      "verdict": "confirmed",
      "confidence": 0.92,
      "supporting_stat": "Q4 mean: 142000, overall mean: 87000"
    }
  ]
}
```

verdict rules:
  • "confirmed"    — the data clearly supports the hypothesis (strong, consistent signal)
  • "refuted"      — the data clearly contradicts the hypothesis
  • "inconclusive" — the signal is weak, ambiguous, or data is insufficient

confidence: your honest 0.0–1.0 estimate based on sample size and signal strength.
finding: one or two plain-English sentences. Lead with the key number.
supporting_stat: the single most important number from your test result (or null).

Only include hypotheses you actually tested. Do not invent results.
"""

_INSIGHT_TOOLS = [test_hypothesis]


def _build_context(state: AgentState) -> str:
    pr = state.get("profile_report")
    stat_results = state.get("stat_results", [])

    return json.dumps({
        "df_path": pr.clean_csv_path if pr else state.get("dataframe_path", ""),
        "profile": {
            "shape": list(pr.shape) if pr else [0, 0],
            "numeric_cols": pr.numeric_cols if pr else [],
            "categorical_cols": pr.categorical_cols if pr else [],
            "date_cols": pr.date_cols if pr else [],
            "skewness": pr.skewness if pr else {},
            "outlier_flags": pr.outlier_flags if pr else {},
            "describe_stats": pr.describe_stats if pr else {},
        },
        "significant_findings": [
            {
                "test": s.test_name,
                "col_a": s.col_a,
                "col_b": s.col_b,
                "interpretation": s.interpretation,
            }
            for s in stat_results if s.significant
        ],
        "all_stat_interpretations": [
            s.interpretation for s in stat_results[:20]
        ],
    }, indent=2)


def insight_agent_node(state: AgentState) -> dict:
    """LangGraph node for the Insight Agent."""
    pr = state.get("profile_report")
    clean_path = pr.clean_csv_path if pr else state.get("dataframe_path", "")
    if not os.path.exists(clean_path):
        clean_path = state.get("dataframe_path", "")

    context = _build_context(state)

    agent = create_react_agent(_llm, _INSIGHT_TOOLS, prompt=INSIGHT_AGENT_PROMPT)
    result = agent.invoke({
        "messages": [
            ("user",
             f"Generate and test hypotheses for this dataset.\n\n"
             f"Cleaned CSV path: {clean_path}\n\n"
             f"Full analysis context:\n{context}")
        ]
    })

    final_text = last_ai_text(result.get("messages", []))
    parsed = parse_json_from_text(final_text)
    updates: dict = {}

    if parsed:
        raw = parsed.get("insights", [])
        updates["insights"] = [
            DataInsight(
                hypothesis=item.get("hypothesis", ""),
                test_code=item.get("test_code", ""),
                finding=item.get("finding", ""),
                verdict=item.get("verdict", "inconclusive"),
                confidence=float(item.get("confidence", 0.5)),
                supporting_stat=item.get("supporting_stat"),
            )
            for item in raw
        ]
    else:
        updates["errors"] = [
            f"insight_agent: failed to parse output — {final_text[:300]}"
        ]
        updates["insights"] = []

    updates["next_agent"] = "supervisor"
    return updates
