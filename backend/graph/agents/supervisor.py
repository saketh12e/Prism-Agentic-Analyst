"""
PRISM — Supervisor Agent
Deterministic pipeline orchestrator: routes based on state facts, not LLM guesswork.
Only calls the LLM once — to write the final narrative after all EDA is done.
"""

from __future__ import annotations

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from graph.state import AgentState
from graph.utils import extract_text

# ── LLM (used only for narrative generation) ─────────────────────────────────

_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
    temperature=0.3,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# ── Narrative prompt ──────────────────────────────────────────────────────────

_NARRATIVE_SYSTEM = """
You are a senior data analyst at PRISM.
Write a clear 2–3 paragraph narrative summary of this dataset analysis.
Rules:
- Lead with the most interesting finding.
- Mention key relationships, distributions, and data quality.
- Write for a non-technical business audience.
- Do NOT mention p-values, test names, or statistical jargon.
- Do NOT use markdown headers or bullet points — write flowing prose only.
- Return ONLY the narrative text, nothing else.
"""


def _generate_narrative(state: AgentState) -> str:
    """Call LLM once to write the final narrative. Returns a plain-text string."""
    pr = state.get("profile_report")
    stat_results = state.get("stat_results", [])
    chart_specs = state.get("chart_specs", [])

    context = json.dumps({
        "shape": list(pr.shape) if pr else [0, 0],
        "numeric_cols": pr.numeric_cols if pr else [],
        "categorical_cols": pr.categorical_cols if pr else [],
        "significant_findings": [
            {"cols": f"{s.col_a} × {s.col_b}", "interpretation": s.interpretation}
            for s in stat_results if s.significant
        ],
        "all_interpretations": [s.interpretation for s in stat_results[:15]],
        "charts_generated": [c.title for c in chart_specs],
        "outlier_flags": pr.outlier_flags if pr else {},
        "duplicate_count": pr.duplicate_count if pr else 0,
        "cleaning_notes": {
            "nulls": pr.null_counts if pr else {},
            "skewness": pr.skewness if pr else {},
        },
    }, indent=2)

    messages = [
        SystemMessage(content=_NARRATIVE_SYSTEM),
        HumanMessage(content=f"Analysis results:\n{context}"),
    ]

    try:
        response = _llm.invoke(messages)
        return extract_text(response.content).strip()
    except Exception:
        # Fallback: construct a simple narrative from facts
        shape = list(pr.shape) if pr else [0, 0]
        sig_count = sum(1 for s in stat_results if s.significant)
        return (
            f"The dataset contains {shape[0]:,} rows and {shape[1]} columns. "
            f"{sig_count} statistically significant relationships were identified. "
            f"Review the Statistics and Charts tabs for detailed findings."
        )


# ── Main supervisor node ──────────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> dict:
    """
    Deterministic supervisor — no LLM calls for routing.

    Pipeline order: profile_agent → stat_agent → chart_agent → narrative → end
    Chat turns:     routed directly to chat_agent (no re-running EDA).
    """
    iteration = state.get("iteration_count", 0) + 1

    # ── Safety circuit-breaker ────────────────────────────────────────────────
    if iteration > 20:
        return {"next_agent": "end", "iteration_count": iteration}

    # ── Fast-path: chat / export turn ─────────────────────────────────────────
    # The /chat endpoint explicitly sets next_agent="chat_agent" in state.
    # Honour it without any LLM call.
    if state.get("next_agent") == "chat_agent":
        return {"next_agent": "chat_agent", "iteration_count": iteration}

    # ── Deterministic EDA pipeline ────────────────────────────────────────────
    profile_done   = state.get("profile_report") is not None
    stat_done      = len(state.get("stat_results", [])) > 0
    chart_done     = len(state.get("chart_specs", [])) > 0
    insight_done   = len(state.get("insights", [])) > 0
    narrative_done = state.get("narrative_summary") is not None

    if not profile_done:
        return {"next_agent": "profile_agent", "iteration_count": iteration}

    if not stat_done:
        return {"next_agent": "stat_agent", "iteration_count": iteration}

    if not chart_done:
        return {"next_agent": "chart_agent", "iteration_count": iteration}

    if not insight_done:
        return {"next_agent": "insight_agent", "iteration_count": iteration}

    # ── Narrative: one LLM call, then done ───────────────────────────────────
    if not narrative_done:
        narrative = _generate_narrative(state)
        return {
            "next_agent": "end",
            "iteration_count": iteration,
            "narrative_summary": narrative,
        }

    return {"next_agent": "end", "iteration_count": iteration}


# ── Routing edge function ─────────────────────────────────────────────────────

def route_to_agent(state: AgentState) -> str:
    """Conditional edge: maps state.next_agent → node name."""
    if state.get("analysis_complete") or state.get("iteration_count", 0) > 20:
        return "end"
    agent = state.get("next_agent", "end")
    valid = {"profile_agent", "stat_agent", "chart_agent", "insight_agent", "chat_agent", "end"}
    return agent if agent in valid else "end"
