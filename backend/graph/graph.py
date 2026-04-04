"""
PRISM — LangGraph Assembly
Wires the Supervisor + 4 agents into a compiled, checkpointed StateGraph.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from graph.agents import chart_agent, chat_agent, profile_agent, stat_agent, supervisor
from graph.state import AgentState


def build_graph():
    """
    Build and compile the PRISM LangGraph.

    Graph topology:
        supervisor
            ├── profile_agent ──┐
            ├── stat_agent     ──┤
            ├── chart_agent    ──┤→ supervisor (loop back)
            ├── chat_agent     ──┘
            └── END

    Every sub-agent returns to supervisor after completing.
    The supervisor decides the next hop based on pipeline state.
    """
    builder = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("supervisor",    supervisor.supervisor_node)
    builder.add_node("profile_agent", profile_agent.profile_agent_node)
    builder.add_node("stat_agent",    stat_agent.stat_agent_node)
    builder.add_node("chart_agent",   chart_agent.chart_agent_node)
    builder.add_node("chat_agent",    chat_agent.chat_agent_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("supervisor")

    # ── Conditional routing from supervisor ──────────────────────────────────
    builder.add_conditional_edges(
        "supervisor",
        supervisor.route_to_agent,
        {
            "profile_agent": "profile_agent",
            "stat_agent":    "stat_agent",
            "chart_agent":   "chart_agent",
            "chat_agent":    "chat_agent",
            "end":           END,
        },
    )

    # ── Every sub-agent loops back to supervisor ──────────────────────────────
    for agent_name in ["profile_agent", "stat_agent", "chart_agent", "chat_agent"]:
        builder.add_edge(agent_name, "supervisor")

    # ── Compile with in-memory checkpointer (enables thread_id persistence) ──
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Singleton — imported by main.py
graph = build_graph()
