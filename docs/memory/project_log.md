# PRISM Project Log

## Project Overview

**PRISM — Agentic Analyst**
Multi-agent data intelligence system built with LangGraph + Gemini + FastAPI + Streamlit.

---

## Build Session: 2026-04-04

### What Was Built

| File | Purpose |
|------|---------|
| `backend/graph/state.py` | Shared AgentState TypedDict + all dataclasses |
| `backend/tools/inspect_tools.py` | Profile Agent Phase 1: 8 inspection tools |
| `backend/tools/clean_tools.py` | Profile Agent Phase 2: 5 cleaning tools |
| `backend/tools/stat_tools.py` | Stat Agent: pearson, chi_square, ttest, anova, time_trend, segment |
| `backend/tools/chart_tools.py` | Chart Agent: 9 Plotly chart tools (PRISM dark theme) |
| `backend/tools/chat_tools.py` | Chat Agent: execute_pandas, auto_chart, query_profile |
| `backend/tools/export_tools.py` | Chat Agent exports: CSV, charts ZIP, PDF report |
| `backend/graph/agents/supervisor.py` | Orchestrator + routing logic + narrative writer |
| `backend/graph/agents/profile_agent.py` | Inspect + clean agent (Gemini 2.0 Flash) |
| `backend/graph/agents/stat_agent.py` | Statistics + time-series agent |
| `backend/graph/agents/chart_agent.py` | Visualisation agent with self-review loop |
| `backend/graph/agents/chat_agent.py` | On-demand analyst + export handler |
| `backend/graph/graph.py` | LangGraph StateGraph assembly + MemorySaver |
| `backend/main.py` | FastAPI: /upload /chat /export /health |
| `frontend/app.py` | Streamlit: 6-tab dark UI + chat + exports |

### Key Design Decisions

1. **Gemini 2.0 Flash** used for all agents (not Claude/Anthropic)
2. **uv** used for package management (no pip/poetry)
3. **No agent naming convention** — agents are profile_agent, stat_agent, chart_agent, chat_agent
4. **LangGraph MemorySaver** checkpointer — full session memory per thread_id
5. **PRISM dark theme** — #0f172a background, #6366f1 indigo accent throughout

### Model

`gemini-2.5-flash-preview-04-17` — used across all 5 agents (supervisor, profile, stat, chart, chat)

### venv

Single uv project at `prism/` root — one `.venv`, one `pyproject.toml`, covers backend + frontend.

### Deployment Status

- Local development only (user confirmed — do not deploy yet)
- Backend: `uv run uvicorn backend/main:app --reload`
- Frontend: `uv run streamlit run frontend/app.py`

---

## How to Run

### 1. Set up environment

```bash
cd prism
cp backend/.env.example backend/.env
# .env already has GEMINI_API_KEY filled in
```

### 2. Sync dependencies (one command)

```bash
cd prism
uv sync
```

### 3. Start backend

```bash
cd prism
uv run uvicorn backend.main:app --reload --port 8000
```

### 4. Start frontend (separate terminal)

```bash
cd prism
BACKEND_URL=http://localhost:8000 uv run streamlit run frontend/app.py
```

### 4. Open browser

http://localhost:8501

---

## Known Limitations / TODO

- [ ] kaleido chart PNG export needs `uv add kaleido` (heavy package, excluded from initial install)
- [ ] Agent JSON parsing is regex-based; LangGraph structured output would be more robust
- [ ] SESSIONS dict in main.py is in-memory only (resets on restart) — consider Redis for prod
- [ ] execute_pandas uses subprocess isolation (secure); auto_chart uses exec() (internal only)
