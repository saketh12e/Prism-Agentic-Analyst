# PRISM — Agentic Data Intelligence

> **Upload a CSV. Four specialized AI agents analyse it end-to-end — profiling, cleaning, statistics, visualisation, and natural-language chat — without a single line of user code.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-FF6B35?logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

---

## Architecture

```text
                         ┌─────────────────────────────────────┐
                         │          PRISM LangGraph             │
                         │                                      │
   CSV Upload            │   ┌─────────────┐                   │
  ──────────────────────►│   │  Supervisor  │ (deterministic)   │
                         │   └──────┬──────┘                   │
                         │          │                           │
                         │    ┌─────▼──────────────────────┐   │
                         │    │  Phase 1: Profile Agent     │   │
                         │    │  • Inspect: shape, dtypes,  │   │
                         │    │    nulls, skewness, outliers│   │
                         │    │  • Clean:  fill nulls,      │   │
                         │    │    fix dtypes, cap outliers │   │
                         │    │  → ProfileReport + clean CSV│   │
                         │    └─────────────────────────────┘   │
                         │          │                           │
                         │    ┌─────▼──────────────────────┐   │
                         │    │  Phase 2: Stat Agent        │   │
                         │    │  • Pearson / Spearman corr  │   │
                         │    │  • t-test / ANOVA           │   │
                         │    │  • Chi-square               │   │
                         │    │  • Time-series trend        │   │
                         │    │  → StatResult[]             │   │
                         │    └─────────────────────────────┘   │
                         │          │                           │
                         │    ┌─────▼──────────────────────┐   │
                         │    │  Phase 3: Chart Agent       │   │
                         │    │  • Selects chart types from │   │
                         │    │    profile + stat context   │   │
                         │    │  • Generates Plotly figures │   │
                         │    │  • Self-reviews each chart  │   │
                         │    │  → ChartSpec[]              │   │
                         │    └─────────────────────────────┘   │
                         │          │                           │
                         │    ┌─────▼──────────────────────┐   │
                         │    │  Supervisor: Narrative      │   │
                         │    │  • LLM writes 2–3 paragraph │   │
                         │    │    plain-English summary    │   │
                         │    └─────────────────────────────┘   │
                         │                                      │
   Chat Question         │   ┌─────────────────────────────┐   │
  ──────────────────────►│   │  Chat Agent  (on-demand)    │   │
                         │   │  • execute_pandas sandbox   │   │
   Answer + Chart        │   │  • auto_chart_from_query    │   │
  ◄──────────────────────│   │  • query_profile            │   │
                         │   │  • export CSV / ZIP / PDF   │   │
                         │   └─────────────────────────────┘   │
                         └─────────────────────────────────────┘
```

**All agents share an `AgentState` TypedDict flowing through a LangGraph `StateGraph`. A `MemorySaver` checkpointer persists each session by `thread_id`, so chat history and all agent outputs survive across turns.**

---

## Features

- **Zero-code EDA** — upload any CSV and the full pipeline runs automatically
- **Auto-cleaning** — fills nulls (mean/median/mode), caps outliers, fixes dtypes, removes duplicates
- **Statistical tests** — Pearson correlation, t-test, ANOVA, chi-square, time-series trend
- **Smart charts** — 9 Plotly chart types selected based on data profile and statistical findings
- **Natural-language chat** — ask any question; the agent writes and executes pandas code live
- **Export suite** — cleaned CSV, chart ZIP bundle, full PDF report
- **LangSmith tracing** — every agent run is observable end-to-end
- **Dark-mode UI** — modern Streamlit dashboard with 6-tab layout

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| Agent orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` + `MemorySaver` |
| LLM | Google Gemini via `langchain-google-genai` |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Data layer | pandas 3, NumPy 2, SciPy |
| Charts | Plotly Express + Graph Objects (dark theme) |
| PDF export | ReportLab |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Observability | LangSmith |

---

## Project Structure

```text
prism/
├── backend/
│   ├── main.py                  # FastAPI — /upload /chat /export /health
│   ├── .env.example             # Environment variable template
│   ├── graph/
│   │   ├── graph.py             # LangGraph StateGraph assembly + MemorySaver
│   │   ├── state.py             # AgentState + ProfileReport, StatResult, ChartSpec …
│   │   ├── utils.py             # Shared: extract_text, parse_json_from_text, last_ai_text
│   │   └── agents/
│   │       ├── supervisor.py    # Deterministic router + narrative writer
│   │       ├── profile_agent.py # Inspect → clean (15 tools)
│   │       ├── stat_agent.py    # Statistical tests (8 tools)
│   │       ├── chart_agent.py   # Visualisation generation (9 tools)
│   │       └── chat_agent.py    # On-demand analyst + export handler
│   └── tools/
│       ├── inspect_tools.py     # get_shape, get_dtypes, get_nulls, get_outliers …
│       ├── clean_tools.py       # fix_nulls, remove_duplicates, cap_outliers …
│       ├── stat_tools.py        # run_pearson, run_ttest, run_anova, run_time_trend …
│       ├── chart_tools.py       # plot_histogram, plot_correlation_heatmap … (9 tools)
│       ├── chat_tools.py        # execute_pandas, auto_chart_from_query, query_profile
│       └── export_tools.py      # export_clean_csv, export_charts_zip, generate_pdf_report
├── frontend/
│   └── app.py                   # Streamlit dark-mode dashboard (6-tab layout)
├── docs/
│   └── memory/
│       └── project_log.md       # Build log and architectural decisions
├── pyproject.toml               # Single uv project (backend + frontend deps)
└── uv.lock                      # Pinned dependency lockfile (commit this)
```

---

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Google AI Studio](https://aistudio.google.com/) API key

### 1. Clone and install

```bash
git clone https://github.com/your-username/prism.git
cd prism
uv sync
```

### 2. Configure environment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and fill in your GEMINI_API_KEY
```

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3-flash-preview

# Optional — LangSmith observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=prism-analyst
```

### 3. Run

**Terminal 1 — Backend:**

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**

```bash
BACKEND_URL=http://localhost:8000 uv run streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) and upload any CSV.

---

## API Reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness check |
| `POST` | `/upload` | Upload CSV → triggers full EDA; returns all results in one response |
| `POST` | `/chat` | Send a question or export request to the Chat Agent |
| `GET` | `/export/{session_id}/{type}` | Download export: `csv` \| `charts_zip` \| `pdf` |

### POST `/upload` example response

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "profile": { "shape": [1000, 12], "numeric_cols": ["revenue", "units"], "null_counts": {} },
  "cleaning": { "nulls_fixed": {}, "duplicates_removed": 0, "rows_before": 1002 },
  "stat_results": [
    {
      "test_name": "pearson",
      "col_a": "revenue",
      "col_b": "units",
      "p_value": 0.0001,
      "significant": true,
      "interpretation": "Revenue and units sold are strongly positively correlated."
    }
  ],
  "charts": [{ "chart_type": "histogram", "title": "Distribution of Revenue", "plotly_json": "..." }],
  "narrative": "The dataset contains 1,000 rows across 12 columns ...",
  "errors": []
}
```

---

## Deployment

### Railway (Backend API)

1. Push this repo to GitHub
2. Create a new Railway project → **Deploy from GitHub repo**
3. Set **Root Directory** to `prism`
4. Set **Start Command** to `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Add all variables from `.env.example` in the Railway **Variables** tab

### Streamlit Cloud (Frontend)

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
2. Connect your GitHub repo
3. Set **Main file path** to `prism/frontend/app.py`
4. In **Advanced settings → Secrets**, add:

```toml
BACKEND_URL = "https://your-railway-app.up.railway.app"
```

---

## Engineering Highlights

### Deterministic Supervisor — Zero LLM Calls for Routing

The original supervisor used an LLM to decide which agent to run next, wasting 1–2 API calls per step and producing unreliable, non-deterministic routing.

The rewritten supervisor is pure Python logic:

```text
profile_done?  → NO  → profile_agent
stat_done?     → NO  → stat_agent
chart_done?    → NO  → chart_agent
narrative?     → NO  → call LLM once, write summary, END
next_agent == "chat_agent"? → fast-path directly, skip all checks
```

The LLM is now invoked **exactly once** per full pipeline run — to write the narrative summary. Everything else is deterministic.

### Chart JSON: Tool Message Extraction

Plotly figures serialise to 3,000–8,000+ character JSON strings. The original approach asked the LLM to relay these strings verbatim in its output — causing silent truncation and `unexpected character` errors in the frontend.

The fix bypasses the LLM output entirely. Charts are extracted directly from `ToolMessage` objects in the LangGraph message history:

```python
for msg in agent_messages:
    if msg.tool_call_id maps to a chart tool:
        json.loads(msg.content)  # validate
        store as ChartSpec        # direct from tool — no LLM relay
```

### Gemini Content Normalisation

Gemini's `response.content` returns either a `str` or a `list[dict]` of content parts depending on the request. A shared `extract_text()` utility in `graph/utils.py` handles both forms across all five agents. Without this, every agent crashed with `AttributeError: 'list' object has no attribute 'strip'`.

### Guaranteed Clean CSV

The Profile Agent pre-copies the raw CSV to `output_path` before invoking the LLM cleaning loop. If the LLM decides no cleaning is needed, downstream agents still find a valid file at the expected path instead of raising `FileNotFoundError`.

### Minimal Chat State

The `/chat` endpoint previously reset `stat_results`, `chart_specs`, and `time_series_results` to `[]` on every request. With LangGraph's `operator.add` reducers, this was redundant. The endpoint now sends only the five fields that actually need to change per chat turn, keeping accumulated pipeline state clean.

---

## Environment Variables

| Variable | Required | Description |
| --- | --- | --- |
| `GEMINI_API_KEY` | ✅ | Google AI Studio API key |
| `GEMINI_MODEL` | ✅ | Model ID (e.g. `gemini-3-flash-preview`) |
| `LANGCHAIN_TRACING_V2` | optional | Enable LangSmith tracing (`true`/`false`) |
| `LANGCHAIN_API_KEY` | optional | LangSmith API key |
| `LANGCHAIN_PROJECT` | optional | LangSmith project name |
| `EXPORT_DIR` | optional | Export output directory (default: `/tmp/prism_exports`) |
| `BACKEND_URL` | optional | Backend URL seen by Streamlit (default: `http://localhost:8000`) |

---

## License

MIT
