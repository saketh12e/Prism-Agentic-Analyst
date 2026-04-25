<div align="center">

# PRISM
### Agentic Data Intelligence

**Upload a CSV. Five specialised AI agents analyse it end-to-end — profiling, cleaning, statistics, autonomous insights, visualisation, and natural-language chat — without a single line of user code.**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-FF6B35?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Gemini](https://img.shields.io/badge/Gemini-Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
</div>

---

## What is PRISM?

PRISM is a **fully agentic EDA (Exploratory Data Analysis) system** built on LangGraph. Drop any CSV and a deterministic supervisor orchestrates five specialised AI agents that run sequentially — each one building on the last — to produce a complete data intelligence report, a scored data quality assessment, autonomous hypotheses tested live on your data, interactive charts, and a natural-language interface for follow-up questions.

No notebooks. No code. No manual steps.

---

## Pipeline Architecture

```
                         ┌──────────────────────────────────────────┐
                         │            PRISM LangGraph                │
                         │                                           │
   CSV Upload            │   ┌──────────────┐                       │
  ──────────────────────►│   │  Supervisor   │  (deterministic)      │
                         │   └──────┬───────┘                       │
                         │          │                                │
                         │    ┌─────▼───────────────────────────┐   │
                         │    │  Phase 1 · Profile Agent         │   │
                         │    │  • Inspect: shape, dtypes,       │   │
                         │    │    nulls, skewness, outliers      │   │
                         │    │  • Clean: fill nulls, fix dtypes, │   │
                         │    │    cap outliers, dedup            │   │
                         │    │  → ProfileReport + clean CSV      │   │
                         │    │  → Data Quality Score (0–100)     │   │
                         │    └─────────────────────────────────-┘   │
                         │          │                                │
                         │    ┌─────▼───────────────────────────┐   │
                         │    │  Phase 2 · Stat Agent            │   │
                         │    │  • Pearson correlation           │   │
                         │    │  • t-test / ANOVA                │   │
                         │    │  • Chi-square                    │   │
                         │    │  • Time-series trend             │   │
                         │    │  → StatResult[]                  │   │
                         │    └─────────────────────────────────-┘   │
                         │          │                                │
                         │    ┌─────▼───────────────────────────┐   │
                         │    │  Phase 3 · Chart Agent           │   │
                         │    │  • Selects chart types from      │   │
                         │    │    profile + stat context        │   │
                         │    │  • Generates 9 Plotly chart types│   │
                         │    │  • Self-reviews each chart       │   │
                         │    │  → ChartSpec[]                   │   │
                         │    └─────────────────────────────────-┘   │
                         │          │                                │
                         │    ┌─────▼───────────────────────────┐   │
                         │    │  Phase 4 · Insight Agent  ★ NEW  │   │
                         │    │  • Generates 3–5 testable        │   │
                         │    │    data hypotheses autonomously  │   │
                         │    │  • Executes pandas code to test  │   │
                         │    │    each hypothesis live          │   │
                         │    │  • Evaluates evidence & assigns  │   │
                         │    │    verdict + confidence score    │   │
                         │    │  → DataInsight[]                 │   │
                         │    └─────────────────────────────────-┘   │
                         │          │                                │
                         │    ┌─────▼───────────────────────────┐   │
                         │    │  Supervisor · Narrative          │   │
                         │    │  • LLM writes 2–3 paragraph      │   │
                         │    │    plain-English summary         │   │
                         │    └─────────────────────────────────-┘   │
                         │                                           │
   Chat Question         │   ┌──────────────────────────────────┐   │
  ──────────────────────►│   │  Chat Agent  (on-demand)          │   │
                         │   │  • execute_pandas sandbox         │   │
   Answer + Chart        │   │  • auto_chart_from_query          │   │
  ◄──────────────────────│   │  • query_profile                  │   │
                         │   │  • export CSV / ZIP / PDF         │   │
                         │   └──────────────────────────────────┘   │
                         └──────────────────────────────────────────┘
```

> All agents share an `AgentState` TypedDict flowing through a LangGraph `StateGraph`. A `MemorySaver` checkpointer persists each session by `thread_id`, so chat history and all agent outputs survive across every turn.

---

## Agent Roster

| # | Agent | Role | Key Outputs |
|---|-------|------|-------------|
| — | **Supervisor** | Deterministic router + narrative writer | Routing decisions, narrative summary |
| 1 | **Profile Agent** | Data inspector & cleaner | `ProfileReport`, `CleaningSummary`, clean CSV |
| 2 | **Stat Agent** | Statistical test runner | `StatResult[]`, `TimeSeriesResult[]` |
| 3 | **Chart Agent** | Visualisation designer with self-review | `ChartSpec[]` (9 chart types) |
| 4 | **Insight Agent** ★ | Autonomous hypothesis generator & tester | `DataInsight[]` (verdicts + confidence) |
| — | **Chat Agent** | On-demand analyst & export handler | Answers, custom charts, CSV/ZIP/PDF |

---

## Features

### Core EDA Pipeline
- **Zero-code EDA** — upload any CSV; the full 4-phase pipeline runs automatically
- **Auto-cleaning** — fills nulls (mean / median / mode), caps outliers (IQR), fixes dtypes, removes duplicates
- **Statistical tests** — Pearson correlation, t-test, ANOVA, chi-square, time-series trend analysis
- **Smart charts** — 9 Plotly chart types (histogram, heatmap, scatter, bar, box, violin, pairplot, line, missing-heatmap) selected based on data profile and statistical findings, with self-review loop
- **LLM narrative** — one Gemini call produces a 2–3 paragraph plain-English business summary

### ★ New: Data Quality Score
- **Weighted 0–100 score** across four dimensions: Completeness (35%), Uniqueness (25%), Validity (20%), Consistency (20%)
- **Letter grade** (A–D) with a plain-English verdict
- **Per-dimension progress bars** shown prominently in the Data Quality tab
- Computed instantly from the profile — no extra agent needed

### ★ New: Autonomous Insight Agent
- **Self-generated hypotheses** — the agent creates its own 3–5 testable questions; never prompted with what to test
- **Live code execution** — writes and runs pandas code in an isolated subprocess to gather evidence
- **Self-evaluation** — assigns each hypothesis a verdict (`confirmed` / `refuted` / `inconclusive`) and a confidence score (0.0 – 1.0)
- **Transparent** — test code is shown in collapsible expanders so every finding is auditable
- The most agentic component in PRISM: it reasons → plans → executes → evaluates in a full ReAct loop

### Chat & Export
- **Natural-language chat** — ask any question; the agent writes and executes pandas code live
- **On-demand charts** — chat produces new Plotly charts inline
- **Export suite** — cleaned CSV, all charts as PNG ZIP bundle, full PDF analysis report

### Developer Experience
- **86-test suite** — pytest coverage for all tools, utilities, quality scoring, and insight execution
- **LangSmith tracing** — every agent run is observable end-to-end
- **Dark-mode UI** — modern Streamlit dashboard with 7-tab layout

---

## UI Tabs

| Tab | Contents |
|-----|----------|
| **Overview** | Key metrics, LLM narrative summary, raw data preview |
| **Data Quality** | ★ Quality score + grade + dimension breakdown, cleaning actions, missing-value map |
| **Distributions** | Histograms, box plots, violin plots |
| **Correlations** | Heatmap, scatter plots, pair plot, time-series lines |
| **Statistics** | Statistical test results — significant findings highlighted |
| **Insights** ★ | Autonomous hypotheses — verdict cards, confidence bars, test code |
| **Chat & Export** | Natural-language Q&A, download cleaned CSV / charts ZIP / PDF report |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph` + `MemorySaver` |
| LLM | Google Gemini via `langchain-google-genai` |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Data layer | pandas 3, NumPy 2, SciPy |
| Charts | Plotly Express + Graph Objects (dark theme) |
| PDF export | ReportLab |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Observability | LangSmith |
| Testing | pytest 9 + pytest-asyncio |

---

## Project Structure

```text
prism/
├── backend/
│   ├── main.py                    # FastAPI — /upload /chat /export /health
│   ├── .env.example               # Environment variable template
│   ├── graph/
│   │   ├── graph.py               # LangGraph StateGraph assembly + MemorySaver
│   │   ├── state.py               # AgentState + ProfileReport, StatResult,
│   │   │                          #   ChartSpec, DataInsight, CleaningSummary …
│   │   ├── utils.py               # extract_text, parse_json_from_text, last_ai_text
│   │   └── agents/
│   │       ├── supervisor.py      # Deterministic router + narrative writer
│   │       ├── profile_agent.py   # Phase 1 — Inspect → Clean (15 tools)
│   │       ├── stat_agent.py      # Phase 2 — Statistical tests (8 tools)
│   │       ├── chart_agent.py     # Phase 3 — Visualisation generation (9 tools)
│   │       ├── insight_agent.py   # Phase 4 ★ — Autonomous hypothesis testing
│   │       └── chat_agent.py      # On-demand analyst + export handler
│   └── tools/
│       ├── inspect_tools.py       # get_shape, get_dtypes, get_nulls, get_outliers …
│       ├── clean_tools.py         # fix_nulls, remove_duplicates, cap_outliers …
│       ├── stat_tools.py          # run_pearson, run_ttest, run_anova, run_time_trend …
│       ├── chart_tools.py         # plot_histogram, plot_correlation_heatmap … (9 tools)
│       ├── chat_tools.py          # execute_pandas, auto_chart_from_query, query_profile
│       ├── export_tools.py        # export_clean_csv, export_charts_zip, generate_pdf_report
│       ├── insight_tools.py       # ★ test_hypothesis — isolated pandas subprocess executor
│       └── quality_tools.py       # ★ compute_quality_score — weighted 0-100 quality scoring
├── tests/
│   ├── conftest.py                # Shared fixtures (sample CSVs, correlated data …)
│   ├── test_utils.py              # 17 tests — extract_text, parse_json, last_ai_text
│   ├── test_quality_tools.py      # 13 tests — quality score dimensions, grades, weights
│   ├── test_inspect_tools.py      # 22 tests — all 8 inspection tools
│   ├── test_stat_tools.py         # 17 tests — Pearson, chi-square, t-test, ANOVA …
│   └── test_insight_tools.py      # 7 tests  — hypothesis executor edge cases
├── frontend/
│   └── app.py                     # Streamlit dark-mode dashboard (7-tab layout)
├── docs/
│   └── memory/
│       └── project_log.md         # Build log and architectural decisions
├── pyproject.toml                 # Single uv project (backend + frontend + dev deps)
└── uv.lock                        # Pinned dependency lockfile
```

---

## Quickstart

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Google AI Studio](https://aistudio.google.com/) API key

### 1 · Clone and install

```bash
git clone https://github.com/your-username/prism.git
cd prism
uv sync
```

### 2 · Configure environment

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

### 3 · Run

**Terminal 1 — Backend**

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

**Terminal 2 — Frontend**

```bash
BACKEND_URL=http://localhost:8000 uv run streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) and upload any CSV.

---

## Testing

```bash
uv run python -m pytest backend/tests/ -v
```

```
86 passed in 3.5s  ✓
```

The test suite covers all pure-logic modules without requiring an API key or LLM:

| Module | Tests | Covers |
|--------|-------|--------|
| `test_utils.py` | 17 | `extract_text`, `parse_json_from_text`, `last_ai_text` |
| `test_quality_tools.py` | 13 | All 4 scoring dimensions, grade thresholds, weight formula |
| `test_inspect_tools.py` | 22 | All 8 inspection tools against temporary CSV fixtures |
| `test_stat_tools.py` | 17 | Pearson, chi-square, t-test, ANOVA, correlation matrix, segment compare |
| `test_insight_tools.py` | 7 | Hypothesis executor — success, failure, timeout, output cap |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/upload` | Upload CSV → triggers full EDA pipeline; returns all results in one response |
| `POST` | `/chat` | Send a question or export request to the Chat Agent |
| `GET` | `/export/{session_id}/{type}` | Download export: `csv` \| `charts_zip` \| `pdf` |

### `POST /upload` — example response

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "profile": {
    "shape": [1000, 12],
    "numeric_cols": ["revenue", "units"],
    "null_counts": {}
  },
  "cleaning": {
    "nulls_fixed": {},
    "duplicates_removed": 0,
    "rows_before": 1002,
    "rows_after": 1000
  },
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
  "charts": [
    { "chart_type": "histogram", "title": "Distribution of Revenue", "plotly_json": "..." }
  ],
  "quality_score": {
    "overall": 91.5,
    "completeness": 98.0,
    "uniqueness": 100.0,
    "validity": 83.3,
    "consistency": 87.5,
    "grade": "A",
    "verdict": "Excellent — data is analysis-ready with minimal issues."
  },
  "insights": [
    {
      "hypothesis": "Revenue is significantly higher in Q4 than other quarters",
      "finding": "Q4 average revenue ($142k) is 2.3x the Q1–Q3 average ($61k).",
      "verdict": "confirmed",
      "confidence": 0.92,
      "supporting_stat": "Q4 mean: 142000, overall mean: 87000"
    }
  ],
  "narrative": "The dataset contains 1,000 rows across 12 columns …",
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

The original supervisor used an LLM to decide which agent to run next, wasting 1–2 API calls per hop and producing unreliable routing decisions.

The rewritten supervisor is pure Python:

```text
profile_done?   → NO  →  profile_agent
stat_done?      → NO  →  stat_agent
chart_done?     → NO  →  chart_agent
insight_done?   → NO  →  insight_agent        ← Phase 4
narrative_done? → NO  →  call LLM once, write summary, END
next_agent == "chat_agent"?  →  fast-path directly, skip all checks
```

The LLM is invoked **exactly once** per full pipeline run — to write the narrative. All routing is deterministic Python.

### Autonomous Insight Agent — Emergent Hypothesis Testing

The Insight Agent is the most agentic component in PRISM. After the EDA pipeline completes, it independently:

1. **Generates** 3–5 data-specific hypotheses it has not been told to test
2. **Writes** pandas code for each, assigning the result to `output`
3. **Executes** each test in an isolated subprocess via `test_hypothesis()`
4. **Evaluates** the evidence and assigns `confirmed` / `refuted` / `inconclusive` with a confidence score

Every finding is reproducible — the test code is stored in `DataInsight.test_code` and shown in the UI.

### Data Quality Score — Weighted Dimensional Scoring

The quality scorer runs synchronously after the profile, with no extra agent call:

| Dimension | Weight | Penalises |
|-----------|--------|-----------|
| Completeness | 35% | Average null rate across all columns |
| Uniqueness | 25% | Duplicate row percentage |
| Validity | 20% | Columns with dtype mismatch issues |
| Consistency | 20% | Outlier rate in numeric columns (×2) |

Scores clamp to `[0, 100]` regardless of input extremes. Grade thresholds: A ≥ 90, B ≥ 75, C ≥ 60, D < 60.

### Chart JSON: Tool Message Extraction

Plotly figures serialise to 3,000–8,000+ character JSON strings. Relaying these through the LLM output caused silent truncation and parse errors. The fix bypasses the LLM output entirely — charts are extracted directly from `ToolMessage` objects in the LangGraph message history:

```python
for msg in agent_messages:
    if msg.tool_call_id maps to a chart tool:
        json.loads(msg.content)   # validate
        store as ChartSpec         # direct from tool — no LLM relay
```

### Gemini Content Normalisation

Gemini's `response.content` returns either a `str` or a `list[dict]` of content parts. A shared `extract_text()` utility handles both forms across all agents. Without it, every agent crashed with `AttributeError: 'list' object has no attribute 'strip'`.

### Guaranteed Clean CSV

The Profile Agent pre-copies the raw CSV to `output_path` before invoking the LLM cleaning loop. If the LLM decides no cleaning is needed, downstream agents still find a valid file instead of raising `FileNotFoundError`.

### Session ID Through State

`session_id` is now threaded through `AgentState` from the moment the `/upload` request creates it. Export tools use this to write all session artefacts to an isolated `{session_id}/` subdirectory — preventing file collisions across concurrent sessions.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ | Google AI Studio API key |
| `GEMINI_MODEL` | ✅ | Model ID (e.g. `gemini-3-flash-preview`) |
| `LANGCHAIN_TRACING_V2` | optional | Enable LangSmith tracing (`true` / `false`) |
| `LANGCHAIN_API_KEY` | optional | LangSmith API key |
| `LANGCHAIN_PROJECT` | optional | LangSmith project name |
| `EXPORT_DIR` | optional | Export output directory (default: `/tmp/prism_exports`) |
| `BACKEND_URL` | optional | Backend URL seen by Streamlit (default: `http://localhost:8000`) |

---

## License

MIT
