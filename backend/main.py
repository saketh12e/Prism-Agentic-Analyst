"""
PRISM — FastAPI Backend
Entrypoint for the multi-agent data analysis pipeline.

Endpoints:
  POST /upload          → trigger full EDA (profile → stat → chart)
  POST /chat            → send a question to the Chat Agent
  GET  /export/{sid}/{type} → download csv | charts_zip | pdf
  GET  /health          → liveness check
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

# ── Path bootstrap ────────────────────────────────────────────────────────────
# Must happen before any local imports so graph/ and tools/ are on sys.path
_BACKEND = Path(__file__).parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# ── Load .env FIRST — agent LLMs read GEMINI_MODEL at module import time ─────
from dotenv import load_dotenv
load_dotenv(dotenv_path=_BACKEND / ".env")

# ── LangSmith tracing (optional — only if key is set) ────────────────────────
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "prism-analyst-v1")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graph.graph import graph
from tools.quality_tools import compute_quality_score

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PRISM — Agentic Analyst API",
    version="1.0.0",
    description="Multi-agent data intelligence: Profile → Stat → Chart → Chat",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store: session_id → clean_csv_path ───────────────────────────────
SESSIONS: dict[str, str] = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_asdict(obj) -> dict:
    """Convert a dataclass to dict; return {} if None or conversion fails."""
    if obj is None:
        return {}
    try:
        return asdict(obj)
    except Exception:
        return {}


def _make_initial_state(csv_path: str) -> dict:
    return {
        "dataframe_path": csv_path,
        "user_query": (
            "Perform a complete exploratory data analysis. "
            "Inspect and clean the data. "
            "Run all relevant statistical tests. "
            "Generate all appropriate visualisations."
        ),
        "profile_report": None,
        "cleaning_summary": None,
        "stat_results": [],
        "time_series_results": [],
        "chart_specs": [],
        "narrative_summary": None,
        "chat_history": [],
        "export_paths": [],
        "next_agent": "profile_agent",
        "analysis_complete": False,
        "iteration_count": 0,
        "errors": [],
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "PRISM"}


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file → triggers the full EDA pipeline:
      Profile Agent → Stat Agent → Chart Agent → Supervisor (narrative)
    Returns all results in a single JSON response.
    """
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", prefix="prism_upload_")
    try:
        shutil.copyfileobj(file.file, tmp)
    finally:
        tmp.close()

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = graph.invoke(_make_initial_state(tmp.name), config=config)
    except Exception as exc:
        os.unlink(tmp.name)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    pr = result.get("profile_report")
    SESSIONS[session_id] = pr.clean_csv_path if pr else tmp.name

    quality_score = compute_quality_score(pr) if pr else {}

    return {
        "session_id": session_id,
        "profile": _safe_asdict(pr),
        "cleaning": _safe_asdict(result.get("cleaning_summary")),
        "stat_results": [_safe_asdict(s) for s in result.get("stat_results", [])],
        "time_series": [_safe_asdict(t) for t in result.get("time_series_results", [])],
        "charts": [_safe_asdict(c) for c in result.get("chart_specs", [])],
        "narrative": result.get("narrative_summary", ""),
        "quality_score": quality_score,
        "errors": result.get("errors", []),
    }


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Submit a user question or export request to the Chat Agent.
    The graph checkpointer carries full pipeline memory for the session.
    """
    if req.session_id not in SESSIONS:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Upload a CSV first.",
        )

    config = {"configurable": {"thread_id": req.session_id}}
    # Minimal state update — only override what needs to change for a chat turn.
    # DO NOT pass stat_results/chart_specs/etc. — they accumulate via reducers
    # and are already persisted in the checkpointer.
    state_update = {
        "user_query": req.message,
        "chat_history": [{"role": "user", "content": req.message}],
        "next_agent": "chat_agent",   # supervisor fast-paths on this
        "analysis_complete": False,
        "iteration_count": 0,
    }

    try:
        result = graph.invoke(state_update, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat error: {exc}") from exc

    history = result.get("chat_history", [])
    last_reply = next(
        (m["content"] for m in reversed(history) if m.get("role") == "assistant"),
        "",
    )

    return {
        "response": last_reply,
        "new_charts": [_safe_asdict(c) for c in result.get("chart_specs", [])],
        "exports": result.get("export_paths", []),
    }


@app.get("/export/{session_id}/{export_type}")
async def get_export(session_id: str, export_type: str):
    """
    Trigger file export for a completed session.
    export_type: 'csv' | 'charts_zip' | 'pdf'
    """
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    config = {"configurable": {"thread_id": session_id}}
    state_update = {
        "user_query": f"export {export_type}",
        "chat_history": [{"role": "user", "content": f"export {export_type}"}],
        "next_agent": "chat_agent",
        "analysis_complete": False,
        "iteration_count": 0,
    }

    try:
        result = graph.invoke(state_update, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Export error: {exc}") from exc

    exports = result.get("export_paths", [])
    match = next(
        (e for e in exports if export_type.replace("_", "") in e.get("type", "").replace("_", "")),
        None,
    )
    if not match:
        raise HTTPException(status_code=404, detail=f"Export '{export_type}' was not produced.")

    file_path = match["path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Export file not found on disk.")

    return FileResponse(
        path=file_path,
        filename=match.get("filename", os.path.basename(file_path)),
        media_type="application/octet-stream",
    )
