"""
api/routes.py — FastAPI backend for the research assistant.

Exposes three endpoints:
    POST /research          — start a research run, returns the full result
    GET  /history           — retrieve recent sessions from SQLite (with reports)
    GET  /health            — simple health check

Run with:
    uvicorn api.routes:app --reload --port 8000
"""

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from graph.state import ResearchState
from graph.workflow import compiled_graph
from memory.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

# ── FastAPI app instance ──────────────────────────────────────────────
app = FastAPI(
    title       = "Agentic Research Assistant",
    description = "Autonomous research pipeline powered by LangGraph + Gemini",
    version     = "1.0.0",
)

# Allow Streamlit (port 8501) to call this API without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Shared SQLite store — one instance for the app lifetime
_db: Optional[SQLiteStore] = None

def get_db() -> SQLiteStore:
    global _db
    if _db is None:
        _db = SQLiteStore()
    return _db


# ── Request / Response models ─────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length = 5,
        max_length = 500,
        description= "The research question to investigate",
        example    = "How does CRISPR gene editing work?",
    )


class ResearchResponse(BaseModel):
    query:          str
    final_report:   Optional[str]
    iterations:     int
    findings_count: int
    status:         str
    error:          Optional[str]


class SessionSummary(BaseModel):
    id:             int
    query:          str
    iterations:     int
    findings_count: int
    created_at:     str
    final_report:   Optional[str]   # ← now included so the UI can render it


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Quick liveness check — returns 200 if the server is up."""
    return {"status": "ok", "service": "research-assistant"}


@app.post("/research", response_model=ResearchResponse)
def run_research(request: ResearchRequest):
    """
    Run the full research pipeline for a query.

    Executes all four agents (Planner → Researcher → Critic → Writer)
    synchronously and returns the final report.

    Note: This can take 30-90 seconds depending on query complexity
    and network speed. For production use, consider async + polling.
    """
    logger.info(f"API: research request — '{request.query[:60]}'")

    try:
        initial_state: ResearchState = {
            "query":        request.query,
            "plan":         [],
            "findings":     [],
            "gaps":         [],
            "iterations":   0,
            "final_report": None,
            "status":       "Starting...",
            "error":        None,
        }

        final_state = compiled_graph.invoke(initial_state)

        return ResearchResponse(
            query          = final_state["query"],
            final_report   = final_state.get("final_report"),
            iterations     = final_state.get("iterations", 0),
            findings_count = len(final_state.get("findings", [])),
            status         = final_state.get("status", ""),
            error          = final_state.get("error"),
        )

    except Exception as e:
        logger.error(f"Research pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=List[SessionSummary])
def get_history(limit: int = 10):
    """
    Return the most recent research sessions from SQLite.
    Now includes final_report so the UI can display past reports directly.

    Args:
        limit: Maximum number of sessions to return (default 10, max 50).
    """
    limit = min(limit, 50)   # safety cap
    try:
        db       = get_db()
        sessions = db.get_recent_sessions(limit=limit)
        return [
            SessionSummary(
                id             = s["id"],
                query          = s["query"],
                iterations     = s["iterations"],
                findings_count = s["findings_count"],
                created_at     = s["created_at"],
                final_report   = s.get("final_report"),   # ← now passed through
            )
            for s in sessions
        ]
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))