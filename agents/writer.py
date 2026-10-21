"""
agents/writer.py — Writer Agent.

Responsibility:
    Synthesise all collected findings into a well-structured markdown
    report with citations. This is the final agent in the graph.

LangGraph contract:
    - Input:  ResearchState with 'query', 'plan', 'findings'
    - Output: ResearchState with 'final_report' set (markdown string)

Strategy:
    1. Query ChromaDB semantically for the most relevant findings
    2. Build a focused prompt with those findings + citations
    3. Ask Gemini to write a structured report
    4. Store the session in SQLite
"""

import logging
from typing import List

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL
from graph.state import ResearchState
from memory.vector_store import VectorStore
from memory.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

_client      = genai.Client(api_key=GEMINI_API_KEY)
_vector_store: VectorStore = None
_sqlite_store: SQLiteStore = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_sqlite_store() -> SQLiteStore:
    global _sqlite_store
    if _sqlite_store is None:
        _sqlite_store = SQLiteStore()
    return _sqlite_store


WRITER_PROMPT = """You are an expert research report writer.

Write a comprehensive, well-structured research report based on the
findings provided below. The report must be in markdown format.

Research Query:
{query}

Research Sub-Questions Covered:
{plan}

Findings with Sources:
{findings_text}

Report Requirements:
- Start with a concise executive summary (2-3 sentences)
- Use clear markdown headers (##) for each major section
- Each major claim must be followed by a citation like [Source: URL]
- Include a "## Key Findings" section with bullet points
- End with a "## References" section listing all source URLs
- Be factual — only use information from the findings provided
- Length: 400-700 words

Write the report now:
"""


def run_writer(state: ResearchState) -> ResearchState:
    """
    LangGraph node function — generates the final research report.

    Args:
        state: Current ResearchState.

    Returns:
        Updated ResearchState with 'final_report' set and
        session saved to SQLite.
    """
    query    = state["query"]
    plan     = state["plan"]
    findings = state["findings"]

    logger.info(f"Writer running — {len(findings)} findings to synthesise")
    state["status"] = "Writing final report..."

    try:
        # ── Step 1: get best findings via semantic search ─────────────
        # Rather than dumping all findings into the prompt (could be huge),
        # we retrieve the top-N most relevant to the original query.
        relevant = _get_relevant_findings(query, findings)

        if not relevant:
            logger.warning("No relevant findings — writing from raw state findings")
            relevant = findings[:6]   # fallback: use first 6 directly

        # ── Step 2: build the findings block for the prompt ───────────
        findings_text = _format_findings_for_prompt(relevant)
        plan_text     = "\n".join(f"  - {q}" for q in plan)

        prompt = WRITER_PROMPT.format(
            query        = query,
            plan         = plan_text,
            findings_text= findings_text,
        )

        # ── Step 3: call Gemini ───────────────────────────────────────
        response = _client.models.generate_content(
            model   = GEMINI_MODEL,
            contents= prompt,
            config  = types.GenerateContentConfig(
                temperature=0.4,   # slight creativity for readable prose
            ),
        )
        report = response.text.strip()

        logger.info(f"Report generated — {len(report)} characters")

        state["final_report"] = report
        state["status"]       = "Report complete"
        state["error"]        = None

        # ── Step 4: persist session to SQLite ─────────────────────────
        _save_session(state)

        return state

    except Exception as e:
        logger.error(f"Writer failed: {e}")
        # Return a minimal fallback report so the UI doesn't show nothing
        state["final_report"] = _fallback_report(query, findings)
        state["status"]       = "Report complete (fallback)"
        state["error"]        = f"Writer warning: {e}"
        _save_session(state)
        return state


def _get_relevant_findings(query: str, findings: List[dict]) -> List[dict]:
    """
    Use ChromaDB semantic search to retrieve the most relevant findings.
    Falls back to returning all findings if vector store is unavailable.
    """
    try:
        store   = _get_vector_store()
        results = store.search_findings(query, n_results=min(6, len(findings)))

        # search_findings returns {content, url, sub_question, score}
        # remap to match our finding dict shape
        return [
            {
                "source_url":   r["url"],
                "content":      r["content"],
                "sub_question": r["sub_question"],
            }
            for r in results
        ]
    except Exception as e:
        logger.warning(f"Vector search failed in writer, using raw findings: {e}")
        return findings[:6]


def _format_findings_for_prompt(findings: List[dict]) -> str:
    """
    Format findings into a numbered, citation-ready block for the prompt.
    Keeps each finding to 400 chars to control prompt size.
    """
    lines = []
    for i, f in enumerate(findings, 1):
        url     = f.get("source_url", "unknown")
        content = f.get("content", "")[:400].replace("\n", " ")
        sub_q   = f.get("sub_question", "")
        lines.append(
            f"[{i}] Topic: {sub_q}\n"
            f"     URL: {url}\n"
            f"     Content: {content}..."
        )
    return "\n\n".join(lines)


def _fallback_report(query: str, findings: List[dict]) -> str:
    """
    Minimal markdown report built without Gemini — used if API call fails.
    Ensures the user always sees something meaningful.
    """
    sources = list({f.get("source_url","") for f in findings if f.get("source_url")})
    refs    = "\n".join(f"- {s}" for s in sources[:10]) or "- No sources collected"
    return f"""# Research Report: {query}

## Note
*This report was generated in fallback mode due to an API error.
The sources below were collected successfully.*

## Sources Collected ({len(findings)} findings)

{refs}

## Recommendation
Please retry the query or check your Gemini API key.
"""


def _save_session(state: ResearchState) -> None:
    """Persist the completed session to SQLite (best-effort)."""
    try:
        db = _get_sqlite_store()
        db.save_session(
            query          = state["query"],
            final_report   = state.get("final_report"),
            iterations     = state.get("iterations", 0),
            findings_count = len(state.get("findings", [])),
        )
    except Exception as e:
        logger.warning(f"Failed to save session to SQLite: {e}")