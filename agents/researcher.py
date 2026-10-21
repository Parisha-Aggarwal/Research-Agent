"""
agents/researcher.py — Researcher Agent.

Responsibility:
    For each target question (sub-questions on first run, gaps on re-runs),
    search the web, scrape the top results, and store findings in ChromaDB.
    Also appends raw finding dicts to state["findings"] for the Critic
    and Writer agents to read.

LangGraph contract:
    - Input:  ResearchState with 'plan' and optionally 'gaps' populated
    - Output: ResearchState with 'findings' appended and 'status' updated

On first run  → researches state["plan"]   (sub-questions from Planner)
On re-run     → researches state["gaps"]   (gaps identified by Critic)
"""

import logging
from typing import List, Dict

from tools.search import search_web
from tools.scrapper import scrape_page
from memory.vector_store import VectorStore
from graph.state import ResearchState
from config import MAX_SEARCH_RESULTS

logger = logging.getLogger(__name__)

# VectorStore is initialised once and reused across calls.
# Lazy-initialised on first use to avoid slow startup if not needed.
_vector_store: VectorStore = None


def _get_vector_store() -> VectorStore:
    """Return the shared VectorStore, initialising it on first call."""
    global _vector_store
    if _vector_store is None:
        logger.info("Initialising VectorStore (first researcher call)")
        _vector_store = VectorStore()
    return _vector_store


def run_researcher(state: ResearchState) -> ResearchState:
    """
    LangGraph node function — searches the web and stores findings.

    Decision logic:
        - If this is the first run (iterations == 0), research all
          sub-questions from the plan.
        - If this is a re-run (iterations > 0), research only the
          gaps identified by the Critic — faster and more targeted.

    Args:
        state: Current ResearchState.

    Returns:
        Updated ResearchState with new findings appended.
    """
    iterations = state.get("iterations", 0)

    # Decide what to research this iteration
    if iterations == 0:
        targets = state["plan"]
        logger.info(f"Researcher (iteration 0): researching {len(targets)} sub-questions")
    else:
        targets = state.get("gaps", [])
        logger.info(f"Researcher (iteration {iterations}): filling {len(targets)} gaps")

    if not targets:
        logger.warning("No targets to research — skipping")
        state["status"] = "Research skipped (no targets)"
        return state

    state["status"] = f"Researching {len(targets)} question(s)..."

    store = _get_vector_store()
    new_findings: List[Dict] = []

    for i, target in enumerate(targets, 1):
        logger.info(f"  [{i}/{len(targets)}] Searching: '{target[:60]}'")
        findings_for_target = _research_one_target(target, store)
        new_findings.extend(findings_for_target)
        logger.info(f"  [{i}/{len(targets)}] Got {len(findings_for_target)} findings")

    # Append to existing findings (don't overwrite — previous runs' findings
    # are still in ChromaDB and still valuable for the Writer)
    state["findings"] = state.get("findings", []) + new_findings

    total = len(state["findings"])
    logger.info(f"Researcher done. Total findings so far: {total}")
    state["status"] = f"Research complete — {total} total findings"

    return state


def _research_one_target(target: str, store: VectorStore) -> List[Dict]:
    """
    Search, scrape, and store findings for a single sub-question or gap.

    Args:
        target: The sub-question or gap string to research.
        store:  Shared VectorStore instance.

    Returns:
        List of finding dicts: {source_url, content, sub_question}.
        Empty list if search returns nothing or all scrapes fail.
    """
    findings = []

    # Step 1: Search the web for this target
    search_results = search_web(target)

    if not search_results:
        logger.warning(f"No search results for: '{target[:50]}'")
        return []

    # Step 2: Scrape each result URL
    for result in search_results:
        url     = result.get("url", "")
        snippet = result.get("snippet", "")

        if not url:
            continue

        # Scrape the full page content
        page_content = scrape_page(url)

        # If scraping failed, fall back to the DuckDuckGo snippet.
        # A snippet is short but better than nothing.
        if not page_content and snippet:
            logger.debug(f"Scrape failed, using snippet for: {url[:50]}")
            page_content = snippet

        if not page_content:
            logger.debug(f"No content at all for: {url[:50]}")
            continue

        # Step 3: Store in ChromaDB for semantic retrieval later
        store.add_finding(
            url=url,
            content=page_content,
            sub_question=target,
        )

        # Step 4: Also keep a raw dict in state for the Critic and Writer.
        # The Critic reads these to check coverage; the Writer cites them.
        findings.append({
            "source_url":   url,
            "content":      page_content,
            "sub_question": target,
        })

    return findings