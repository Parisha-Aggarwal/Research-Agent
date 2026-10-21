"""
agents/critic.py — Critic Agent.

Responsibility:
    Review what the Researcher has found so far and decide:
      (a) Is the research complete enough to write a report?
      (b) If not, what specific topics are still missing?

LangGraph contract:
    - Input:  ResearchState with 'query', 'plan', 'findings', 'iterations'
    - Output: ResearchState with 'gaps' set and 'iterations' incremented
              The graph's conditional edge reads 'gaps' to decide routing:
                  gaps == [] or iterations >= MAX_ITERATIONS → Writer
                  gaps != [] and iterations < MAX_ITERATIONS → Researcher

Gemini is asked to return JSON with exactly two fields:
    {
        "satisfied": true | false,
        "gaps": ["missing topic A", "missing topic B"]
    }
"""

import json
import logging
import re
from typing import List

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL, MAX_ITERATIONS
from graph.state import ResearchState

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=GEMINI_API_KEY)

# How many chars of each finding to include in the critic prompt.
# Enough context to judge coverage; short enough to stay fast.
FINDING_PREVIEW_CHARS = 300


CRITIC_PROMPT = """You are a critical research reviewer.

Your job is to read a research query, the planned sub-questions, and a
summary of findings collected so far — then decide if the research is
complete enough to write a good report.

Research Query:
{query}

Planned Sub-Questions:
{plan}

Findings Collected So Far ({findings_count} total):
{findings_summary}

Instructions:
- If the findings adequately cover the key sub-questions, set satisfied to true and gaps to [].
- If important sub-questions are missing or poorly covered, set satisfied to false
  and list 1-3 specific search queries that would fill the gaps.
- Gaps should be concrete, searchable queries — not vague topics.
- Return ONLY valid JSON. No explanation, no markdown, no code fences.

Return format:
{{
    "satisfied": true or false,
    "gaps": ["specific gap query 1", "specific gap query 2"]
}}
"""


def run_critic(state: ResearchState) -> ResearchState:
    """
    LangGraph node function — reviews findings and identifies gaps.

    Args:
        state: Current ResearchState.

    Returns:
        Updated ResearchState with:
          - 'gaps'       set (empty list if satisfied)
          - 'iterations' incremented by 1
          - 'status'     updated
    """
    query      = state["query"]
    plan       = state["plan"]
    findings   = state["findings"]
    iterations = state.get("iterations", 0)

    logger.info(f"Critic running (iteration {iterations}) — {len(findings)} findings to review")
    state["status"] = f"Reviewing findings (iteration {iterations + 1})..."

    # Always increment — this tracks how many full loops have completed
    state["iterations"] = iterations + 1

    # Safety valve: if we've already hit the max, stop looping regardless
    if iterations >= MAX_ITERATIONS:
        logger.warning(f"Max iterations ({MAX_ITERATIONS}) reached — forcing Writer")
        state["gaps"]   = []
        state["status"] = f"Max iterations reached — proceeding to report"
        return state

    # If no findings at all, flag as gap immediately without calling Gemini
    if not findings:
        logger.warning("No findings to review — sending back to Researcher")
        state["gaps"]   = plan[:2]   # re-try first two sub-questions
        state["status"] = "No findings yet — retrying research"
        return state

    try:
        # Build a condensed summary of findings for the prompt
        findings_summary = _build_findings_summary(findings)
        plan_text        = "\n".join(f"  - {q}" for q in plan)

        prompt = CRITIC_PROMPT.format(
            query           = query,
            plan            = plan_text,
            findings_count  = len(findings),
            findings_summary= findings_summary,
        )

        response = _client.models.generate_content(
            model   = GEMINI_MODEL,
            contents= prompt,
            config  = types.GenerateContentConfig(temperature=0.2),
        )
        raw_text   = response.text.strip()
        clean_text = _strip_code_fences(raw_text)

        logger.debug(f"Critic raw response: {raw_text[:300]}")

        result = json.loads(clean_text)

        # Validate response shape
        if not isinstance(result, dict):
            raise ValueError(f"Expected dict, got {type(result)}")

        satisfied = bool(result.get("satisfied", False))
        gaps      = result.get("gaps", [])

        # Ensure gaps is a clean list of strings
        if not isinstance(gaps, list):
            gaps = []
        gaps = [str(g).strip() for g in gaps if str(g).strip()]

        if satisfied or not gaps:
            logger.info("Critic: research is SUFFICIENT — proceeding to Writer")
            state["gaps"]   = []
            state["status"] = f"Research complete after {state['iterations']} iteration(s)"
        else:
            logger.info(f"Critic: found {len(gaps)} gap(s) — looping back to Researcher")
            for g in gaps:
                logger.info(f"  Gap: {g}")
            state["gaps"]   = gaps
            state["status"] = f"Gaps found — researching {len(gaps)} more topic(s)"

        state["error"] = None
        return state

    except json.JSONDecodeError as e:
        logger.error(f"Critic JSON parse error: {e}")
        return _fallback_satisfied(state, f"JSON parse error: {e}")

    except Exception as e:
        logger.error(f"Critic failed: {e}")
        return _fallback_satisfied(state, str(e))


def _build_findings_summary(findings: List[dict]) -> str:
    """
    Build a short, readable summary of findings for the Critic prompt.

    Each finding gets one line with its sub-question and a content preview.
    Keeps the prompt small — critical for low-end machine performance.
    """
    lines = []
    for i, f in enumerate(findings, 1):
        sub_q   = f.get("sub_question", "unknown")
        content = f.get("content", "")
        url     = f.get("source_url", "")
        preview = content[:FINDING_PREVIEW_CHARS].replace("\n", " ")
        lines.append(
            f"[{i}] Sub-question: {sub_q}\n"
            f"     Source : {url}\n"
            f"     Preview: {preview}..."
        )
    return "\n\n".join(lines)


def _strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers from Gemini output."""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$",          "", text, flags=re.MULTILINE)
    return text.strip()


def _fallback_satisfied(state: ResearchState, error_msg: str) -> ResearchState:
    """
    If Gemini fails, treat research as satisfied and proceed to Writer.
    Better to write a report with what we have than to loop infinitely.
    """
    logger.warning(f"Critic using fallback (satisfied=True) due to: {error_msg}")
    state["gaps"]   = []
    state["status"] = "Critic error — proceeding to report (fallback)"
    state["error"]  = f"Critic warning: {error_msg}"
    return state