"""
agents/planner.py — Planner Agent.

Responsibility:
    Take the user's raw research query and break it into 3-5
    focused sub-questions. These sub-questions guide the Researcher
    agent on exactly what to search for.

LangGraph contract:
    - Input:  ResearchState with 'query' populated
    - Output: ResearchState with 'plan' populated (list of sub-questions)
              and 'status' updated

Uses the current google-genai SDK (google.genai), not the deprecated
google-generativeai package.
"""

import json
import logging
import re

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL
from graph.state import ResearchState

logger = logging.getLogger(__name__)

# Initialise the Gemini client once at module load time
_client = genai.Client(api_key=GEMINI_API_KEY)


# ── Prompt template ───────────────────────────────────────────────────
PLANNER_PROMPT = """You are a research planning assistant.

Your job is to break a broad research query into 3 to 5 focused sub-questions.
Each sub-question should target a specific, searchable aspect of the topic.

Rules:
- Return ONLY a valid JSON array of strings. No explanation, no markdown, no code fences.
- Each sub-question must be specific enough to search on the web.
- Cover different angles: definitions, mechanisms, examples, limitations, recent developments.
- Do not number the questions.

Example input:
"How does CRISPR gene editing work?"

Example output:
[
  "What is the CRISPR-Cas9 mechanism and how does it cut DNA?",
  "What is guide RNA and how does it direct CRISPR to the target gene?",
  "What are the main applications of CRISPR in medicine and agriculture?",
  "What are the off-target effects and limitations of CRISPR gene editing?",
  "What are the latest CRISPR breakthroughs in 2024?"
]

Now do the same for this query:
"{query}"
"""


def run_planner(state: ResearchState) -> ResearchState:
    """
    LangGraph node function — generates a research plan from the query.

    Args:
        state: Current ResearchState. Reads 'query'.

    Returns:
        Updated ResearchState with 'plan' and 'status' set.
        On any failure, sets 'error' and returns a fallback single-item
        plan so the graph continues rather than crashing.
    """
    query = state["query"]
    logger.info(f"Planner running for query: '{query[:60]}'")
    state["status"] = "Planning research..."

    try:
        prompt = PLANNER_PROMPT.format(query=query)

        # Current SDK: client.models.generate_content()
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                # Lower temperature = more predictable, structured output
                temperature=0.3,
            ),
        )
        raw_text = response.text.strip()
        logger.debug(f"Gemini raw response: {raw_text[:200]}")

        # Strip markdown code fences Gemini sometimes adds anyway
        clean_text = _strip_code_fences(raw_text)
        sub_questions = json.loads(clean_text)

        # Validate: must be a non-empty list
        if not isinstance(sub_questions, list) or len(sub_questions) == 0:
            raise ValueError(f"Expected non-empty list, got: {type(sub_questions)}")

        # Ensure every item is a non-empty string
        sub_questions = [str(q).strip() for q in sub_questions if str(q).strip()]

        # Cap at 5 to keep research sessions lightweight
        sub_questions = sub_questions[:5]

        logger.info(f"Plan created with {len(sub_questions)} sub-questions")
        for i, q in enumerate(sub_questions, 1):
            logger.info(f"  {i}. {q}")

        state["plan"]   = sub_questions
        state["status"] = f"Plan ready — {len(sub_questions)} sub-questions"
        state["error"]  = None
        return state

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Gemini JSON: {e}")
        return _fallback_plan(state, query, f"JSON parse error: {e}")

    except Exception as e:
        logger.error(f"Planner failed: {e}")
        return _fallback_plan(state, query, str(e))


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json...``` or ```...```) from text."""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$",          "", text, flags=re.MULTILINE)
    return text.strip()


def _fallback_plan(state: ResearchState, query: str, error_msg: str) -> ResearchState:
    """
    If Gemini fails, use the raw query as a single sub-question.
    Lets the graph continue rather than hard-crashing.
    """
    logger.warning(f"Using fallback plan due to: {error_msg}")
    state["plan"]   = [query]
    state["status"] = "Plan ready (fallback — 1 sub-question)"
    state["error"]  = f"Planner warning: {error_msg}"
    return state