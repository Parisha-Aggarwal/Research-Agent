"""
graph/state.py — The shared state that flows through every agent.

LangGraph passes this TypedDict between nodes.
Each agent reads what it needs and writes its output back here.
Think of it as the "whiteboard" all agents share.
"""

from typing import TypedDict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Finding:
    """One piece of research: where it came from and what it says."""
    source_url: str
    content: str
    sub_question: str   # Which sub-question this finding answers


class ResearchState(TypedDict):
    """
    The complete state of a research session.

    Fields are populated progressively as agents run:
      - query       → set by the user before the graph starts
      - plan        → set by Planner agent
      - findings    → appended by Researcher agent (can loop multiple times)
      - gaps        → set by Critic agent each iteration
      - iterations  → incremented each time the Critic loops back
      - final_report→ set by Writer agent at the end
      - status      → human-readable label of the current step (for UI)
      - error       → any error message (so the graph doesn't crash silently)
    """
    query: str
    plan: List[str]                      # Sub-questions from Planner
    findings: List[dict]                 # Raw dicts (source_url, content, sub_question)
    gaps: List[str]                      # Missing areas from Critic
    iterations: int                      # Loop counter
    final_report: Optional[str]          # Markdown report from Writer
    status: str                          # e.g. "Planning...", "Searching..."
    error: Optional[str]                 # Error message if something fails