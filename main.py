"""
main.py — Entry point for running the research assistant directly.

Usage:
    python main.py
    python main.py "your custom query here"

Runs the full LangGraph pipeline end-to-end and prints the report.
Use this to test the complete workflow before wiring up the API/UI.
"""

import sys
import logging
from graph.state import ResearchState
from graph.workflow import compiled_graph

# Show INFO logs when running directly so progress is visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_research(query: str) -> ResearchState:
    """
    Run the full research pipeline for a given query.

    Args:
        query: The research question to investigate.

    Returns:
        The final ResearchState containing the report and all metadata.
    """
    initial_state: ResearchState = {
        "query":        query,
        "plan":         [],
        "findings":     [],
        "gaps":         [],
        "iterations":   0,
        "final_report": None,
        "status":       "Starting...",
        "error":        None,
    }

    logger.info(f"Starting research: '{query}'")

    # LangGraph runs the full graph synchronously and returns the final state
    final_state = compiled_graph.invoke(initial_state)

    return final_state


def print_report(state: ResearchState) -> None:
    """Pretty-print the research results to the terminal."""
    print("\n" + "="*65)
    print("RESEARCH COMPLETE")
    print("="*65)
    print(f"Query      : {state['query']}")
    print(f"Iterations : {state['iterations']}")
    print(f"Findings   : {len(state['findings'])}")
    print(f"Status     : {state['status']}")

    if state.get("error"):
        print(f"Warning    : {state['error']}")

    print("\n" + "-"*65)
    print("FINAL REPORT")
    print("-"*65)
    print(state.get("final_report") or "No report generated.")
    print("="*65 + "\n")


if __name__ == "__main__":
    # Accept query from user input at runtime
    query = input("Enter your research query: ").strip()
    if not query:
        print("No query entered. Exiting.")
        sys.exit()

    result = run_research(query)
    print_report(result)