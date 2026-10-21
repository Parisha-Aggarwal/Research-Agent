"""
graph/workflow.py — LangGraph state machine wiring.

This file connects all four agents into a directed graph with a
conditional loop. It's the "nervous system" of the whole project.

Graph structure:
    START
      ↓
    planner          (always runs once)
      ↓
    researcher       (runs on first pass and after critic finds gaps)
      ↓
    critic           (reviews findings, decides: loop or finish)
      ↓ (conditional)
    ├── gaps found AND iterations < MAX → back to researcher
    └── satisfied OR max iterations hit → writer
      ↓
    writer           (generates final report)
      ↓
    END

The conditional edge is the key mechanism — it reads state["gaps"]
after the critic runs and routes accordingly.
"""

from langgraph.graph import StateGraph, START, END

from graph.state import ResearchState
from agents.planner    import run_planner
from agents.researcher import run_researcher
from agents.critic     import run_critic
from agents.writer     import run_writer
from config import MAX_ITERATIONS


def _should_continue(state: ResearchState) -> str:
    """
    Conditional edge function — called after the Critic node runs.

    LangGraph calls this with the current state and uses the return
    value to decide which node to go to next.

    Returns:
        "researcher"  → loop back, more research needed
        "writer"      → proceed, research is sufficient
    """
    gaps       = state.get("gaps", [])
    iterations = state.get("iterations", 0)

    # Two reasons to stop looping:
    #   1. Critic is satisfied (no gaps)
    #   2. We've hit the safety limit
    if not gaps or iterations >= MAX_ITERATIONS:
        return "writer"

    return "researcher"


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph research workflow.

    Returns a compiled graph ready to call with .invoke(initial_state).

    Why compile?
        LangGraph's .compile() validates the graph structure (no orphan
        nodes, all edges connected) and returns an optimised runnable.
        It's cheap to call once at startup.
    """
    # Create the graph with our shared state schema
    graph = StateGraph(ResearchState)

    # ── Add nodes (one per agent) ─────────────────────────────────────
    # Each node is just a Python function: (state) → state
    graph.add_node("planner",    run_planner)
    graph.add_node("researcher", run_researcher)
    graph.add_node("critic",     run_critic)
    graph.add_node("writer",     run_writer)

    # ── Add edges (define the flow) ───────────────────────────────────
    # Fixed edges: always go from A → B
    graph.add_edge(START,        "planner")
    graph.add_edge("planner",    "researcher")
    graph.add_edge("researcher", "critic")

    # Conditional edge: critic → (researcher OR writer) based on gaps
    graph.add_conditional_edges(
        source    = "critic",          # after this node runs...
        path      = _should_continue,  # call this function to decide...
        path_map  = {                  # map return values to node names
            "researcher": "researcher",
            "writer":     "writer",
        },
    )

    # Writer is always the final node
    graph.add_edge("writer", END)

    # Compile validates structure and returns the runnable graph
    compiled = graph.compile()
    return compiled


# Module-level compiled graph — built once, reused by main.py and FastAPI
compiled_graph = build_graph()