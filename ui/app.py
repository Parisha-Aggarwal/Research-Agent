"""
ui/app.py — Streamlit frontend for the Research Assistant.

Displays:
  - Query input form
  - Live status updates during research
  - Reasoning workflow steps (plan, findings count, iterations)
  - Final report rendered as markdown
  - Session history sidebar with full click-to-view support

Run with:
    streamlit run ui/app.py

The UI calls the FastAPI backend at localhost:8000.
Start the backend first: uvicorn api.routes:app --port 8000
"""

import time
import requests
import streamlit as st

# ── Page config — must be first Streamlit call ────────────────────────
st.set_page_config(
    page_title = "AI Research Assistant",
    page_icon  = "🔬",
    layout     = "wide",
)

# ── Constants ─────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"


# ── Helper functions ──────────────────────────────────────────────────

def call_research_api(query: str) -> dict:
    """
    POST /research to the FastAPI backend.
    Returns the response dict or an error dict.
    """
    try:
        response = requests.post(
            f"{API_BASE}/research",
            json    = {"query": query},
            timeout = 180,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Is the backend running?\n\n"
                         "Start it with: `uvicorn api.routes:app --port 8000`"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out after 3 minutes. Try a simpler query."}
    except Exception as e:
        return {"error": str(e)}


def fetch_history() -> list:
    """GET /history from the FastAPI backend — now includes final_report."""
    try:
        response = requests.get(f"{API_BASE}/history?limit=10", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return []


def render_workflow_steps(result: dict) -> None:
    """
    Display the reasoning workflow — what the agents did.
    Shows as an expandable section above the report.
    """
    with st.expander("🧠 Reasoning Workflow", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Iterations",         result.get("iterations", 0))
        col2.metric("Findings Collected", result.get("findings_count", 0))
        col3.metric("Status", "✅ Complete" if not result.get("error") else "⚠️ Warning")

        if result.get("error"):
            st.warning(f"**Note:** {result['error']}")


def render_report_view(session: dict) -> None:
    """
    Render a full report in the main area — used for both new results
    and past sessions loaded from history.

    Args:
        session: Dict with keys: query, final_report, iterations,
                 findings_count, created_at (optional), error (optional).
    """
    # Badge showing if this is a past session
    if session.get("created_at"):
        st.caption(f"🕒 Session from {session['created_at'][:19]}")

    # Workflow metadata
    render_workflow_steps(session)

    # Report content
    st.subheader("📋 Research Report")
    report = session.get("final_report") or "No report content available."
    st.markdown(report)

    # Download button
    safe_query = session.get("query","report")[:30].replace(" ","_")
    st.download_button(
        label     = "📥 Download Report (.md)",
        data      = report,
        file_name = f"research_{safe_query}_{int(time.time())}.md",
        mime      = "text/markdown",
    )


def render_sidebar() -> None:
    """
    Render session history in the sidebar.

    Each session shows metadata and a 'View Report' button.
    Clicking the button stores that session in st.session_state
    so the main area can render it.
    """
    with st.sidebar:
        st.title("📚 Research History")
        st.caption("Click a session to view its full report")

        if st.button("🔄 Refresh", use_container_width=True):
            # Clear any active past session so the main area resets
            st.session_state.pop("active_past_session", None)
            st.rerun()

        st.divider()

        history = fetch_history()

        if not history:
            st.info("No past sessions yet.\nRun your first research query!")
            return

        for session in history:
            # Truncate long queries for the sidebar label
            label = session["query"]
            short_label = label[:40] + "..." if len(label) > 40 else label

            with st.expander(f"🔍 {short_label}", expanded=False):
                # Metadata row
                st.caption(f"📅 {session['created_at'][:19]}")
                col1, col2 = st.columns(2)
                col1.caption(f"🔁 {session['iterations']} iteration(s)")
                col2.caption(f"📄 {session['findings_count']} findings")

                # The key fix: a button that loads this session into main area
                if st.button(
                    "📖 View Report",
                    key              = f"view_{session['id']}",
                    use_container_width = True,
                ):
                    # Store in session_state — main() reads this to render
                    st.session_state["active_past_session"] = {
                        "query":          session["query"],
                        "final_report":   session.get("final_report"),
                        "iterations":     session["iterations"],
                        "findings_count": session["findings_count"],
                        "created_at":     session["created_at"],
                        "error":          None,
                    }
                    # Clear any active new result so they don't overlap
                    st.session_state.pop("active_new_result", None)
                    st.rerun()


# ── Main UI ───────────────────────────────────────────────────────────

def main():
    # Initialise session state keys on first load
    if "active_past_session" not in st.session_state:
        st.session_state["active_past_session"] = None
    if "active_new_result" not in st.session_state:
        st.session_state["active_new_result"] = None

    # Header
    st.title("🔬 Agentic AI Research Assistant")
    st.caption(
        "Powered by LangGraph · Gemini · ChromaDB · DuckDuckGo  "
        "| Planner → Researcher → Critic → Writer"
    )
    st.divider()

    # Sidebar (always rendered)
    render_sidebar()

    # ── If user clicked "View Report" in sidebar — show that session ──
    if st.session_state["active_past_session"]:
        past = st.session_state["active_past_session"]

        st.subheader(f"📂 Past Session")
        st.info(f"**Query:** {past['query']}")

        # Button to go back to the new-query view
        if st.button("← Back to New Research"):
            st.session_state["active_past_session"] = None
            st.rerun()

        st.divider()
        render_report_view(past)
        return   # don't render the query form below

    # ── Otherwise show the normal query form ─────────────────────────
    st.subheader("📝 New Research Query")

    query = st.text_area(
        label       = "Enter your research question:",
        placeholder = "e.g. What are the latest advancements in quantum computing?",
        height      = 100,
        key         = "query_input",
    )

    # Example query buttons
    st.caption("💡 Try an example:")
    col1, col2, col3 = st.columns(3)
    if col1.button("🧬 CRISPR gene editing"):
        st.session_state["query_input"] = "How does CRISPR gene editing work and what are its latest applications?"
        st.rerun()
    if col2.button("⚛️ Quantum computing"):
        st.session_state["query_input"] = "What are the latest breakthroughs in quantum computing in 2024?"
        st.rerun()
    if col3.button("🤖 LLM architectures"):
        st.session_state["query_input"] = "How do large language model architectures like transformers work?"
        st.rerun()

    st.divider()

    run_clicked = st.button(
        "🚀 Start Research",
        type     = "primary",
        disabled = not bool(query and query.strip()),
    )

    # ── If a new result is already in state, render it ────────────────
    # (persists across Streamlit reruns caused by sidebar interactions)
    if st.session_state["active_new_result"] and not run_clicked:
        result = st.session_state["active_new_result"]
        st.success("✅ Research complete!")
        render_report_view(result)

    # ── Run new research ──────────────────────────────────────────────
    if run_clicked and query.strip():
        # Clear any previously loaded past session
        st.session_state["active_past_session"] = None
        st.session_state["active_new_result"]   = None

        with st.spinner("Research in progress — this takes 30–90 seconds..."):
            progress_bar = st.progress(0)
            step_display = st.empty()

            steps = [
                ("🗺️",  "Planner",    "Breaking query into sub-questions..."),
                ("🔍",  "Researcher", "Searching the web and scraping pages..."),
                ("🧐",  "Critic",     "Reviewing findings for gaps..."),
                ("✍️",  "Writer",     "Generating final report..."),
            ]

            for i, (icon, name, desc) in enumerate(steps):
                step_display.markdown(f"**{icon} {name} Agent** — {desc}")
                progress_bar.progress((i + 1) * 20)
                time.sleep(0.4)

            step_display.markdown("⏳ **Agents working...** (API call in progress)")
            progress_bar.progress(60)

            result = call_research_api(query.strip())

            progress_bar.progress(100)
            step_display.empty()
            progress_bar.empty()

        if "error" in result and not result.get("final_report"):
            st.error(f"❌ Research failed\n\n{result['error']}")
        else:
            # Store result in session_state so it survives reruns
            st.session_state["active_new_result"] = {
                "query":          query.strip(),
                "final_report":   result.get("final_report"),
                "iterations":     result.get("iterations", 0),
                "findings_count": result.get("findings_count", 0),
                "created_at":     None,
                "error":          result.get("error"),
            }
            st.rerun()


if __name__ == "__main__":
    main()