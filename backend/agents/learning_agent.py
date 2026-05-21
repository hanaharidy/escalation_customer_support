"""
agents/learning_agent.py — Learning Agent

Responsibilities:
- Log successful resolutions to ChromaDB
- This is the feedback loop: human agent answers → stored in KB
- Future similar questions will retrieve this resolution

This node runs at the END of every successful (non-escalated) conversation turn.
It only writes to the KB when a ticket is explicitly resolved by a human agent
via the /resolve-ticket REST endpoint.

The node itself just marks resolution_logged=True to track state.
The actual KB write happens in the REST endpoint via add_resolved_ticket().
"""

from typing import Any
from backend.graph.state import SupportState


# ── Agent Node Function ────────────────────────────────────────────────────────

def learning_agent_node(state: SupportState) -> dict[str, Any]:
    """
    LangGraph node: logs the current resolution for future learning.

    In this MVP, the learning happens when:
    1. A human agent resolves an escalated ticket (via REST endpoint)
    2. A customer marks their issue as resolved (via Streamlit "Resolved" button)

    This node simply marks the turn as logged.
    The actual ChromaDB write is handled by add_resolved_ticket() in ingestion.py.

    Reads:  state["response"], state["current_input"], state["kb_confidence"]
    Writes: state["resolution_logged"]
    """
    response = state.get("response", "")
    current_input = state.get("current_input", "")
    kb_confidence = state.get("kb_confidence", 0.0)

    # Only log if we actually produced a response
    if response and current_input:
        print(f"[LearningAgent] Turn logged | KB confidence: {kb_confidence}")
        resolution_logged = True
    else:
        resolution_logged = False

    return {"resolution_logged": resolution_logged}