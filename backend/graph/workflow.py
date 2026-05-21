"""
graph/workflow.py — LangGraph workflow assembly.

Correct flow per supervisor spec:
  1. Conversation Manager  — classify intent
  2. Knowledge Base Agent  — search for answer
  3. Action Agent          — handle transactional requests
  4. Sentiment Analyzer    — monitor tone (after attempt)
  5. Escalation Decider    — escalate ONLY if genuinely needed
  6. Learning Agent        — log resolution

Escalation is the LAST resort, not the first response.
"""

from langgraph.graph import StateGraph, END

from backend.graph.state import SupportState
from backend.agents.conversation_manager import conversation_manager_node
from backend.agents.sentiment_analyzer import sentiment_analyzer_node
from backend.agents.knowledge_base_agent import knowledge_base_agent_node
from backend.agents.action_agent import action_agent_node
from backend.agents.escalation_decider import escalation_decider_node
from backend.agents.learning_agent import learning_agent_node


# ── Escalation Handler ────────────────────────────────────────────────────────

def escalation_handler_node(state: SupportState) -> dict:
    """Composes the human-handoff message shown to the customer."""
    reason = state.get("escalation_reason", "")
    ticket_id = state.get("escalation_ticket_id", "")
    customer_name = state.get("customer_name", "there")

    reason_messages = {
        "high_frustration":           "I can see this has been a very frustrating experience, and I sincerely apologize we couldn't resolve it for you.",
        "persistent_negative":        "I understand we haven't been able to resolve your issue, and I'm sorry for the inconvenience.",
        "repeated_failed_attempts":   "I've tried my best but wasn't able to fully resolve your issue through our automated system.",
        "user_requested_human":       "Of course! I'll connect you with a human agent right away.",
        "angry_complaint":            "I'm truly sorry about this experience. You deserve much better service than this.",
        "low_kb_confidence":          "Your question requires specialist knowledge that's beyond what I can confidently answer.",
    }

    reason_msg = reason_messages.get(reason, "Let me connect you with a specialist who can better help you.")

    response = (
        f"{reason_msg}\n\n"
        f"A human support agent will have the full context of our conversation. "
        f"Your support ticket **{ticket_id}** has been created.\n\n"
        f"⏱ Expected wait time: **2–5 minutes**. Thank you for your patience, {customer_name}."
    )

    return {"response": response}


# ── Conditional Edge: after escalation check ─────────────────────────────────

def route_after_escalation_check(state: SupportState) -> str:
    if state.get("should_escalate", False):
        return "escalation_handler"
    return "learning_agent"


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(SupportState)

    # Add nodes
    graph.add_node("conversation_manager",  conversation_manager_node)
    graph.add_node("knowledge_base_agent",  knowledge_base_agent_node)
    graph.add_node("action_agent",          action_agent_node)
    graph.add_node("sentiment_analyzer",    sentiment_analyzer_node)
    graph.add_node("escalation_decider",    escalation_decider_node)
    graph.add_node("escalation_handler",    escalation_handler_node)
    graph.add_node("learning_agent",        learning_agent_node)

    # Entry point
    graph.set_entry_point("conversation_manager")

    # Step 1 → 2: intent classified → search KB
    graph.add_edge("conversation_manager", "knowledge_base_agent")

    # Step 2 → 3: KB answer ready → run action if needed
    graph.add_edge("knowledge_base_agent", "action_agent")

    # Step 3 → 4: after action attempt → analyze sentiment
    graph.add_edge("action_agent", "sentiment_analyzer")

    # Step 4 → 5: sentiment scored → decide escalation
    graph.add_edge("sentiment_analyzer", "escalation_decider")

    # Step 5: escalate or finish normally
    graph.add_conditional_edges(
        "escalation_decider",
        route_after_escalation_check,
        {
            "escalation_handler": "escalation_handler",
            "learning_agent":     "learning_agent",
        },
    )

    # Escalation ends immediately
    graph.add_edge("escalation_handler", END)

    # Normal path ends after learning
    graph.add_edge("learning_agent", END)

    return graph.compile()


_graph = None

def get_graph():
    global _graph
    if _graph is None:
        print("[Graph] Building LangGraph workflow...")
        _graph = build_graph()
        print("[Graph] Workflow ready.")
    return _graph