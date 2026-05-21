"""
graph/workflow.py - LangGraph workflow assembly.

Flow:
  conversation_manager
    ├── intent == "escalate_attempted" → END directly (response already set)
    └── everything else →
        knowledge_base_agent
        → action_agent
        → sentiment_analyzer
        → escalation_decider
        → escalation_handler OR learning_agent
"""

from langgraph.graph import StateGraph, END

from backend.graph.state import SupportState
from backend.agents.conversation_manager import conversation_manager_node
from backend.agents.sentiment_analyzer import sentiment_analyzer_node
from backend.agents.knowledge_base_agent import knowledge_base_agent_node
from backend.agents.action_agent import action_agent_node
from backend.agents.escalation_decider import escalation_decider_node
from backend.agents.learning_agent import learning_agent_node


def escalation_handler_node(state: SupportState) -> dict:
    reason        = state.get("escalation_reason", "")
    ticket_id     = state.get("escalation_ticket_id", "")
    customer_name = state.get("customer_name", "there")

    reason_messages = {
        "user_requested_human":           "Of course! I'll connect you with a human agent right away.",
        "all_resolution_levels_exhausted": "I've done my best but wasn't able to fully resolve your issue. Let me connect you with a specialist.",
        "persistent_high_frustration":    "I can see this has been very frustrating. I sincerely apologize and am connecting you with a human agent now.",
        "unresolved_angry_complaint":     "I'm truly sorry we haven't been able to resolve this. You deserve better service.",
    }

    reason_msg = reason_messages.get(reason, "Let me connect you with a specialist who can better help you.")

    response = (
        f"{reason_msg}\n\n"
        f"A human support agent will have the full context of our conversation. "
        f"Your support ticket **{ticket_id}** has been created.\n\n"
        f"Expected wait time: **2-5 minutes**. Thank you for your patience, {customer_name}."
    )

    return {"response": response}


def route_after_conversation_manager(state: SupportState) -> str:
    """
    If bot already set a response (escalate_attempted), go straight to END.
    Otherwise continue the full pipeline.
    """
    if state.get("intent") == "escalate_attempted":
        return "end_early"
    return "knowledge_base_agent"


def route_after_escalation_check(state: SupportState) -> str:
    if state.get("should_escalate", False):
        return "escalation_handler"
    return "learning_agent"


def build_graph() -> StateGraph:
    graph = StateGraph(SupportState)

    graph.add_node("conversation_manager",  conversation_manager_node)
    graph.add_node("knowledge_base_agent",  knowledge_base_agent_node)
    graph.add_node("action_agent",          action_agent_node)
    graph.add_node("sentiment_analyzer",    sentiment_analyzer_node)
    graph.add_node("escalation_decider",    escalation_decider_node)
    graph.add_node("escalation_handler",    escalation_handler_node)
    graph.add_node("learning_agent",        learning_agent_node)

    graph.set_entry_point("conversation_manager")

    # After conversation manager — check if we need to short-circuit
    graph.add_conditional_edges(
        "conversation_manager",
        route_after_conversation_manager,
        {
            "end_early":           END,
            "knowledge_base_agent": "knowledge_base_agent",
        },
    )

    graph.add_edge("knowledge_base_agent", "action_agent")
    graph.add_edge("action_agent",         "sentiment_analyzer")
    graph.add_edge("sentiment_analyzer",   "escalation_decider")

    graph.add_conditional_edges(
        "escalation_decider",
        route_after_escalation_check,
        {
            "escalation_handler": "escalation_handler",
            "learning_agent":     "learning_agent",
        },
    )

    graph.add_edge("escalation_handler", END)
    graph.add_edge("learning_agent",     END)

    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        print("[Graph] Building LangGraph workflow...")
        _graph = build_graph()
        print("[Graph] Workflow ready.")
    return _graph