"""
agents/escalation_decider.py - Escalation Decider Agent

Human escalation is the LAST resort.
The bot must exhaust all resolution levels first:

  Level 1: KB answer
  Level 2: Tool execution
  Level 3: Clarification
  Level 4: Alternative solutions
  Level 5: Human escalation

Escalation only triggers when:
  - User explicitly asks for human (always respected)
  - Bot reached level 5 (all options exhausted)
  - Genuine persistent anger across multiple turns
"""

from typing import Any
from backend.graph.state import SupportState
from backend.config import get_settings
from backend.escalation.ticket_manager import create_ticket

settings = get_settings()


def escalation_decider_node(state: SupportState) -> dict[str, Any]:
    sentiment_score        = state.get("sentiment_score", 0.0)
    frustration_turns      = state.get("frustration_turns", 0)
    intent                 = state.get("intent", "other")
    sentiment              = state.get("sentiment", "neutral")
    session_id             = state.get("session_id", "unknown")
    customer_name          = state.get("customer_name", "Customer")
    messages               = state.get("messages", [])
    resolution_level       = state.get("resolution_level", 1)
    action_taken           = state.get("action_taken", "")
    alternatives_suggested = state.get("alternatives_suggested", False)
    message_count          = len(messages)

    should_escalate   = False
    escalation_reason = ""

    # Rule 1: User explicitly asked for human — always respect this
    if intent == "escalate":
        should_escalate   = True
        escalation_reason = "user_requested_human"

    # Rule 2: Bot exhausted all levels (clarification + alternatives tried)
    elif (
        resolution_level >= 4 and
        alternatives_suggested and
        action_taken == "all_options_exhausted"
    ):
        should_escalate   = True
        escalation_reason = "all_resolution_levels_exhausted"

    # Rule 3: Genuine persistent anger — multiple turns of real anger
    elif sentiment_score < -0.7 and frustration_turns >= 3:
        should_escalate   = True
        escalation_reason = "persistent_high_frustration"

    # Rule 4: Angry complaint AFTER bot already tried alternatives
    elif (
        intent == "complaint" and
        sentiment == "angry" and
        alternatives_suggested and
        message_count > 2
    ):
        should_escalate   = True
        escalation_reason = "unresolved_angry_complaint"

    escalation_ticket_id = ""
    if should_escalate:
        ticket = create_ticket(
            session_id=session_id,
            customer_name=customer_name,
            reason=escalation_reason,
            conversation_history=messages,
            sentiment=sentiment,
            intent=intent,
        )
        escalation_ticket_id = ticket["ticket_id"]
        print(f"[EscalationDecider] ESCALATING — Reason: {escalation_reason} | Level: {resolution_level} | Ticket: {escalation_ticket_id}")
    else:
        print(f"[EscalationDecider] Continuing. Level: {resolution_level} | Sentiment: {sentiment} ({sentiment_score}) | Alternatives suggested: {alternatives_suggested}")

    return {
        "should_escalate":      should_escalate,
        "escalation_reason":    escalation_reason,
        "escalation_ticket_id": escalation_ticket_id,
    }