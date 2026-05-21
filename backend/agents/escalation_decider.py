"""
agents/escalation_decider.py — Escalation Decider Agent

Escalation rules:
  1. intent == "escalate"                           
  2. score < -0.7 AND frustration_turns >= 2        
  3. attempt_count >= 2 AND kb_confidence < 0.3     
  4. kb_confidence < 0.2 AND action failed/missing 
  5. complaint AND angry AND message_count > 1 
"""

from typing import Any
from backend.graph.state import SupportState
from backend.config import get_settings
from backend.escalation.ticket_manager import create_ticket

settings = get_settings()

ANGER_THRESHOLD       = -0.7
MIN_FRUSTRATION_TURNS = 2
LOW_CONFIDENCE_HARD   = 0.2
LOW_CONFIDENCE_SOFT   = 0.3


def escalation_decider_node(state: SupportState) -> dict[str, Any]:
    sentiment_score   = state.get("sentiment_score", 0.0)
    frustration_turns = state.get("frustration_turns", 0)
    attempt_count     = state.get("attempt_count", 0)
    kb_confidence     = state.get("kb_confidence", 0.0)
    intent            = state.get("intent", "other")
    sentiment         = state.get("sentiment", "neutral")
    session_id        = state.get("session_id", "unknown")
    customer_name     = state.get("customer_name", "Customer")
    messages          = state.get("messages", [])
    action_taken      = state.get("action_taken", "")
    action_result     = state.get("action_result", {})
    message_count     = len(messages)

    action_failed = (
        action_taken != "" and
        action_taken != "requested_order_id" and
        not action_result.get("success", True)
    )

    
    ai_cant_resolve = (
        kb_confidence < LOW_CONFIDENCE_SOFT and
        (action_failed or action_taken == "")
    )

    should_escalate   = False
    escalation_reason = ""

    # Rule 1 — user explicitly asked for human
    if intent == "escalate":
        should_escalate   = True
        escalation_reason = "user_requested_human"

    # Rule 2 —  high frustration (even if sentiment isn't fully "angry", the repeated frustration is a strong signal)
    elif sentiment_score < ANGER_THRESHOLD and frustration_turns >= MIN_FRUSTRATION_TURNS:
        should_escalate   = True
        escalation_reason = "high_frustration"

    # Rule 3 — no root cause
    elif attempt_count >= settings.max_failed_attempts and kb_confidence < LOW_CONFIDENCE_SOFT:
        should_escalate   = True
        escalation_reason = "ai_cannot_resolve"

    # Rule 4 — (low confidence + action failed)
    elif ai_cant_resolve and message_count > 1:
        should_escalate   = True
        escalation_reason = "ai_cannot_find_root_cause"

    # Rule 5   — angry complaint after multiple attempts
    elif intent == "complaint" and sentiment == "angry" and message_count > 1:
        should_escalate   = True
        escalation_reason = "angry_complaint"

    # Create ticket if escalating
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
        print(f"[EscalationDecider] ESCALATING — Reason: {escalation_reason} | Ticket: {escalation_ticket_id}")
    else:
        print(f"[EscalationDecider] No escalation. Confidence: {kb_confidence:.2f} | AI can resolve: {not ai_cant_resolve} | Messages: {message_count}")

    return {
        "should_escalate":      should_escalate,
        "escalation_reason":    escalation_reason,
        "escalation_ticket_id": escalation_ticket_id,
    }