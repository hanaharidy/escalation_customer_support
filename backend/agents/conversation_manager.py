"""
agents/conversation_manager.py - Conversation Manager Agent

Classifies intent and extracts entities.
Special handling for escalate intent:
  - First time: convert to clarification attempt
  - Second time: keep as escalate (user insists)
"""

from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.state import SupportState
from backend.llm.client import get_llm_with_fallback


INTENT_SYSTEM_PROMPT = """You are an intent classifier for an e-commerce customer support system.

Analyze the customer message and return a JSON object with:
1. "intent": one of these exact values:
   - "order_tracking"  → asking about order status, delivery, tracking
   - "refund"          → wants refund, money back, charge dispute
   - "product_issue"   → wrong item, damaged item, product not working
   - "complaint"       → expressing frustration, dissatisfaction
   - "escalate"        → explicitly asks for human agent, manager, supervisor
   - "other"           → anything else

2. "entities": extract if present:
   - "order_id": any order number (e.g. "ORD-123", "order 456")
   - "product_name": any product mentioned
   - "reason": reason for refund or complaint

Return ONLY valid JSON. No explanation. No markdown.
Example: {"intent": "order_tracking", "entities": {"order_id": "ORD-789"}}
"""


def conversation_manager_node(state: SupportState) -> dict[str, Any]:
    import json

    user_input       = state["current_input"]
    escalate_count   = state.get("escalate_requested_count", 0)
    updated_messages = list(state.get("messages", []))
    updated_messages.append(HumanMessage(content=user_input))

    llm = get_llm_with_fallback(temperature=0.1)

    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f"Customer message: {user_input}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed   = json.loads(raw)
        intent   = parsed.get("intent", "other")
        entities = parsed.get("entities", {})

    except Exception as e:
        print(f"[ConversationManager] Intent parsing failed: {e}")
        intent   = "other"
        entities = {}

    # Special handling for escalate intent
    # First time: try to help first
    # Second time: user insists, respect their request
    if intent == "escalate":
        if escalate_count == 0:
            print(f"[ConversationManager] Escalate requested (first time) — attempting to help first.")
            return {
                "messages":               updated_messages,
                "intent":                 "escalate_attempted",
                "entities":               entities,
                "escalate_requested_count": 1,
                "response": (
                    "I understand you'd like to speak with a human agent. "
                    "Before I connect you, let me see if I can resolve this for you directly. "
                    "Could you please tell me what issue you're experiencing?"
                ),
            }
        else:
            print(f"[ConversationManager] Escalate requested (second time) — escalating.")
            return {
                "messages":               updated_messages,
                "intent":                 "escalate",
                "entities":               entities,
                "escalate_requested_count": escalate_count + 1,
            }

    print(f"[ConversationManager] Intent: {intent} | Entities: {entities}")

    return {
        "messages":               updated_messages,
        "intent":                 intent,
        "entities":               entities,
        "escalate_requested_count": escalate_count,
    }