"""
agents/conversation_manager.py — Conversation Manager Agent

Responsibilities:
- Classify the user's intent (what do they want?)
- Extract entities (order IDs, product names, etc.)
- Append the new message to conversation history

This is the FIRST node in the LangGraph workflow.
It sets the intent and entities that all other agents use.
"""

from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from backend.graph.state import SupportState
from backend.llm.client import get_llm_with_fallback


# ── Prompt ────────────────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are an intent classifier for an e-commerce customer support system.

Your job is to analyze the customer's message and return a JSON object with:
1. "intent": one of these exact values:
   - "order_tracking"  → customer asking about order status, delivery, tracking
   - "refund"          → customer wants a refund, money back, charge dispute
   - "product_issue"   → wrong item, damaged item, product not working
   - "complaint"       → expressing frustration, dissatisfaction, bad experience
   - "escalate"        → customer explicitly asks for human agent, manager, supervisor
   - "other"           → anything else (general questions, greetings, etc.)

2. "entities": a dict of extracted values. Extract these if present:
   - "order_id": any order number mentioned (e.g. "ORD-123", "order 456")
   - "product_name": any product mentioned
   - "reason": reason for refund or complaint if stated

Return ONLY valid JSON. No explanation. No markdown. Example:
{"intent": "order_tracking", "entities": {"order_id": "ORD-789"}}
"""


# ── Agent Node Function ────────────────────────────────────────────────────────

def conversation_manager_node(state: SupportState) -> dict[str, Any]:
    """
    LangGraph node: classifies intent and extracts entities.

    Reads:  state["current_input"], state["messages"]
    Writes: state["intent"], state["entities"], state["messages"]
    """
    import json

    user_input = state["current_input"]

    # Append the new user message to conversation history
    updated_messages = list(state.get("messages", []))
    updated_messages.append(HumanMessage(content=user_input))

    # Call LLM for intent classification
    llm = get_llm_with_fallback(temperature=0.1)  # low temp for classification

    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=f"Customer message: {user_input}"),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown code fences if LLM adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        intent = parsed.get("intent", "other")
        entities = parsed.get("entities", {})

    except Exception as e:
        print(f"[ConversationManager] Intent parsing failed: {e}")
        intent = "other"
        entities = {}

    print(f"[ConversationManager] Intent: {intent} | Entities: {entities}")

    return {
        "messages": updated_messages,
        "intent": intent,
        "entities": entities,
    }