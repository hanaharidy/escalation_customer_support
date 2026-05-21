"""
agents/action_agent.py — Action Agent

Responsibilities:
- Decide if an action (tool call) is needed based on intent + entities
- Call the appropriate mock tool (order lookup, refund, return check)
- Format the result into a human-readable response

This agent only runs when escalation is NOT triggered AND
the intent maps to an available action tool.
"""

from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.state import SupportState
from backend.llm.client import get_llm_with_fallback
from backend.actions.order_actions import (
    lookup_order,
    process_refund,
    check_return_eligibility,
    ACTION_TOOLS,
)


# ── Agent Node Function ────────────────────────────────────────────────────────

def action_agent_node(state: SupportState) -> dict[str, Any]:
    """
    LangGraph node: executes action tools based on intent and entities.

    Reads:  state["intent"], state["entities"], state["kb_answer"]
    Writes: state["action_result"], state["action_taken"], state["response"]
    """
    intent = state.get("intent", "other")
    entities = state.get("entities", {})
    kb_answer = state.get("kb_answer", "")
    order_id = entities.get("order_id", "")

    # Check if this intent has an associated action tool
    if intent not in ACTION_TOOLS:
        # No action needed — just use the KB answer as the response
        print(f"[ActionAgent] No action tool for intent '{intent}'. Using KB answer.")
        return {
            "action_result": {},
            "action_taken": "",
            "response": kb_answer,
        }

    # Execute the appropriate tool
    action_result = {}
    action_taken = ""

    try:
        if intent == "order_tracking":
            if not order_id:
                # Ask for order ID if missing
                return {
                    "action_result": {},
                    "action_taken": "requested_order_id",
                    "response": "I'd be happy to check your order status. Could you please provide your order ID? It usually looks like ORD-12345.",
                }
            action_result = lookup_order(order_id)
            action_taken = "order_lookup"
            print(f"[ActionAgent] Looked up order: {order_id}")

        elif intent == "refund":
            if not order_id:
                return {
                    "action_result": {},
                    "action_taken": "requested_order_id",
                    "response": "I can help with your refund. Could you please provide your order ID?",
                }
            reason = entities.get("reason", "Customer requested refund")
            action_result = process_refund(order_id, reason)
            action_taken = "refund_processed"
            print(f"[ActionAgent] Processed refund for order: {order_id}")

        elif intent == "product_issue":
            if not order_id:
                return {
                    "action_result": {},
                    "action_taken": "requested_order_id",
                    "response": "I'm sorry to hear about the issue with your product. Could you provide your order ID so I can check your return eligibility?",
                }
            action_result = check_return_eligibility(order_id)
            action_taken = "return_eligibility_checked"
            print(f"[ActionAgent] Checked return eligibility for order: {order_id}")

    except Exception as e:
        print(f"[ActionAgent] Tool execution failed: {e}")
        return {
            "action_result": {},
            "action_taken": "failed",
            "response": kb_answer or "I encountered an issue processing your request. Let me connect you with a human agent.",
        }

    # Use LLM to format the action result into a natural response
    response = _format_action_response(
        intent=intent,
        action_result=action_result,
        kb_answer=kb_answer,
        original_question=state.get("current_input", ""),
    )

    return {
        "action_result": action_result,
        "action_taken": action_taken,
        "response": response,
    }


def _format_action_response(
    intent: str,
    action_result: dict,
    kb_answer: str,
    original_question: str,
) -> str:
    """
    Uses LLM to turn raw action result data into a friendly response.
    Falls back to a simple formatted string if LLM fails.
    """
    if not action_result.get("success"):
        error = action_result.get("error", "Unknown error")
        return f"I wasn't able to process that request: {error}. Please try again or contact support."

    FORMAT_PROMPT = """You are a helpful e-commerce support agent.
A customer asked a question and we retrieved data from our system.
Write a clear, friendly response that directly answers their question using the data provided.
Be concise. Use the actual values from the data."""

    llm = get_llm_with_fallback(temperature=0.3)

    messages = [
        SystemMessage(content=FORMAT_PROMPT),
        HumanMessage(content=f"""Customer question: {original_question}

System data retrieved:
{action_result}

Additional context from knowledge base:
{kb_answer}

Write a helpful response to the customer."""),
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"[ActionAgent] Response formatting failed: {e}")
        # Fallback: return raw data as readable string
        return f"Here's what I found: {action_result.get('message', str(action_result))}"