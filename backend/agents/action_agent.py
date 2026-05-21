"""
agents/action_agent.py - Action Agent with multi-level resolution strategy.

Resolution levels before escalating to human:
  Level 1: KB answer (handled by knowledge_base_agent)
  Level 2: Tool execution (order lookup, refund, return check)
  Level 3: Clarification request (ask user for more details)
  Level 4: Alternative solutions (suggest other channels/options)
  Level 5: Human escalation (last resort only)
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


def action_agent_node(state: SupportState) -> dict[str, Any]:
    intent               = state.get("intent", "other")
    entities             = state.get("entities", {})
    kb_answer            = state.get("kb_answer", "")
    kb_confidence        = state.get("kb_confidence", 0.0)
    order_id             = entities.get("order_id", "")
    clarification_asked  = state.get("clarification_asked", False)
    alternatives_suggested = state.get("alternatives_suggested", False)
    current_input        = state.get("current_input", "")
    messages             = state.get("messages", [])
    resolution_level     = state.get("resolution_level", 1)

    # ── Level 2: Tool Execution ───────────────────────────────────────────
    if intent in ACTION_TOOLS:
        if not order_id:
            return {
                "action_result":    {},
                "action_taken":     "requested_order_id",
                "resolution_level": 2,
                "clarification_asked": True,
                "response": _ask_for_order_id(intent),
            }

        try:
            if intent == "order_tracking":
                action_result = lookup_order(order_id)
                action_taken  = "order_lookup"

            elif intent == "refund":
                reason        = entities.get("reason", "Customer requested refund")
                action_result = process_refund(order_id, reason)
                action_taken  = "refund_processed"

            elif intent == "product_issue":
                action_result = check_return_eligibility(order_id)
                action_taken  = "return_eligibility_checked"

            else:
                action_result = {}
                action_taken  = ""

            if action_result.get("success"):
                response = _format_action_response(
                    intent=intent,
                    action_result=action_result,
                    kb_answer=kb_answer,
                    original_question=current_input,
                )
                return {
                    "action_result":    action_result,
                    "action_taken":     action_taken,
                    "resolution_level": 2,
                    "response":         response,
                }

        except Exception as e:
            print(f"[ActionAgent] Tool execution failed: {e}")

    # ── Level 3: Clarification ────────────────────────────────────────────
    if not clarification_asked and kb_confidence < 0.4:
        clarification = _ask_clarification(current_input, intent, messages)
        return {
            "action_result":       {},
            "action_taken":        "clarification_requested",
            "resolution_level":    3,
            "clarification_asked": True,
            "response":            clarification,
        }

    # ── Level 4: Alternative Solutions ───────────────────────────────────
    if not alternatives_suggested:
        alternatives = _suggest_alternatives(current_input, intent, kb_answer)
        return {
            "action_result":          {},
            "action_taken":           "alternatives_suggested",
            "resolution_level":       4,
            "alternatives_suggested": True,
            "response":               alternatives,
        }

    # ── Level 5: Signal for human escalation ─────────────────────────────
    # Bot tried everything — escalation_decider will handle the actual escalation
    return {
        "action_result":    {},
        "action_taken":     "all_options_exhausted",
        "resolution_level": 5,
        "response":         kb_answer or "",
    }


def _ask_for_order_id(intent: str) -> str:
    messages = {
        "order_tracking": "I'd be happy to check your order status. Could you please provide your order ID? It usually looks like ORD-12345.",
        "refund":         "I can help process your refund. Could you please provide your order ID so I can look into this for you?",
        "product_issue":  "I'm sorry to hear about the issue. Could you provide your order ID so I can check your return eligibility?",
    }
    return messages.get(intent, "Could you please provide your order ID?")


def _ask_clarification(user_input: str, intent: str, messages: list) -> str:
    llm = get_llm_with_fallback(temperature=0.3)

    conversation = "\n".join([
        f"{'Customer' if m.__class__.__name__ == 'HumanMessage' else 'Agent'}: {m.content}"
        for m in messages[-4:]
    ])

    prompt = f"""You are a helpful e-commerce support agent.

The customer's issue is not entirely clear yet. Based on their message and conversation history,
ask ONE specific clarifying question to better understand their problem.

Be empathetic, specific, and keep it to one question only.

Conversation so far:
{conversation}

Current message: {user_input}
Detected intent: {intent}

Ask a clarifying question:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return "Could you please provide more details about your issue so I can better assist you?"


def _suggest_alternatives(user_input: str, intent: str, kb_answer: str) -> str:
    llm = get_llm_with_fallback(temperature=0.3)

    prompt = f"""You are a helpful e-commerce support agent.

You have tried to resolve the customer's issue but need to suggest alternative ways to help them.

Suggest 2-3 of these communication channels and self-service options as relevant:
- Check our FAQ page at help.store.com
- Email us at support@store.com (response within 24 hours)
- Live chat available Mon-Fri 9AM-6PM EST
- Check order status directly at store.com/orders
- Visit our returns portal at store.com/returns
- Call us at 1-800-SUPPORT (Mon-Fri 9AM-8PM EST)

Also include any relevant information from the knowledge base if available.

Customer issue: {user_input}
Intent: {intent}
KB context: {kb_answer[:300] if kb_answer else 'Not available'}

Provide helpful alternatives in a friendly, apologetic tone:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return (
            "I want to make sure you get the help you need. Here are some ways to reach us:\n\n"
            "- Email: support@store.com (24-hour response)\n"
            "- Live Chat: Available Mon-Fri 9AM-6PM\n"
            "- Phone: 1-800-SUPPORT\n"
            "- Self-service: store.com/orders\n\n"
            "Would any of these work for you?"
        )


def _format_action_response(
    intent: str,
    action_result: dict,
    kb_answer: str,
    original_question: str,
) -> str:
    if not action_result.get("success"):
        return action_result.get("error", "I was unable to process that request.")

    llm = get_llm_with_fallback(temperature=0.3)

    prompt = f"""You are a helpful e-commerce support agent.
A customer asked a question and we retrieved data from our system.
Write a clear, friendly response using the actual values from the data.
Be concise and direct.

Customer question: {original_question}
System data: {action_result}
Additional KB context: {kb_answer[:200] if kb_answer else 'N/A'}

Write the response:"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return action_result.get("message", str(action_result))