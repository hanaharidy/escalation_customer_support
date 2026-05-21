"""
graph/state.py — LangGraph shared state definition.
"""

from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage


class SupportState(TypedDict):
    # ── Session ───────────────────────────────────────────────────────────
    session_id: str
    customer_name: str
    messages: List[BaseMessage]
    current_input: str
    attempt_count: int

    # ── Intent ────────────────────────────────────────────────────────────
    intent: str
    entities: dict

    # ── Knowledge Base (RAG) ──────────────────────────────────────────────
    retrieved_docs: List[str]
    kb_answer: str
    kb_confidence: float
    kb_sources: List[str]       # which docs were retrieved e.g. ["faqs.md"]

    # ── Sentiment ─────────────────────────────────────────────────────────
    sentiment: str
    sentiment_score: float
    frustration_turns: int

    # ── Actions ───────────────────────────────────────────────────────────
    action_result: dict
    action_taken: str

    # ── Escalation ────────────────────────────────────────────────────────
    should_escalate: bool
    escalation_reason: str
    escalation_ticket_id: str

    # ── Learning ──────────────────────────────────────────────────────────
    resolution_logged: bool

    # ── Output ────────────────────────────────────────────────────────────
    response: str