from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage


class SupportState(TypedDict):
    session_id: str
    customer_name: str
    messages: List[BaseMessage]
    current_input: str
    attempt_count: int

    intent: str
    entities: dict

    retrieved_docs: List[str]
    kb_answer: str
    kb_confidence: float
    kb_sources: List[str]

    sentiment: str
    sentiment_score: float
    frustration_turns: int

    action_result: dict
    action_taken: str

    resolution_level: int
    clarification_asked: bool
    alternatives_suggested: bool
    escalate_requested_count: int

    should_escalate: bool
    escalation_reason: str
    escalation_ticket_id: str

    resolution_logged: bool
    response: str