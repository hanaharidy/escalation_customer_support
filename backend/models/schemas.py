"""
schemas.py — Pydantic models for API request/response validation.
"""

from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class Intent(str, Enum):
    ORDER_TRACKING  = "order_tracking"
    REFUND          = "refund"
    PRODUCT_ISSUE   = "product_issue"
    COMPLAINT       = "complaint"
    ESCALATE        = "escalate"
    OTHER           = "other"


class Sentiment(str, Enum):
    POSITIVE   = "positive"
    NEUTRAL    = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY      = "angry"


class UserMessage(BaseModel):
    session_id: str
    content: str


class AgentMessage(BaseModel):
    session_id: str
    content: str
    intent: Optional[str] = None
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    is_escalated: bool = False
    escalation_reason: Optional[str] = None
    escalation_ticket_id: Optional[str] = None
    action_taken: Optional[str] = None
    sources_used: Optional[List[str]] = None


class SessionStartRequest(BaseModel):
    session_id: str
    customer_name: Optional[str] = "Customer"


class SessionStartResponse(BaseModel):
    session_id: str
    message: str


class ResolveTicketRequest(BaseModel):
    ticket_id: str
    original_question: str
    resolution_answer: str


class ResolveTicketResponse(BaseModel):
    ticket_id: str
    kb_updated: bool
    message: str


class KBIngestionRequest(BaseModel):
    force_reload: bool = False


class KBIngestionResponse(BaseModel):
    documents_loaded: int
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    chroma_ready: bool
    message: str