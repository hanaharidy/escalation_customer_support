"""
main.py - FastAPI application entry point.
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.graph.workflow import get_graph
from backend.memory.session_store import get_or_create_session, update_session
from backend.rag.ingestion import ingest_knowledge_base, add_resolved_ticket
from backend.rag.retriever import get_kb_status
from backend.actions.database import init_database
from backend.escalation.ticket_manager import (
    get_ticket, resolve_ticket,
    get_all_tickets, get_all_open_tickets,
)
from backend.models.schemas import (
    SessionStartRequest, SessionStartResponse,
    ResolveTicketRequest, ResolveTicketResponse,
    KBIngestionRequest, KBIngestionResponse,
    HealthResponse,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Initializing database...")
    init_database()

    print("[Startup] Ingesting knowledge base...")
    ingest_knowledge_base()

    print("[Startup] Warming up LangGraph...")
    get_graph()

    print("[Startup] Ready.")
    yield
    print("[Shutdown] Goodbye.")


app = FastAPI(title="AI Customer Support System", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    kb = get_kb_status()
    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        chroma_ready=kb["ready"],
        message=f"KB has {kb.get('total_chunks', 0)} chunks. LLM: {settings.llm_provider}.",
    )


@app.post("/start-session", response_model=SessionStartResponse)
async def start_session(request: SessionStartRequest):
    session = get_or_create_session(
        session_id=request.session_id,
        customer_name=request.customer_name or "Customer",
    )
    return SessionStartResponse(
        session_id=session["session_id"],
        message=f"Session started. Hello, {session['customer_name']}!",
    )


@app.post("/ingest-kb", response_model=KBIngestionResponse)
async def ingest_kb(request: KBIngestionRequest):
    result = ingest_knowledge_base(force_reload=request.force_reload)
    return KBIngestionResponse(
        documents_loaded=result["documents_loaded"],
        chunks_created=result["chunks_created"],
        message="Knowledge base ingestion complete.",
    )


@app.post("/resolve-ticket", response_model=ResolveTicketResponse)
async def resolve_ticket_endpoint(request: ResolveTicketRequest):
    ticket = get_ticket(request.ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {request.ticket_id} not found.")

    resolve_ticket(request.ticket_id, request.resolution_answer)

    kb_updated = add_resolved_ticket(
        question=request.original_question,
        resolution=request.resolution_answer,
        ticket_id=request.ticket_id,
    )

    return ResolveTicketResponse(
        ticket_id=request.ticket_id,
        kb_updated=kb_updated,
        message=f"Ticket {request.ticket_id} resolved. KB updated: {kb_updated}.",
    )


@app.get("/tickets")
async def list_tickets(open_only: bool = False):
    tickets = get_all_open_tickets() if open_only else get_all_tickets()
    return {"tickets": tickets, "count": len(tickets)}


@app.get("/tickets/{ticket_id}")
async def get_single_ticket(ticket_id: str):
    ticket = get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found.")
    return ticket


@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    print(f"[WebSocket] Connected: {session_id}")

    session = get_or_create_session(session_id)
    graph   = get_graph()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data       = json.loads(raw)
                user_input = data.get("content", "").strip()
            except json.JSONDecodeError:
                user_input = raw.strip()

            if not user_input:
                continue

            print(f"[WebSocket] [{session_id}] User: {user_input}")

            graph_input = {
                "session_id":              session_id,
                "customer_name":           session.get("customer_name", "Customer"),
                "messages":                session.get("messages", []),
                "current_input":           user_input,
                "attempt_count":           session.get("attempt_count", 0),
                "intent":                  "",
                "entities":                {},
                "retrieved_docs":          [],
                "kb_answer":               "",
                "kb_confidence":           0.0,
                "kb_sources":              [],
                "sentiment":               "neutral",
                "sentiment_score":         0.0,
                "frustration_turns":       session.get("frustration_turns", 0),
                "action_result":           {},
                "action_taken":            "",
                "resolution_level":        session.get("resolution_level", 1),
                "clarification_asked":     session.get("clarification_asked", False),
                "alternatives_suggested":  session.get("alternatives_suggested", False),
                "escalate_requested_count": session.get("escalate_requested_count", 0),
                "should_escalate":         False,
                "escalation_reason":       "",
                "escalation_ticket_id":    "",
                "resolution_logged":       False,
                "response":                "",
            }

            result = graph.invoke(graph_input)

            update_session(session_id, {
                "messages":               result.get("messages", []),
                "frustration_turns":      result.get("frustration_turns", 0),
                "resolution_level":       result.get("resolution_level", 1),
                "clarification_asked":    result.get("clarification_asked", False),
                "alternatives_suggested":   result.get("alternatives_suggested", False),
                "escalate_requested_count": result.get("escalate_requested_count", 0),
                "attempt_count": (
                    session.get("attempt_count", 0) + 1
                    if not result.get("response") else 0
                ),
            })

            raw_sources      = result.get("kb_sources", [])
            friendly_sources = [
                s.replace(".md", "").replace("_", " ").title()
                for s in raw_sources
            ]

            response_payload = {
                "content":              result.get("response", "I'm sorry, I couldn't process that."),
                "intent":               result.get("intent", ""),
                "sentiment":            result.get("sentiment", ""),
                "sentiment_score":      result.get("sentiment_score", 0.0),
                "kb_confidence":        result.get("kb_confidence", 0.0),
                "kb_sources":           friendly_sources,
                "resolution_level":     result.get("resolution_level", 1),
                "is_escalated":         result.get("should_escalate", False),
                "escalation_reason":    result.get("escalation_reason", ""),
                "escalation_ticket_id": result.get("escalation_ticket_id", ""),
                "action_taken":         result.get("action_taken", ""),
            }

            await websocket.send_text(json.dumps(response_payload))
            print(f"[WebSocket] [{session_id}] Level: {result.get('resolution_level', 1)} | Action: {result.get('action_taken', 'none')}")

    except WebSocketDisconnect:
        print(f"[WebSocket] Disconnected: {session_id}")
    except Exception as e:
        print(f"[WebSocket] Error in session {session_id}: {e}")
        try:
            await websocket.send_text(json.dumps({
                "content": "I encountered an internal error. Please try again.",
                "is_escalated": False,
            }))
        except Exception:
            pass