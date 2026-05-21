"""
memory/session_store.py — In-memory session store.

Stores conversation state per session_id.
FastAPI keeps this alive for the lifetime of the server process.

Structure:
    {
        "session_id": {
            "session_id": str,
            "customer_name": str,
            "messages": List[BaseMessage],
            "attempt_count": int,
            "frustration_turns": int,
            "created_at": str,
        }
    }
"""

from datetime import datetime
from typing import Optional
from langchain_core.messages import BaseMessage

from backend.config import get_settings

settings = get_settings()

# The store — lives in memory for the lifetime of the FastAPI process
_sessions: dict = {}


def create_session(session_id: str, customer_name: str = "Customer") -> dict:
    """Creates a new session and returns it."""
    session = {
        "session_id": session_id,
        "customer_name": customer_name,
        "messages": [],
        "attempt_count": 0,
        "frustration_turns": 0,
        "created_at": datetime.now().isoformat(),
    }
    _sessions[session_id] = session
    print(f"[SessionStore] Created session: {session_id}")
    return session


def get_session(session_id: str) -> Optional[dict]:
    """Returns session by ID, or None if not found."""
    return _sessions.get(session_id)


def get_or_create_session(session_id: str, customer_name: str = "Customer") -> dict:
    """Returns existing session or creates a new one."""
    session = _sessions.get(session_id)
    if session is None:
        session = create_session(session_id, customer_name)
    return session


def update_session(session_id: str, updates: dict) -> Optional[dict]:
    """
    Updates specific fields in a session.
    Called after each LangGraph turn to persist state.
    """
    session = _sessions.get(session_id)
    if session is None:
        return None

    session.update(updates)
    return session


def add_message(session_id: str, message: BaseMessage) -> None:
    """Appends a message to session history, respecting max_history limit."""
    session = _sessions.get(session_id)
    if session is None:
        return

    session["messages"].append(message)

    # Trim to last N messages to avoid unbounded growth
    max_msgs = settings.max_history_messages
    if len(session["messages"]) > max_msgs:
        session["messages"] = session["messages"][-max_msgs:]


def delete_session(session_id: str) -> bool:
    """Deletes a session. Returns True if it existed."""
    if session_id in _sessions:
        del _sessions[session_id]
        print(f"[SessionStore] Deleted session: {session_id}")
        return True
    return False


def get_all_sessions() -> list:
    """Returns all active sessions. Used for monitoring."""
    return list(_sessions.values())