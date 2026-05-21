"""
memory/session_store.py - In-memory session store.
"""

from datetime import datetime
from typing import Optional
from backend.config import get_settings

settings  = get_settings()
_sessions = {}


def create_session(session_id: str, customer_name: str = "Customer") -> dict:
    session = {
        "session_id":               session_id,
        "customer_name":            customer_name,
        "messages":                 [],
        "attempt_count":            0,
        "frustration_turns":        0,
        "resolution_level":         1,
        "clarification_asked":      False,
        "alternatives_suggested":   False,
        "escalate_requested_count": 0,
        "created_at":               datetime.now().isoformat(),
    }
    _sessions[session_id] = session
    print(f"[SessionStore] Created session: {session_id}")
    return session


def get_session(session_id: str) -> Optional[dict]:
    return _sessions.get(session_id)


def get_or_create_session(session_id: str, customer_name: str = "Customer") -> dict:
    return _sessions.get(session_id) or create_session(session_id, customer_name)


def update_session(session_id: str, updates: dict) -> Optional[dict]:
    session = _sessions.get(session_id)
    if session is None:
        return None
    session.update(updates)
    return session


def delete_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def get_all_sessions() -> list:
    return list(_sessions.values())