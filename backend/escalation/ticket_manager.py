"""
escalation/ticket_manager.py — Escalation Ticket Manager

Creates and stores escalation tickets when a conversation needs
to be handed off to a human agent.

For the MVP, tickets are stored in memory (dict).
In production this would write to a database.
"""

import uuid
from datetime import datetime
from typing import Optional

# In-memory ticket store: {ticket_id: ticket_dict}
# In production, replace with a database
_tickets: dict = {}


def create_ticket(
    session_id: str,
    customer_name: str,
    reason: str,
    conversation_history: list,
    sentiment: str,
    intent: str,
) -> dict:
    """
    Creates a new escalation ticket and stores it.

    Args:
        session_id:           The customer's session ID
        customer_name:        Customer name if available
        reason:               Why escalation was triggered
        conversation_history: Full message history for context
        sentiment:            Customer's current sentiment
        intent:               What the customer was trying to do

    Returns:
        The created ticket dict
    """
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"

    # Serialize messages to plain text for storage
    history_text = []
    for msg in conversation_history:
        role = "Customer" if msg.__class__.__name__ == "HumanMessage" else "Agent"
        history_text.append(f"{role}: {msg.content}")

    ticket = {
        "ticket_id": ticket_id,
        "session_id": session_id,
        "customer_name": customer_name,
        "reason": reason,
        "intent": intent,
        "sentiment": sentiment,
        "conversation_history": history_text,
        "created_at": datetime.now().isoformat(),
        "status": "open",           # open | resolved
        "resolution": None,         # filled in when human resolves
        "resolved_at": None,
    }

    _tickets[ticket_id] = ticket
    print(f"[TicketManager] Created ticket {ticket_id} | Reason: {reason}")

    return ticket


def get_ticket(ticket_id: str) -> Optional[dict]:
    """Retrieves a ticket by ID."""
    return _tickets.get(ticket_id)


def resolve_ticket(ticket_id: str, resolution: str) -> Optional[dict]:
    """
    Marks a ticket as resolved with the human agent's answer.
    Called by the Learning Agent hook.

    Returns the updated ticket or None if not found.
    """
    ticket = _tickets.get(ticket_id)
    if not ticket:
        return None

    ticket["status"] = "resolved"
    ticket["resolution"] = resolution
    ticket["resolved_at"] = datetime.now().isoformat()

    print(f"[TicketManager] Resolved ticket {ticket_id}")
    return ticket


def get_all_open_tickets() -> list:
    """Returns all open (unresolved) tickets. Used by the agent dashboard."""
    return [t for t in _tickets.values() if t["status"] == "open"]


def get_all_tickets() -> list:
    """Returns all tickets. Used by the agent dashboard."""
    return list(_tickets.values())