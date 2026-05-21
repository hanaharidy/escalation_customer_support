"""
frontend/app.py — Streamlit Chat UI
"""

import streamlit as st
import websockets
import asyncio
import json
import httpx
import uuid

BACKEND_URL = "http://localhost:8000"
WS_URL      = "ws://localhost:8000"

st.set_page_config(
    page_title="AI Customer Support",
    page_icon="🤖",
    layout="wide",
)

if "session_id"          not in st.session_state: st.session_state.session_id = str(uuid.uuid4())[:8]
if "chat_history"        not in st.session_state: st.session_state.chat_history = []
if "is_escalated"        not in st.session_state: st.session_state.is_escalated = False
if "current_ticket_id"   not in st.session_state: st.session_state.current_ticket_id = ""
if "customer_name"       not in st.session_state: st.session_state.customer_name = "Customer"


async def send_message_ws(session_id: str, message: str) -> dict:
    uri = f"{WS_URL}/ws/{session_id}"
    try:
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({"content": message}))
            raw = await ws.recv()
            return json.loads(raw)
    except Exception as e:
        return {
            "content": f"Connection error: {str(e)}. Is the FastAPI server running?",
            "is_escalated": False,
            "intent": "",
            "sentiment": "",
            "kb_sources": [],
            "kb_confidence": 0.0,
        }


def chat(message: str) -> dict:
    return asyncio.run(send_message_ws(st.session_state.session_id, message))


def get_tickets() -> list:
    try:
        r = httpx.get(f"{BACKEND_URL}/tickets", timeout=5)
        return r.json().get("tickets", [])
    except Exception:
        return []


def resolve_ticket(ticket_id: str, original_question: str, resolution: str) -> dict:
    try:
        r = httpx.post(
            f"{BACKEND_URL}/resolve-ticket",
            json={
                "ticket_id": ticket_id,
                "original_question": original_question,
                "resolution_answer": resolution,
            },
            timeout=10,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def sentiment_badge(sentiment: str, score: float) -> str:
    badges = {
        "positive":   "🟢 Positive",
        "neutral":    "⚪ Neutral",
        "frustrated": "🟡 Frustrated",
        "angry":      "🔴 Angry",
    }
    return f"{badges.get(sentiment, sentiment)} ({score:+.1f})"


def confidence_badge(score: float) -> str:
    if score >= 0.6:   return f"🟢 High ({score:.0%})"
    elif score >= 0.4: return f"🟡 Medium ({score:.0%})"
    elif score > 0.0:  return f"🔴 Low ({score:.0%})"
    else:              return "⚫ N/A"


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_chat, tab_dashboard = st.tabs(["💬 Customer Chat", "🎫 Agent Dashboard"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CUSTOMER CHAT
# ══════════════════════════════════════════════════════════════════════════════

with tab_chat:
    col_title, col_info = st.columns([3, 1])
    with col_title:
        st.title("🤖 AI Customer Support")
        st.caption(f"Session ID: `{st.session_state.session_id}`")
    with col_info:
        st.metric("Status", "🟢 Online" if not st.session_state.is_escalated else "🔴 Escalated")

    with st.sidebar:
        st.header("⚙️ Session")
        name = st.text_input("Your name", value=st.session_state.customer_name)
        if name != st.session_state.customer_name:
            st.session_state.customer_name = name

        st.divider()
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.session_id         = str(uuid.uuid4())[:8]
            st.session_state.chat_history       = []
            st.session_state.is_escalated       = False
            st.session_state.current_ticket_id  = ""
            st.rerun()

        st.divider()
        st.caption("**Try asking:**")
        st.caption("• Where is my order ORD-123?")
        st.caption("• I want a refund for ORD-456")
        st.caption("• What is your return policy?")
        st.caption("• My package arrived damaged")
        st.caption("• I want to speak to a human")

    # Chat history
    if not st.session_state.chat_history:
        st.info("👋 Welcome! How can I help you today?")
    else:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                if msg["role"] == "assistant" and msg.get("meta"):
                    meta = msg["meta"]
                    cols = st.columns(4)

                    if meta.get("intent"):
                        cols[0].caption(f"🎯 `{meta['intent']}`")

                    if meta.get("sentiment"):
                        cols[1].caption(f"😊 {sentiment_badge(meta['sentiment'], meta.get('sentiment_score', 0))}")

                    if meta.get("kb_confidence", 0) > 0:
                        cols[2].caption(f"📚 {confidence_badge(meta['kb_confidence'])}")

                    if meta.get("kb_sources"):
                        sources = ", ".join(meta["kb_sources"])
                        cols[3].caption(f"🗂 {sources}")

                    if meta.get("action_taken") and meta["action_taken"] not in ("", "requested_order_id"):
                        st.caption(f"⚡ Action taken: `{meta['action_taken']}`")

    if st.session_state.is_escalated:
        st.warning(
            f"🎫 Escalated to human agent. Ticket: **{st.session_state.current_ticket_id}**",
            icon="⚠️",
        )

    if not st.session_state.is_escalated:
        user_input = st.chat_input("Type your message here...")
        if user_input:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "meta": {},
            })

            with st.spinner("Thinking..."):
                response = chat(user_input)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response.get("content", "Sorry, something went wrong."),
                "meta": {
                    "intent":          response.get("intent", ""),
                    "sentiment":       response.get("sentiment", ""),
                    "sentiment_score": response.get("sentiment_score", 0.0),
                    "kb_confidence":   response.get("kb_confidence", 0.0),
                    "kb_sources":      response.get("kb_sources", []),
                    "action_taken":    response.get("action_taken", ""),
                },
            })

            if response.get("is_escalated"):
                st.session_state.is_escalated     = True
                st.session_state.current_ticket_id = response.get("escalation_ticket_id", "")

            st.rerun()
    else:
        st.chat_input("Chat disabled — connected to human agent.", disabled=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — AGENT DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

with tab_dashboard:
    st.title("🎫 Human Agent Dashboard")
    st.caption("Resolve escalated tickets. Resolutions are stored in the knowledge base automatically.")

    if st.button("🔄 Refresh", use_container_width=False):
        st.rerun()

    tickets        = get_tickets()
    open_tickets   = [t for t in tickets if t["status"] == "open"]
    closed_tickets = [t for t in tickets if t["status"] == "resolved"]

    col1, col2 = st.columns(2)
    col1.metric("Open Tickets",     len(open_tickets))
    col2.metric("Resolved Tickets", len(closed_tickets))

    if not tickets:
        st.info("No tickets yet. Escalated conversations will appear here.")

    if open_tickets:
        st.subheader("🔴 Open Tickets")
        for ticket in open_tickets:
            with st.expander(
                f"🎫 {ticket['ticket_id']} — {ticket['reason']} — {ticket['created_at'][:16]}",
                expanded=True,
            ):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Customer:** {ticket['customer_name']}")
                c1.markdown(f"**Intent:** `{ticket['intent']}`")
                c2.markdown(f"**Sentiment:** `{ticket['sentiment']}`")
                c2.markdown(f"**Reason:** `{ticket['reason']}`")

                st.markdown("**Conversation History:**")
                history = ticket.get("conversation_history", [])
                if history:
                    st.text_area(
                        "history",
                        value="\n".join(history),
                        height=150,
                        disabled=True,
                        key=f"hist_{ticket['ticket_id']}",
                        label_visibility="collapsed",
                    )

                st.divider()
                st.markdown("**✍️ Resolve this ticket:**")

                original_q = st.text_input(
                    "Original customer question",
                    key=f"q_{ticket['ticket_id']}",
                    placeholder="What was the customer asking?",
                )
                resolution = st.text_area(
                    "Your resolution",
                    key=f"res_{ticket['ticket_id']}",
                    placeholder="Write the answer that resolves this issue...",
                    height=100,
                )

                if st.button(
                    "✅ Resolve & Update Knowledge Base",
                    key=f"btn_{ticket['ticket_id']}",
                    type="primary",
                ):
                    if original_q and resolution:
                        result = resolve_ticket(ticket["ticket_id"], original_q, resolution)
                        if result.get("kb_updated"):
                            st.success(
                                f"✅ Resolved! Knowledge base updated — "
                                f"future similar questions will use this answer."
                            )
                            st.rerun()
                        else:
                            st.error("Failed to resolve. Try again.")
                    else:
                        st.warning("Fill in both fields before resolving.")

    if closed_tickets:
        st.subheader("✅ Resolved Tickets")
        for ticket in closed_tickets:
            with st.expander(
                f"✅ {ticket['ticket_id']} — resolved {ticket.get('resolved_at','')[:16]}"
            ):
                st.markdown(f"**Resolution:** {ticket.get('resolution', 'N/A')}")
                st.markdown(f"**Escalation reason:** `{ticket['reason']}`")