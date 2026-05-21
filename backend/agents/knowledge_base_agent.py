"""
agents/knowledge_base_agent.py — Knowledge Base Agent (RAG Pipeline)

Full RAG implementation:
  1. Retrieve — semantic search over ChromaDB
  2. Augment  — build context from retrieved chunks
  3. Generate — LLM synthesizes answer from context

Also returns sources and confidence so the UI can display them.
"""

from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage

from backend.graph.state import SupportState
from backend.llm.client import get_llm_with_fallback
from backend.rag.retriever import search_knowledge_base


KB_SYSTEM_PROMPT = """You are a helpful e-commerce customer support agent.

You will be given:
1. A customer's question
2. Relevant excerpts from our knowledge base (each labeled with its source)

Your job is to provide a clear, helpful, and accurate answer based ONLY on the provided context.

Rules:
- Be concise but complete
- Use bullet points for multi-step instructions
- If the context doesn't contain enough information, say:
  "I don't have enough information to answer that. Let me connect you with a human agent."
- Never make up information not present in the context
- Be empathetic and professional
- End your response with a brief "Source: [filename]" line indicating which document helped most
"""


def knowledge_base_agent_node(state: SupportState) -> dict[str, Any]:
    """
    LangGraph node: full RAG pipeline.

    Reads:  current_input, intent, entities
    Writes: retrieved_docs, kb_answer, kb_confidence, kb_sources
    """
    user_input = state["current_input"]
    entities   = state.get("entities", {})

    # Enrich query with entities for better retrieval
    search_query = user_input
    if entities.get("order_id"):
        search_query = f"{user_input} order tracking status"

    # ── Retrieve ──────────────────────────────────────────────────────────
    results = search_knowledge_base(search_query)

    if not results:
        print("[KBAgent] No relevant chunks found.")
        return {
            "retrieved_docs": [],
            "kb_answer": "",
            "kb_confidence": 0.0,
            "kb_sources": [],
        }

    # ── Augment ───────────────────────────────────────────────────────────
    # Build labeled context so the LLM knows where each chunk came from
    context_parts = []
    for i, r in enumerate(results, 1):
        source = r["source"].replace(".md", "").replace("_", " ").title()
        context_parts.append(
            f"[Source {i}: {source} | Relevance: {r['relevance']:.0%}]\n{r['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # Track unique sources used
    sources = list(dict.fromkeys(r["source"] for r in results))
    avg_confidence = sum(r["relevance"] for r in results) / len(results)

    # ── Generate ──────────────────────────────────────────────────────────
    llm = get_llm_with_fallback(temperature=0.3)

    messages = [
        SystemMessage(content=KB_SYSTEM_PROMPT),
        HumanMessage(content=f"""Knowledge Base Context:
{context}

Customer Question: {user_input}

Please provide a helpful answer based on the context above."""),
    ]

    try:
        response = llm.invoke(messages)
        kb_answer = response.content.strip()
    except Exception as e:
        print(f"[KBAgent] LLM call failed: {e}")
        kb_answer = ""
        avg_confidence = 0.0

    print(f"[KBAgent] Retrieved {len(results)} chunks | Sources: {sources} | Confidence: {avg_confidence:.2f}")

    return {
        "retrieved_docs": [r["content"] for r in results],
        "kb_answer":      kb_answer,
        "kb_confidence":  round(avg_confidence, 4),
        "kb_sources":     sources,
    }