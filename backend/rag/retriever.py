"""
rag/retriever.py — Semantic search interface over ChromaDB.
"""

from typing import Optional, List, Tuple
from backend.config import get_settings
from backend.rag.ingestion import get_collection, embed_texts

settings = get_settings()


def search_knowledge_base(
    query: str,
    top_k: Optional[int] = None,
    filter_type: Optional[str] = None,
) -> List[dict]:
    """
    Searches the knowledge base for chunks relevant to the query.

    Returns list of dicts with keys: content, source, distance, relevance
    """
    if top_k is None:
        top_k = settings.rag_top_k

    collection = get_collection()
    total = collection.count()

    if total == 0:
        print("[Retriever] Warning: Knowledge base is empty. Run ingestion first.")
        return []

    query_embedding = embed_texts([query])[0]

    where = None
    if filter_type:
        where = {"type": {"$eq": filter_type}}

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"[Retriever] Search failed: {e}")
        return []

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    chunks = []
    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append({
            "content": doc,
            "source": meta.get("source", "unknown"),
            "distance": round(dist, 4),
            "relevance": round(1 - dist, 4),
        })

    return chunks


def get_kb_status() -> dict:
    """Returns the current state of the knowledge base."""
    try:
        collection = get_collection()
        count = collection.count()
        return {
            "ready": count > 0,
            "total_chunks": count,
            "collection": settings.chroma_collection_name,
        }
    except Exception as e:
        return {"ready": False, "total_chunks": 0, "error": str(e)}


def get_relevant_context(query: str, top_k: Optional[int] = None) -> Tuple[List[str], float]:
    """
    Convenience function for agents.
    Returns (chunks, confidence_score).
    """
    results = search_knowledge_base(query, top_k=top_k)

    if not results:
        return [], 0.0

    chunks = [r["content"] for r in results]
    avg_relevance = sum(r["relevance"] for r in results) / len(results)

    return chunks, round(avg_relevance, 4)