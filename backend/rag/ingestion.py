"""
rag/ingestion.py — Knowledge Base ingestion pipeline.
"""

import os
import hashlib
from typing import Optional, List
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from backend.config import get_settings

settings = get_settings()

# ─────────────────────────────────────────────
# Embedding Model (initialized once, reused)
# ─────────────────────────────────────────────

_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    """
    Returns a cached SentenceTransformer model.
    Downloads ~80MB on first run, then cached locally.
    """
    global _embedding_model
    if _embedding_model is None:
        print(f"[Embeddings] Loading model: {settings.embedding_model}")
        _embedding_model = SentenceTransformer(settings.embedding_model)
        print("[Embeddings] Model loaded.")
    return _embedding_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings, returns list of float vectors."""
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


# ─────────────────────────────────────────────
# ChromaDB Client
# ─────────────────────────────────────────────

def get_chroma_client() -> chromadb.ClientAPI:
    """Returns a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def get_collection(client: Optional[chromadb.ClientAPI] = None):
    """Returns the ChromaDB collection, creating it if it doesn't exist."""
    if client is None:
        client = get_chroma_client()

    return client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ─────────────────────────────────────────────
# Document Loading
# ─────────────────────────────────────────────

def load_documents(kb_dir: str = "./data/knowledge_base") -> List[dict]:
    """Loads all .md and .txt files from the knowledge base directory."""
    documents = []

    if not os.path.exists(kb_dir):
        print(f"[Ingestion] Knowledge base directory not found: {kb_dir}")
        return documents

    for filename in os.listdir(kb_dir):
        if filename.endswith((".md", ".txt")):
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    documents.append({
                        "content": content,
                        "source": filepath,
                        "filename": filename,
                    })
                    print(f"[Ingestion] Loaded: {filename} ({len(content)} chars)")

    print(f"[Ingestion] Total documents loaded: {len(documents)}")
    return documents


# ─────────────────────────────────────────────
# Text Splitting
# ─────────────────────────────────────────────

def split_documents(documents: List[dict]) -> List[dict]:
    """Splits documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = []
    for doc in documents:
        split_texts = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(split_texts):
            chunks.append({
                "content": chunk_text,
                "source": doc["filename"],
                "chunk_index": i,
            })

    print(f"[Ingestion] Total chunks created: {len(chunks)}")
    return chunks


# ─────────────────────────────────────────────
# Main Ingestion Function
# ─────────────────────────────────────────────

def ingest_knowledge_base(
    kb_dir: str = "./data/knowledge_base",
    force_reload: bool = False,
) -> dict:
    """Full pipeline: load -> split -> embed -> store in ChromaDB."""
    client = get_chroma_client()
    collection = get_collection(client)

    if force_reload:
        client.delete_collection(settings.chroma_collection_name)
        collection = get_collection(client)
        print("[Ingestion] Collection cleared for full reload.")

    existing_count = collection.count()
    if existing_count > 0 and not force_reload:
        print(f"[Ingestion] Collection already has {existing_count} chunks. Skipping.")
        return {"documents_loaded": 0, "chunks_created": existing_count}

    documents = load_documents(kb_dir)
    if not documents:
        print("[Ingestion] No documents found.")
        return {"documents_loaded": 0, "chunks_created": 0}

    chunks = split_documents(documents)

    ids = []
    texts = []
    metadatas = []

    for chunk in chunks:
        chunk_id = hashlib.md5(chunk["content"].encode()).hexdigest()
        ids.append(chunk_id)
        texts.append(chunk["content"])
        metadatas.append({
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
            "type": "knowledge_base",
        })

    print("[Ingestion] Computing embeddings...")
    embeddings = embed_texts(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"[Ingestion] Successfully stored {len(chunks)} chunks in ChromaDB.")
    return {"documents_loaded": len(documents), "chunks_created": len(chunks)}


# ─────────────────────────────────────────────
# Learning Agent Hook
# ─────────────────────────────────────────────

def add_resolved_ticket(question: str, resolution: str, ticket_id: str) -> bool:
    """
    Called by the Learning Agent after a human resolves a ticket.
    Stores the Q&A pair in ChromaDB for future retrieval.
    """
    try:
        collection = get_collection()
        qa_text = f"Customer Question: {question}\n\nResolution: {resolution}"
        chunk_id = f"resolved_{ticket_id}"
        embeddings = embed_texts([qa_text])

        collection.add(
            documents=[qa_text],
            embeddings=embeddings,
            ids=[chunk_id],
            metadatas=[{
                "source": "resolved_ticket",
                "ticket_id": ticket_id,
                "type": "resolved_resolution",
            }],
        )
        print(f"[Learning] Stored resolution for ticket {ticket_id} in ChromaDB.")
        return True

    except Exception as e:
        print(f"[Learning] Failed to store resolution: {e}")
        return False