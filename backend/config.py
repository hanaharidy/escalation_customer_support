"""
config.py — Central configuration for the AI Support System.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    llm_provider: str = "groq"

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    embedding_model: str = "all-MiniLM-L6-v2"

    chroma_persist_dir: str = "./chroma_data"
    chroma_collection_name: str = "support_knowledge_base"

    max_history_messages: int = 10

    # Escalation — raised thresholds, escalate less eagerly
    sentiment_escalation_threshold: float = -0.7   # was -0.5, now requires true anger
    max_frustration_turns: int = 2                  # must be angry for 2+ turns
    max_failed_attempts: int = 2                    # failed twice with low confidence

    rag_top_k: int = 3

    app_name: str = "AI Customer Support System"
    debug: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()