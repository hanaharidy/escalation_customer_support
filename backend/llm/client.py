"""
llm/client.py — LLM abstraction layer.

Supports Groq, Gemini, and Ollama behind a single interface.
Switch providers by changing LLM_PROVIDER in .env — zero code changes.
"""

from typing import Optional
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel

from backend.config import get_settings


def get_llm(
    temperature: float = 0.3,
    provider: Optional[str] = None
) -> BaseChatModel:
    """
    Returns a LangChain chat model for the configured provider.

    Args:
        temperature: 0.1 for classification tasks, 0.3 for responses.
        provider:    Override the default provider from config.
    """
    settings = get_settings()
    chosen_provider = provider or settings.llm_provider

    if chosen_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set in .env")
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=temperature,
        )

    elif chosen_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")
        return ChatGoogleGenerativeAI(
            google_api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            temperature=temperature,
        )

    elif chosen_provider == "ollama":
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{chosen_provider}'. "
            "Valid options: groq | gemini | ollama"
        )


def get_llm_with_fallback(temperature: float = 0.3) -> BaseChatModel:
    """
    Tries the primary provider first.
    Falls back to Gemini if primary fails (e.g. Groq rate limit).
    """
    settings = get_settings()

    try:
        return get_llm(temperature=temperature)
    except Exception as primary_error:
        print(f"[LLM] Primary provider '{settings.llm_provider}' failed: {primary_error}")
        print("[LLM] Falling back to Gemini...")

        if settings.llm_provider != "gemini" and settings.gemini_api_key:
            return get_llm(temperature=temperature, provider="gemini")

        raise RuntimeError(
            "Primary LLM failed and no fallback available. "
            "Set GEMINI_API_KEY in .env for automatic fallback."
        ) from primary_error