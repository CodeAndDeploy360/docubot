"""Load settings from environment (.env via python-dotenv in app entrypoints)."""

from __future__ import annotations

import os
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def chroma_path() -> Path:
    return Path(os.getenv("DOCUBOT_CHROMA_PATH", "./vector_db")).resolve()


def pii_redaction_enabled() -> bool:
    return _bool("DOCUBOT_PII_REDACTION_ENABLED", True)


def chat_model() -> str:
    # Default chat model follows DOCUBOT_LLM_PROVIDER (Gemini).
    # gemini-1.5-flash is often 404 on current Gemini Developer API (v1beta); use a 2.5/2.0 id.
    # Default avoids gemini-2.5-flash-lite as primary: free tier often caps it very low (e.g. 20
    # generateContent calls/day per project). gemini-2.5-flash usually has a separate quota pool.
    default_model = (
        "gemini-2.5-flash"
        if llm_provider() == "gemini"
        else "gpt-4o-mini"
    )
    return os.getenv("DOCUBOT_CHAT_MODEL", default_model)


def embedding_provider() -> str:
    return os.getenv("DOCUBOT_EMBEDDING_PROVIDER", "openai").lower()


def embedding_model() -> str:
    return os.getenv("DOCUBOT_EMBEDDING_MODEL", "text-embedding-3-small")


def llm_provider() -> str:
    return os.getenv("DOCUBOT_LLM_PROVIDER", "gemini").lower()


def gemini_api_key() -> str | None:
    """Non-empty Gemini / Google API key from env (after ``load_dotenv`` / Streamlit secrets)."""
    raw = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if raw is None:
        return None
    key = raw.strip()
    return key or None


def ollama_base_url() -> str | None:
    return os.getenv("DOCUBOT_OLLAMA_BASE_URL")


def llm_timeout_sec() -> float:
    """HTTP timeout for a single LLM request (Gemini / OpenAI / Ollama)."""
    return float(os.getenv("DOCUBOT_LLM_TIMEOUT_SEC", "120"))


def graph_invoke_timeout_sec() -> float:
    """Max wall time for the full LangGraph run (planner + retriever + grader + replans)."""
    # Default 900s: full mode issues many LLM calls; 429 retries can sleep 15–120s each.
    return float(os.getenv("DOCUBOT_GRAPH_TIMEOUT_SEC", "900"))


def web_search_timeout_sec() -> float:
    return float(os.getenv("DOCUBOT_WEB_SEARCH_TIMEOUT_SEC", "25"))


def max_sources_display() -> int:
    """Max source lines under the assistant bubble (top chunks by retrieval score)."""
    return max(1, min(12, int(os.getenv("DOCUBOT_MAX_SOURCES_DISPLAY", "3"))))


def retriever_top_k() -> int:
    """Chunks to retrieve per question (hybrid fusion). Lower = slightly faster and shorter prompts."""
    return max(1, min(32, int(os.getenv("DOCUBOT_RETRIEVER_K", "8"))))
