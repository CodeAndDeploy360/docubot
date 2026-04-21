"""Chat LLM: Google Gemini (default), OpenAI, or Ollama. Embeddings stay in rag/embedder.py (OpenAI)."""

from __future__ import annotations

import os
import random
import re
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from config import chat_model, gemini_api_key, llm_provider, llm_timeout_sec, ollama_base_url

T = TypeVar("T")

_llm_cache: dict[tuple[Any, ...], BaseChatModel] = {}
_llm_lock = threading.Lock()


def _llm_retry_attempts() -> int:
    return max(1, int(os.getenv("DOCUBOT_LLM_RETRY_ATTEMPTS", "5")))


def is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if "429" in str(exc):
        return True
    if "resource_exhausted" in msg:
        return True
    if "quota" in msg and ("exceed" in msg or "exceeded" in msg):
        return True
    if "rate" in msg and "limit" in msg:
        return True
    return False


def sleep_rate_limit_backoff(exc: BaseException) -> None:
    wait = 15.0
    m = re.search(r"retry in ([\d.]+)s", str(exc), re.I)
    if m:
        wait = float(m.group(1))
    time.sleep(min(wait + random.uniform(0.25, 2.0), 120.0))


def invoke_with_rate_limit_retry(fn: Callable[[], T]) -> T:
    """Retry callable on Gemini/Google 429 RESOURCE_EXHAUSTED and similar quota errors."""
    attempts = _llm_retry_attempts()
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last = e
            if not is_rate_limit_error(e) or attempt == attempts - 1:
                raise
            sleep_rate_limit_backoff(e)
    assert last is not None
    raise last


def _gemini_api_key() -> str:
    key = gemini_api_key()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY or GOOGLE_API_KEY is required when DOCUBOT_LLM_PROVIDER=gemini."
        )
    return key


def _flatten_message_content(content: Any) -> str:
    """Turn LangChain / provider message content into plain text (streaming-safe)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype in ("thinking", "reasoning", "tool_use", "function_call"):
                continue
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
                continue
            if btype == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content)


def _build_chat_model_impl(*, temperature: float, streaming: bool) -> BaseChatModel:
    provider = llm_provider()
    model = chat_model()
    timeout = llm_timeout_sec()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=_gemini_api_key(),
            temperature=temperature,
            streaming=streaming,
            timeout=timeout,
        )

    if provider == "ollama":
        from langchain_openai import ChatOpenAI

        base = ollama_base_url() or "http://localhost:11434/v1"
        return ChatOpenAI(
            base_url=base,
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            model=model,
            temperature=temperature,
            streaming=streaming,
            timeout=timeout,
        )

    from langchain_openai import ChatOpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set (required when DOCUBOT_LLM_PROVIDER=openai).")
    return ChatOpenAI(
        api_key=key,
        model=model,
        temperature=temperature,
        streaming=streaming,
        timeout=timeout,
    )


def build_chat_model(*, temperature: float = 0.2, streaming: bool = False) -> BaseChatModel:
    """Return a cached model instance (same provider/model/streaming/temperature) to avoid per-turn client setup."""
    key = (llm_provider(), chat_model(), streaming, round(float(temperature), 5))
    with _llm_lock:
        if key not in _llm_cache:
            _llm_cache[key] = _build_chat_model_impl(temperature=temperature, streaming=streaming)
        return _llm_cache[key]


def stream_messages(messages: list[BaseMessage], **kwargs: Any) -> Iterator[tuple[str, dict[str, Any] | None]]:
    attempts = _llm_retry_attempts()
    for attempt in range(attempts):
        try:
            yield from _stream_messages_once(messages, **kwargs)
            return
        except Exception as e:
            if not is_rate_limit_error(e) or attempt == attempts - 1:
                raise
            sleep_rate_limit_backoff(e)


def _stream_messages_once(
    messages: list[BaseMessage], **kwargs: Any
) -> Iterator[tuple[str, dict[str, Any] | None]]:
    llm = build_chat_model(streaming=True)
    t0 = time.perf_counter()
    usage: dict[str, Any] = {}
    yielded_text = False
    used_fallback = False

    for chunk in llm.stream(messages, **kwargs):
        if not isinstance(chunk, AIMessageChunk):
            continue
        piece = _flatten_message_content(chunk.content)
        if piece:
            yielded_text = True
            yield piece, None
        meta = getattr(chunk, "usage_metadata", None)
        if meta:
            usage.update(dict(meta))

    if not yielded_text:
        fb = _invoke_plain_after_empty_stream(messages, **kwargs)
        if fb:
            yielded_text = True
            used_fallback = True
            yield fb, None

    usage["latency_sec"] = round(time.perf_counter() - t0, 4)
    usage["stream_fallback_invoke"] = used_fallback
    yield "", usage


def _invoke_plain_after_empty_stream(messages: list[BaseMessage], **kwargs: Any) -> str:
    """Non-streaming invoke when streaming yielded no text (common with some Gemini/LC paths)."""
    llm = build_chat_model(streaming=False)

    def _call() -> AIMessage:
        return llm.invoke(messages, **kwargs)

    res = invoke_with_rate_limit_retry(_call)
    if isinstance(res, AIMessage):
        return _flatten_message_content(res.content)
    return _flatten_message_content(getattr(res, "content", res))
