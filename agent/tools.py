"""LangChain tools: hybrid RAG search and web search."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.tools import tool

from config import retriever_top_k, web_search_timeout_sec
from rag.retriever import RetrievedChunk, hybrid_search


def _serialize_chunk(ch: RetrievedChunk) -> dict[str, Any]:
    meta = dict(ch.metadata)
    return {
        "text": ch.text,
        "score": ch.score,
        "metadata": meta,
    }


@tool
def knowledge_base_search(query: str) -> str:
    """Search the user's indexed documents using hybrid vector + BM25 retrieval."""
    hits = hybrid_search(query, k=retriever_top_k())
    payload = [_serialize_chunk(h) for h in hits]
    return json.dumps(payload, ensure_ascii=False)


def _web_search_duckduckgo(query: str) -> str:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    try:
        from duckduckgo_search import DDGS
    except Exception as exc:  # pragma: no cover
        return json.dumps({"error": f"duckduckgo_search unavailable: {exc}"})

    def _run() -> list:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=5))

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run)
            results = fut.result(timeout=web_search_timeout_sec())
        slim = [{"title": r.get("title"), "href": r.get("href"), "body": r.get("body")} for r in results]
        return json.dumps(slim, ensure_ascii=False)
    except FuturesTimeout:
        return json.dumps({"error": f"web_search timed out after {web_search_timeout_sec():.0f}s"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _web_search_tavily(query: str) -> str:
    """Live web search via Tavily API (set TAVILY_API_KEY)."""
    if not os.getenv("TAVILY_API_KEY"):
        return json.dumps({"error": "TAVILY_API_KEY is not set (required for DOCUBOT_WEB_SEARCH_BACKEND=tavily)."})
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
    except Exception as exc:  # pragma: no cover
        return json.dumps({"error": f"tavily integration unavailable: {exc}"})

    try:
        tool = TavilySearchResults(max_results=5)
        raw = tool.invoke({"query": query})
        return json.dumps(raw, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@tool
def web_search(query: str) -> str:
    """Search the public web (DuckDuckGo by default; set DOCUBOT_WEB_SEARCH_BACKEND=tavily for Tavily API)."""
    backend = os.getenv("DOCUBOT_WEB_SEARCH_BACKEND", "duckduckgo").lower().strip()
    if backend == "tavily":
        return _web_search_tavily(query)
    return _web_search_duckduckgo(query)


TOOLS = [knowledge_base_search, web_search]
