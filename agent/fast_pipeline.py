"""Minimal RAG path: hybrid search once → one Gemini generation (saves quota vs full LangGraph)."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent import prompts
from agent.graph import _evidence_block, format_unique_source_lines
from config import retriever_top_k
from rag.retriever import hybrid_search


def run_fast_turn(user_query: str) -> dict[str, Any]:
    """
    Return the same keys the UI expects as ``graph.invoke`` (generation_messages, sources, trace).
    Uses exactly **one** chat LLM call after retrieval (plus retries inside ``stream_messages``).
    """
    t0 = time.perf_counter()
    hits = hybrid_search(user_query, k=retriever_top_k())
    retrieved: list[dict[str, Any]] = []
    for i, h in enumerate(hits, start=1):
        retrieved.append(
            {
                "kind": "rag",
                "label": f"S{i}",
                "text": h.text,
                "metadata": dict(h.metadata),
                "score": float(h.score),
            }
        )

    context = _evidence_block(retrieved)
    sys = SystemMessage(content=prompts.GENERATOR_SYSTEM)
    human = HumanMessage(
        content=prompts.GENERATOR_USER.format(context=context, question=user_query)
    )

    sources = format_unique_source_lines(retrieved)

    trace: list[dict[str, Any]] = [
        {
            "node": "fast_pipeline",
            "detail": {"chunks": len(retrieved), "mode": "hybrid_search_then_generate"},
            "usage": {},
            "ts": time.time(),
        },
        {
            "node": "generator",
            "detail": {"mode": "prompt_ready", "sources": sources},
            "usage": {},
            "ts": time.time(),
        },
    ]

    return {
        "generation_messages": [sys, human],
        "sources": sources,
        "trace": trace,
        "workflow_started_at": t0,
    }
