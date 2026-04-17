"""Embedding generation (OpenAI or local sentence-transformers)."""

from __future__ import annotations

import os
import threading
from typing import Any, Sequence

from config import embedding_model, embedding_provider

# Reuse clients across Streamlit reruns and chat turns — constructing new LangChain
# embedding objects every call adds TLS/connection overhead and noticeably slows Q&A.
_openai_emb: Any = None
_local_emb: Any = None
_emb_lock = threading.Lock()


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    texts = list(texts)
    if not texts:
        return []
    provider = embedding_provider()
    if provider == "local":
        return _embed_local(texts)
    return _embed_openai(texts)


def _get_openai_embeddings():
    global _openai_emb
    from langchain_openai import OpenAIEmbeddings

    with _emb_lock:
        if _openai_emb is None:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("OPENAI_API_KEY is required for OpenAI embeddings.")
            _openai_emb = OpenAIEmbeddings(model=embedding_model(), api_key=key)
        return _openai_emb


def _embed_openai(texts: list[str]) -> list[list[float]]:
    emb = _get_openai_embeddings()
    if len(texts) == 1:
        return [emb.embed_query(texts[0])]
    return emb.embed_documents(texts)


def _get_local_embeddings():
    global _local_emb
    from langchain_community.embeddings import HuggingFaceEmbeddings

    with _emb_lock:
        if _local_emb is None:
            model_name = os.getenv("DOCUBOT_LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            _local_emb = HuggingFaceEmbeddings(model_name=model_name)
        return _local_emb


def _embed_local(texts: list[str]) -> list[list[float]]:
    return _get_local_embeddings().embed_documents(texts)
