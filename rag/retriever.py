"""Hybrid retrieval: Chroma vector search + BM25 with RRF fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from rag.embedder import embed_texts
from rag.store import get_collection, index_revision


def _tokenize(s: str) -> list[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if t]


def _rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for rank, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float


class HybridRetriever:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._id_to_text: dict[str, str] = {}
        self._id_to_meta: dict[str, dict[str, Any]] = {}
        self._cache_revision: int | None = None

    def refresh(self) -> None:
        col = get_collection()
        data = col.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        self._id_to_text = {i: d for i, d in zip(ids, docs)}
        self._id_to_meta = {i: (metas[idx] or {}) for idx, i in enumerate(ids)}
        tokenized = [_tokenize(self._id_to_text[i]) for i in ids]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, k: int = 8) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        col = get_collection()
        count = col.count()
        if count == 0:
            self._id_to_text.clear()
            self._id_to_meta.clear()
            self._bm25 = None
            self._cache_revision = index_revision()
            return []

        rev = index_revision()
        if self._cache_revision != rev or self._bm25 is None:
            self.refresh()
            self._cache_revision = rev

        vec_k = min(max(k * 2, k), count)
        q_emb = embed_texts([query])[0]
        vres = col.query(query_embeddings=[q_emb], n_results=vec_k, include=["documents", "metadatas", "distances"])
        v_ids = (vres.get("ids") or [[]])[0]

        bm25_ids: list[str] = []
        if self._bm25 is not None and self._id_to_text:
            all_ids = list(self._id_to_text.keys())
            scores = self._bm25.get_scores(_tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            bm25_ids = [all_ids[i] for i in order[:vec_k]]

        fused = _rrf_fuse([v_ids, bm25_ids])
        out: list[RetrievedChunk] = []
        for cid, sc in fused[:k]:
            text = self._id_to_text.get(cid, "")
            meta = dict(self._id_to_meta.get(cid, {}))
            meta["chunk_id"] = cid
            out.append(RetrievedChunk(text=text, metadata=meta, score=float(sc)))
        return out


_default_hybrid = HybridRetriever()


def hybrid_search(query: str, k: int = 8) -> list[RetrievedChunk]:
    return _default_hybrid.search(query, k=k)
