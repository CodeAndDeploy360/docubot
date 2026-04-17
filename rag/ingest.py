"""Orchestrate parse → clean → chunk → dedupe → embed → Chroma."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from pipeline.chunker import chunk_text_by_tokens
from pipeline.cleaner import CleaningReport, clean_text
from pipeline.ingestor import ParsedDocument
from rag import store
from rag.embedder import embed_texts


@dataclass
class IndexingResult:
    doc_id: str
    chunk_count: int
    audit: dict[str, Any]


def _dedupe_chunks(chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    removed = 0
    for c in chunks:
        h = hashlib.sha256(c["text"].encode("utf-8")).hexdigest()
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        kept.append(c)
    return kept, removed


def index_parsed_document(
    parsed: ParsedDocument,
    *,
    source_name: str,
    doc_id: str | None = None,
) -> IndexingResult:
    doc_id = doc_id or store.new_doc_id()
    all_reports: list[CleaningReport] = []
    raw_chunks: list[dict[str, Any]] = []

    for seg_text, page in parsed.segments:
        report = clean_text(seg_text, dedupe_exact_chunks=True)
        all_reports.append(report)
        text = report.final_text
        if not text.strip():
            continue
        parts = chunk_text_by_tokens(text)
        for idx, part in enumerate(parts):
            meta: dict[str, Any] = {
                "source": source_name,
                "page": page if page is not None else -1,
                "chunk_index": len(raw_chunks) + idx,
            }
            raw_chunks.append({"text": part, "meta": meta})

    # Re-number chunk_index after merge
    merged: list[dict[str, Any]] = []
    offset = 0
    for c in raw_chunks:
        m = dict(c["meta"])
        m["chunk_index"] = offset
        offset += 1
        merged.append({"text": c["text"], "meta": m})

    merged, dup_removed = _dedupe_chunks(merged)
    for i, c in enumerate(merged):
        c["meta"]["chunk_index"] = i
    texts = [c["text"] for c in merged]
    metas = [c["meta"] for c in merged]
    embeddings = embed_texts(texts) if texts else []
    if texts:
        store.add_document_chunks(doc_id, source_name, texts, embeddings, metas)

    audit = {
        "stages": [r.to_dict() for r in all_reports],
        "duplicate_chunks_removed": dup_removed,
        "indexed_chunks": len(texts),
    }
    return IndexingResult(doc_id=doc_id, chunk_count=len(texts), audit=audit)
