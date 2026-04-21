"""Hybrid retriever tests (Chroma + BM25) with mocked embeddings (spec: RAG retrieval)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from unittest.mock import patch

from rag import store
from rag.retriever import HybridRetriever


def _fake_embed(texts: list[str]) -> list[list[float]]:
    dim = 12
    return [[float((j + i + 1) % 7) / 7.0 for j in range(dim)] for i in range(len(texts))]


def test_hybrid_search_ranks_chunks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    doc_id = store.new_doc_id()
    chunks = [
        "Acme quarterly revenue was four point two million dollars.",
        "Employee picnic scheduled for May. No financial data here.",
    ]
    metas = [
        {"source": "report.pdf", "page": 1, "chunk_index": 0, "doc_id": doc_id},
        {"source": "report.pdf", "page": 2, "chunk_index": 1, "doc_id": doc_id},
    ]
    with patch("rag.retriever.embed_texts", side_effect=_fake_embed):
        embs = _fake_embed(chunks)
        store.add_document_chunks(doc_id, "report.pdf", chunks, embs, metas)

    with patch("rag.retriever.embed_texts", side_effect=_fake_embed):
        r = HybridRetriever()
        hits = r.search("What was the revenue?", k=2)

    assert len(hits) >= 1
    assert any("revenue" in h.text.lower() for h in hits)


def test_delete_document_removes_chunks():
    doc_id = store.new_doc_id()
    chunks = ["only chunk"]
    metas = [{"source": "a.txt", "page": -1, "chunk_index": 0, "doc_id": doc_id}]
    with patch("rag.retriever.embed_texts", side_effect=_fake_embed):
        embs = _fake_embed(chunks)
        store.add_document_chunks(doc_id, "a.txt", chunks, embs, metas)
    n = store.delete_document(doc_id)
    assert n == 1
    col = store.get_collection()
    assert col.count() == 0


def test_list_documents_and_legacy_from_index():
    doc_id = store.new_doc_id()
    chunks = ["a", "legacy"]
    metas = [
        {"source": "sample.pdf", "page": 1, "chunk_index": 0, "doc_id": doc_id},
        {"source": "old.bin", "page": -1, "chunk_index": 0},
    ]
    with patch("rag.retriever.embed_texts", side_effect=_fake_embed):
        embs = _fake_embed(chunks)
        store.add_document_chunks(doc_id, "sample.pdf", [chunks[0]], [embs[0]], [metas[0]])
        col = store.get_collection()
        col.add(
            ids=["legacy-only:0"],
            documents=[chunks[1]],
            embeddings=[embs[1]],
            metadatas=[metas[1]],
        )
    docs, legacy = store.list_documents_and_legacy_from_index()
    assert legacy == 1
    assert len(docs) == 1
    assert docs[0]["chunk_count"] == 1


def test_list_documents_and_legacy_groups_chunks():
    doc_id = store.new_doc_id()
    chunks = ["a", "b", "c"]
    metas = [
        {"source": "sample.pdf", "page": 1, "chunk_index": i, "doc_id": doc_id}
        for i in range(3)
    ]
    with patch("rag.retriever.embed_texts", side_effect=_fake_embed):
        embs = _fake_embed(chunks)
        store.add_document_chunks(doc_id, "sample.pdf", chunks, embs, metas)
    docs, _legacy = store.list_documents_and_legacy_from_index()
    assert len(docs) == 1
    assert docs[0]["doc_id"] == doc_id
    assert docs[0]["name"] == "sample.pdf"
    assert docs[0]["status"] == "indexed"
    assert docs[0]["chunk_count"] == 3


def test_clear_index_empties_collection():
    doc_id = store.new_doc_id()
    chunks = ["x", "y"]
    metas = [
        {"source": "a.txt", "page": -1, "chunk_index": 0, "doc_id": doc_id},
        {"source": "a.txt", "page": -1, "chunk_index": 1, "doc_id": doc_id},
    ]
    with patch("rag.retriever.embed_texts", side_effect=_fake_embed):
        embs = _fake_embed(chunks)
        store.add_document_chunks(doc_id, "a.txt", chunks, embs, metas)
    assert store.index_chunk_count() == 2
    n = store.clear_index()
    assert n == 2
    assert store.index_chunk_count() == 0


def test_delete_document_removes_all_chunks_large_doc():
    """Regression: delete must remove every chunk, not only the first get() page."""
    doc_id = store.new_doc_id()
    n_chunks = 120
    chunks = [f"segment {i} about revenue and picnics" for i in range(n_chunks)]
    metas = [
        {"source": "big.pdf", "page": i, "chunk_index": i, "doc_id": doc_id} for i in range(n_chunks)
    ]
    with patch("rag.retriever.embed_texts", side_effect=_fake_embed):
        embs = _fake_embed(chunks)
        store.add_document_chunks(doc_id, "big.pdf", chunks, embs, metas)
    assert store.get_collection().count() == n_chunks
    n = store.delete_document(doc_id)
    assert n == n_chunks
    assert store.get_collection().count() == 0


_TEST_DATA = Path(__file__).resolve().parent.parent / "test_data"


def test_demo_data_fixtures_present():
    """Spec review assets: revenue snippet, messy CSV ragged row, PDF sample exist."""
    quarterly = (_TEST_DATA / "quarterly_report.txt").read_text(encoding="utf-8").lower()
    assert "revenue" in quarterly and "4.2" in quarterly
    messy = (_TEST_DATA / "messy_data.csv").read_text(encoding="utf-8")
    assert "Alice" in messy and "Carol" in messy
    rows = list(csv.reader(io.StringIO(messy)))
    assert rows[0][0] == "Name" and any("Carol" in r for r in rows)
    pdf = _TEST_DATA / "sample.pdf"
    assert pdf.is_file() and pdf.stat().st_size > 0
