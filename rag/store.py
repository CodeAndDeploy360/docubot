"""ChromaDB wrapper: add, delete by document id, fetch all for BM25 refresh."""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from config import chroma_path

_COLLECTION_NAME = "docubot_docs"
_client: chromadb.PersistentClient | None = None
_index_revision = 0


def index_revision() -> int:
    """Bumps when chunks are added or removed; BM25 cache can skip full reload while this is unchanged."""

    return _index_revision


def _bump_index_revision() -> None:
    global _index_revision
    _index_revision += 1


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        chroma_path().mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(chroma_path()))
    return _client


def get_collection() -> Collection:
    return get_client().get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_document_chunks(
    doc_id: str,
    source_name: str,
    chunks: list[str],
    embeddings: list[list[float]],
    metas: list[dict[str, Any]],
) -> None:
    col = get_collection()
    ids = [f"{doc_id}:{i}" for i in range(len(chunks))]
    documents = chunks
    for m in metas:
        m.setdefault("doc_id", doc_id)
        m.setdefault("source", source_name)
    col.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metas)
    _bump_index_revision()


def delete_document(doc_id: str) -> int:
    """
    Remove every chunk for this doc_id from Chroma.

    Uses ``delete(where=...)`` so all matching rows are removed in one call; the older
    ``get`` + ``delete(ids=...)`` path could miss rows when ``get`` applied a default
    limit, leaving vectors behind so RAG still answered from "deleted" documents.
    """
    col = get_collection()
    doc_id = str(doc_id)
    result = col.delete(where={"doc_id": {"$eq": doc_id}})
    deleted = int(result.get("deleted", 0))
    if deleted:
        _bump_index_revision()
    return deleted


def new_doc_id() -> str:
    return str(uuid.uuid4())


def index_chunk_count() -> int:
    """Rows in the vector collection (persists on disk; may differ from the Streamlit doc list)."""
    return int(get_collection().count())


def list_documents_and_legacy_from_index() -> tuple[list[dict[str, Any]], int]:
    """
    One Chroma scan: sidebar doc list + count of chunks missing ``doc_id`` (legacy).

    Call this only on session init (or after index-clear), not on every Streamlit rerun.
    """
    col = get_collection()
    if col.count() == 0:
        return [], 0
    data = col.get(include=["metadatas"])
    metas: list[dict[str, Any] | None] = data.get("metadatas") or []
    by_doc: dict[str, dict[str, Any]] = {}
    legacy = 0
    for meta in metas:
        if not meta:
            continue
        if not meta.get("doc_id"):
            legacy += 1
            continue
        did = str(meta["doc_id"])
        src = str(meta.get("source", "unknown"))
        if did not in by_doc:
            by_doc[did] = {
                "doc_id": did,
                "name": src,
                "status": "indexed",
                "chunk_count": 0,
                "error": None,
                "audit": None,
            }
        by_doc[did]["chunk_count"] = int(by_doc[did]["chunk_count"]) + 1
        by_doc[did]["name"] = src
    docs = sorted(by_doc.values(), key=lambda d: (d["name"].lower(), d["doc_id"]))
    return docs, legacy


def clear_index() -> int:
    """Remove all chunks from storage. Returns how many rows were removed (0 if none)."""
    client = get_client()
    n = 0
    try:
        col = client.get_collection(_COLLECTION_NAME)
        n = int(col.count())
    except Exception:
        return 0
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        return n
    _bump_index_revision()
    return n
