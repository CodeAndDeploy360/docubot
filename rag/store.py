"""ChromaDB wrapper: per-user persistent clients, add/delete/list, BM25 revision tracking."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from config import data_dir

_COLLECTION_NAME = "docubot_docs"
_thread = threading.local()

# user_id -> PersistentClient
_clients: dict[str, chromadb.PersistentClient] = {}
_index_revision: dict[str, int] = {}
_lock = threading.Lock()


def _user_chroma_dir(user_id: str) -> str:
    return str(data_dir() / "users" / user_id / "chroma")


def set_active_user(user_id: str | None) -> None:
    """Call on the Streamlit thread before any store op; retriever_node sets it on the graph worker thread."""
    _thread.user_id = user_id


def get_active_user() -> str:
    uid = getattr(_thread, "user_id", None)
    if not uid:
        raise RuntimeError("No active user — set_active_user() before using the vector store.")
    return uid


def reset_for_tests() -> None:
    """Clear client cache (pytest / isolated runs)."""
    global _clients, _index_revision
    with _lock:
        _clients = {}
        _index_revision = {}
    _thread.user_id = None
    from rag import retriever

    retriever.clear_hybrid_cache_for_tests()


def index_revision() -> int:
    uid = get_active_user()
    return _index_revision.get(uid, 0)


def _bump_index_revision() -> None:
    uid = get_active_user()
    _index_revision[uid] = _index_revision.get(uid, 0) + 1


def get_client() -> chromadb.PersistentClient:
    uid = get_active_user()
    with _lock:
        if uid not in _clients:
            p = _user_chroma_dir(uid)
            Path(p).mkdir(parents=True, exist_ok=True)
            _clients[uid] = chromadb.PersistentClient(path=p)
        return _clients[uid]


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
    return int(get_collection().count())


def list_documents_and_legacy_from_index() -> tuple[list[dict[str, Any]], int]:
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
