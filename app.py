"""DocuBot — Streamlit entrypoint."""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

import html
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

import streamlit as st

import config
from agent.graph import compile_app_graph
from agent.state import AgentState
from pipeline.ingestor import parse_upload
from rag import store
from rag.ingest import index_parsed_document
from services.llm import stream_messages

try:
    import markdown as md_lib
except ImportError:  # pragma: no cover
    md_lib = None

st.set_page_config(page_title="DocuBot", layout="wide", initial_sidebar_state="expanded")

CUSTOM_CSS = """
<style>
  :root {
    /* Streamlit toolbar (Deploy / menu) — DocuBot band sits below this */
    --docubot-streamlit-header-height: 3.5rem;
    /* Two-line title (DocuBot + subtitle) + padding */
    --docubot-banner-height: 4.55rem;
    --docubot-top-chrome-height: calc(var(--docubot-streamlit-header-height) + var(--docubot-banner-height));
  }
  /* Full-width top band + partition (viewport-fixed; avoids broken layout from main-column fixed positioning) */
  body::before {
    content: "" !important;
    position: fixed !important;
    top: var(--docubot-streamlit-header-height) !important;
    left: 0 !important;
    width: 100vw !important;
    height: var(--docubot-banner-height) !important;
    z-index: 120 !important;
    pointer-events: none !important;
    background: linear-gradient(180deg, #faf9f7 0%, #f3f2f0 100%) !important;
    border-bottom: 1px solid #d0ccc4 !important;
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.07) !important;
    box-sizing: border-box !important;
  }
  /* Streamlit header: same palette as banner (removes white “split” next to DocuBot strip) */
  header[data-testid="stHeader"],
  [data-testid="stHeader"] {
    z-index: 1000010 !important;
    position: relative !important;
    background: linear-gradient(180deg, #faf9f7 0%, #f5f4f2 100%) !important;
    border-bottom: 1px solid transparent !important;
  }
  /* Title row: text only; strip is body::before (always full viewport width) */
  .docubot-app-header {
    position: fixed !important;
    top: var(--docubot-streamlit-header-height) !important;
    left: 0 !important;
    width: 100vw !important;
    max-width: 100vw !important;
    height: var(--docubot-banner-height) !important;
    margin: 0 !important;
    padding: 0.55rem 1rem 0.5rem !important;
    text-align: center !important;
    z-index: 130 !important;
    box-sizing: border-box !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    pointer-events: none !important;
  }
  .docubot-app-header * {
    pointer-events: auto !important;
  }
  .docubot-app-header .docubot-header-title {
    margin: 0 !important;
    padding: 0 !important;
    color: #6b7280 !important;
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    line-height: 1.2 !important;
    letter-spacing: 0.02em !important;
  }
  .docubot-app-header .docubot-header-subtitle {
    margin: 0.2rem 0 0 0 !important;
    padding: 0 !important;
    color: #9ca3af !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    line-height: 1.2 !important;
  }
  /* Push app chrome below toolbar + banner */
  .stApp {
    background-color: #faf9f7 !important;
    padding-top: var(--docubot-top-chrome-height) !important;
    overflow-x: hidden !important;
  }
  /* Sidebar: starts below DocuBot banner — top edge reads as separate from heading */
  section[data-testid="stSidebar"],
  section.stSidebar {
    top: var(--docubot-top-chrome-height) !important;
    height: calc(100dvh - var(--docubot-top-chrome-height)) !important;
    max-height: calc(100dvh - var(--docubot-top-chrome-height)) !important;
    background: linear-gradient(180deg, #ebe9e5 0%, #eeedea 55%, #ebe9e4 100%) !important;
    border-right: 1px solid #e5e3df !important;
    border-top: 1px solid #c9c4bc !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.65) !important;
  }
  /* Hide collapse control for a cleaner chrome, but NEVER hide "expand" — if the sidebar
     gets collapsed (shortcut, session, or edge click), users must still see the chevron. */
  [data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
  }
  [data-testid="stSidebarHeader"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    overflow: hidden !important;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h3,
  section.stSidebar h1,
  section.stSidebar h3 {
    font-size: 0.85rem !important;
    letter-spacing: 0.06em;
    color: #5c5a57 !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    margin-top: 0 !important;
  }
  /* Sidebar content: breathing room below collapse row (clear of DocuBot heading band) */
  section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.5rem !important;
  }
  section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {
    margin-top: 0 !important;
    background: #ffffff !important;
    border-color: #dcd9d4 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.05) !important;
  }
  section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] h3 {
    margin-top: 0 !important;
    margin-bottom: 0.45rem !important;
    padding-top: 0 !important;
  }
  .docubot-main-tagline {
    text-align: center;
    color: #6b7280;
    font-size: 0.9rem;
    margin-top: 0.15rem;
    margin-bottom: 0.75rem;
    padding-left: 1rem;
    padding-right: 1rem;
  }
  /* Keep pinned chat dock in view (avoid extra padding that clips on short viewports / mobile) */
  [data-testid="stBottom"] {
    padding-bottom: max(0.5rem, env(safe-area-inset-bottom, 0px)) !important;
    box-sizing: border-box !important;
    overflow: visible !important;
  }

  /* Chat bubbles: user right (blue), assistant left (off-white) */
  .chat-bubble-user {
    background: #e3f2fd;
    color: #0d47a1;
    padding: 0.9rem 1.1rem;
    border-radius: 14px;
    text-align: right;
    max-width: 92%;
    margin-left: auto;
    margin-bottom: 0.75rem;
    font-size: 0.95rem;
    line-height: 1.45;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .chat-bubble-assistant {
    background: #f7f6f4;
    border: 1px solid #e9e6e1;
    color: #374151;
    padding: 0.9rem 1.1rem;
    border-radius: 14px;
    max-width: 92%;
    margin-bottom: 0.75rem;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }
  .chat-bubble-assistant p { margin: 0.35em 0; }
  .chat-bubble-assistant p:first-child { margin-top: 0; }
  .chat-bubble-assistant p:last-child { margin-bottom: 0; }
  .sources-block { margin-top: 0.85rem; padding-top: 0.65rem; border-top: 1px solid #e0ddd8; }
  .sources-block h4 { font-size: 0.8rem; font-weight: 600; color: #6b7280; margin: 0 0 0.4rem 0; letter-spacing: 0.05em; }
  .sources-block .src-line { font-size: 0.88rem; color: #2563eb; margin: 0.2rem 0; }

  /* Sidebar doc cards */
  .doc-card {
    padding: 0.75rem 1rem;
    border-radius: 10px;
    background: #ffffff;
    margin-bottom: 0.5rem;
    border: 1px solid #e9ecef;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .badge-indexed { background: #d4edda; color: #155724; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; }
  .badge-processing { background: #fff3cd; color: #856404; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; }
  .badge-error { background: #f8d7da; color: #721c24; padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; }

  /* Workflow trace is rendered in bottom container (only expander in this app) */
  .stApp [data-testid="stExpander"] {
    margin-top: 0 !important;
    margin-bottom: 0.15rem !important;
    padding: 0.2rem 0 0 !important;
    border-top: 1px solid #e5e3df !important;
    background: #faf9f7 !important;
  }
  .stApp [data-testid="stExpander"] details {
    margin-bottom: 0 !important;
  }
  .stApp [data-testid="stExpander"] summary {
    padding-top: 0.1rem !important;
    padding-bottom: 0.1rem !important;
  }

  /* Live streaming reply: bordered block (matches assistant bubble look) */
  .stApp [data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    border-color: #e9e6e1 !important;
    background: #f7f6f4 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
  }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "docs" not in st.session_state:
        # One Chroma scan: doc list + legacy chunk count (do not rescan every rerun — that freezes the UI).
        docs, legacy = store.list_documents_and_legacy_from_index()
        st.session_state.docs = docs
        st.session_state.legacy_chunk_count = legacy
    elif "legacy_chunk_count" not in st.session_state:
        _, st.session_state.legacy_chunk_count = store.list_documents_and_legacy_from_index()
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = []
    if "last_workflow_ms" not in st.session_state:
        st.session_state.last_workflow_ms = 0.0
    if "last_gen_usage" not in st.session_state:
        st.session_state.last_gen_usage = {}


def _chat_history_tuples() -> list[tuple[str, str]]:
    hist: list[tuple[str, str]] = []
    pending_user: str | None = None
    for m in st.session_state.messages:
        if m["role"] == "user":
            pending_user = m["content"]
        elif m["role"] == "assistant" and pending_user is not None:
            hist.append((pending_user, m["content"]))
            pending_user = None
    return hist


_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _enqueue_upload_from_file(uploaded) -> bool:
    """Store file bytes in session and show PROCESSING immediately; indexing runs on the next run(s)."""
    if uploaded.size > _MAX_UPLOAD_BYTES:
        st.error("File exceeds 10 MB limit.")
        return False
    if any(
        d["name"] == uploaded.name and d["status"] in ("indexed", "processing")
        for d in st.session_state.docs
    ):
        return False
    raw = uploaded.getvalue()
    doc_id = str(uuid.uuid4())
    st.session_state.docs.append(
        {
            "doc_id": doc_id,
            "name": uploaded.name,
            "status": "processing",
            "chunk_count": None,
            "error": None,
            "audit": None,
            "_pending_bytes": raw,
        }
    )
    return True


def _has_pending_ingest() -> bool:
    return any(
        d.get("status") == "processing" and d.get("_pending_bytes") is not None
        for d in st.session_state.docs
    )


def _drain_pending_ingest_one() -> None:
    """Finish indexing for the first queued upload, then rerun so the uploader stays clear and status updates."""
    for doc in st.session_state.docs:
        if doc.get("status") != "processing":
            continue
        raw = doc.pop("_pending_bytes", None)
        if raw is None:
            continue
        try:
            parsed = parse_upload(doc["name"], raw)
            result = index_parsed_document(parsed, source_name=doc["name"], doc_id=str(doc["doc_id"]))
            doc["status"] = "indexed"
            doc["chunk_count"] = result.chunk_count
            doc["audit"] = result.audit
        except Exception as exc:  # noqa: BLE001 — surface in UI
            doc["status"] = "error"
            doc["error"] = str(exc)
            try:
                store.delete_document(str(doc["doc_id"]))
            except Exception:
                pass
        st.rerun()
        return


def _sources_html(sources: list[str]) -> str:
    if not sources:
        return ""
    lines = "".join(
        f'<div class="src-line">📄 {html.escape(s)}</div>' for s in sources
    )
    return f'<div class="sources-block"><h4>Sources</h4>{lines}</div>'


def _user_bubble_html(text: str) -> str:
    safe = html.escape(text).replace("\n", "<br/>")
    return f'<div class="chat-bubble-user">{safe}</div>'


def _assistant_body_html(markdown_text: str) -> str:
    if md_lib:
        return md_lib.markdown(
            markdown_text,
            extensions=["nl2br", "fenced_code", "tables"],
        )
    return f"<p>{html.escape(markdown_text).replace(chr(10), '<br/>')}</p>"


def _assistant_bubble_html(content: str, sources: list[str] | None = None) -> str:
    body = _assistant_body_html(content)
    return f'<div class="chat-bubble-assistant">{body}{_sources_html(sources or [])}</div>'


def _render_chat_row(
    role: str,
    content: str,
    sources: list[str] | None = None,
) -> None:
    """User bubble right-aligned; assistant bubble left-aligned (mockup layout)."""
    if role == "user":
        spacer, col = st.columns([1, 2])
        with spacer:
            st.empty()
        with col:
            st.markdown(_user_bubble_html(content), unsafe_allow_html=True)
        return
    col, spacer = st.columns([2, 1])
    with col:
        st.markdown(_assistant_bubble_html(content, sources), unsafe_allow_html=True)
    with spacer:
        st.empty()


def _render_sources_block(sources: list[str]) -> None:
    """Sources below streaming bubble (same styling as history)."""
    if not sources:
        return
    st.markdown(_sources_html(sources), unsafe_allow_html=True)


def _format_exception_for_markdown(exc: Exception, *, max_chars: int = 24_000) -> str:
    """Full exception text for display (not a substitute for the real API message)."""
    raw = str(exc)
    if len(raw) > max_chars:
        return raw[:max_chars] + "\n\n… [truncated for UI length]"
    return raw


def _fenced_api_error(text: str) -> str:
    """Put raw API / exception text in a markdown fence; avoid breaking the fence if the message contains ```."""
    safe = text.replace("```", "`\u200b``")
    return f"```text\n{safe}\n```"


def _format_chat_error(exc: Exception) -> str:
    raw = _format_exception_for_markdown(exc)
    low = raw.lower()
    exact = _fenced_api_error(raw)

    if "429" in raw or "resource_exhausted" in low or ("quota" in low and "exceed" in low):
        return (
            "**429 — quota / rate limit (from the API)**\n\n"
            f"{exact}\n\n"
            "**Tips**\n"
            "- If the message says **retry in Ns**, wait and try again.\n"
            "- The LangGraph workflow makes **several** `generateContent` calls per question; ensure "
            "your quota or billing supports that, or use **`DOCUBOT_LLM_PROVIDER=openai`** with a suitable model.\n"
            "- Try another model in `.env`, e.g. **`DOCUBOT_CHAT_MODEL=gemini-2.0-flash`** "
            "([AI Studio → Models](https://aistudio.google.com/)).\n"
            "- Enable billing on the Google Cloud project for this key, or use **`DOCUBOT_LLM_PROVIDER=openai`** "
            "with `OPENAI_API_KEY`.\n"
            "- Docs: https://ai.google.dev/gemini-api/docs/rate-limits"
        )
    if "404" in raw and "not found" in low and "model" in low:
        return (
            "**Model not found (from the API)**\n\n"
            f"{exact}\n\n"
            "**Tips**\n"
            "- Pick a model id your key supports under [AI Studio → Models](https://aistudio.google.com/) and set "
            "**`DOCUBOT_CHAT_MODEL`** in `.env`."
        )
    if "timeout" in low:
        return f"**Timeout**\n\n{exact}"

    return f"**Error**\n\n{exact}"


def _remove_doc(doc: dict[str, Any]) -> None:
    doc_id = str(doc["doc_id"])
    try:
        deleted = store.delete_document(doc_id)
    except Exception as exc:  # noqa: BLE001 — show in UI, keep list in sync with DB
        st.error(f"Could not remove this document from the index: {exc}")
        return
    if doc.get("status") == "indexed" and (doc.get("chunk_count") or 0) > 0 and deleted == 0:
        st.warning(
            "Nothing was deleted in the vector store (0 chunks). If questions still cite this file, "
            "chunks may lack `doc_id` metadata (legacy index). Clear `DOCUBOT_CHROMA_PATH` or re-upload."
        )
    st.session_state.docs = [d for d in st.session_state.docs if d["doc_id"] != doc_id]
    # New file_uploader instance so it does not keep showing a deleted file as "selected".
    st.session_state.uploader_key += 1


def _reset_file_uploader() -> None:
    st.session_state.uploader_key += 1
    st.rerun()


_init_session()

if config.llm_provider() == "gemini" and not config.gemini_api_key():
    st.error(
        "**Missing Gemini API key.** Add `GEMINI_API_KEY` or `GOOGLE_API_KEY` to your `.env` in the project root "
        "(copy `.env.example` → `.env`, then paste your key). For Streamlit Cloud, add the same key under **Secrets**. "
        "Restart the app after saving."
    )
elif config.llm_provider() == "openai" and not (os.getenv("OPENAI_API_KEY") or "").strip():
    st.error(
        "**Missing OpenAI API key.** Set `OPENAI_API_KEY` in `.env` or Streamlit **Secrets**, then restart the app."
    )

# Full-width banner (fixed CSS) — title spans entire viewport; sidebar starts below via --docubot-banner-height
st.markdown(
    '<div class="docubot-app-header" role="banner">'
    '<p class="docubot-header-title">DocuBot</p>'
    '<p class="docubot-header-subtitle">Document Q&amp;A Assistant</p>'
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="docubot-main-tagline">Upload documents, ask questions, get answers with source citations.</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    with st.container(border=True):
        st.markdown("### Upload documents")
        uploads = st.file_uploader(
            "Add files",
            type=["pdf", "txt", "csv"],
            accept_multiple_files=True,
            key=f"doc_uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
            help="PDF, TXT, or CSV · max 10 MB per file. Files appear in the list as PROCESSING, then INDEXED.",
        )
    if uploads:
        enqueued = False
        for f in uploads:
            if _enqueue_upload_from_file(f):
                enqueued = True
        if enqueued:
            # Clear the picker immediately so large CSV/PDF work does not keep the file “stuck” in the drop zone.
            _reset_file_uploader()

    st.markdown("### Indexed documents")
    st.caption("PDF, TXT, or CSV · max 10 MB per file")
    legacy = int(st.session_state.get("legacy_chunk_count", 0))
    if legacy:
        st.warning(
            f"**{legacy} chunk(s)** in the index have no document id (legacy data). "
            "Per-file delete may not remove them — use **Remove all vectors** below or delete the `vector_db` folder."
        )
    if not st.session_state.docs:
        st.info("No documents yet.")
    for doc in list(st.session_state.docs):
        status = doc["status"]
        badge_class = f"badge-{status}"
        label = status.upper()
        chunks = doc.get("chunk_count")
        chunk_line = f"{chunks} chunks" if chunks is not None else "—"
        st.markdown(
            f'<div class="doc-card"><strong>{doc["name"]}</strong><br/>'
            f'<span class="{badge_class}">{label}</span> · {chunk_line}</div>',
            unsafe_allow_html=True,
        )
        if doc.get("error"):
            st.caption(doc["error"])
        if st.button("Delete", key=f"del-{doc['doc_id']}"):
            _remove_doc(doc)
            st.rerun()

    # Visible reset (no nested expander). Do not call store.index_chunk_count() here — it runs every
    # Streamlit rerun and slowed Q&A when the Advanced section briefly used it.
    _has_clearable_index = st.session_state.get("legacy_chunk_count", 0) > 0 or any(
        d.get("status") == "indexed" for d in st.session_state.docs
    )
    if _has_clearable_index:
        if st.button("Remove all vectors", key="remove_all_vectors", help="Clear the entire Chroma index for this app"):
            store.clear_index()
            st.session_state.docs = []
            st.session_state.legacy_chunk_count = 0
            st.success("All vectors removed.")
            st.rerun()

    # After the list is drawn so PROCESSING rows are visible; then index one queued file per run.
    if _has_pending_ingest():
        _drain_pending_ingest_one()

for msg in st.session_state.messages:
    _render_chat_row(
        msg["role"],
        msg["content"],
        msg.get("sources") if msg["role"] == "assistant" else None,
    )

# Pin workflow trace in Streamlit’s bottom region directly above the fixed chat input.
with st._bottom:
    with st.expander("▶ 🔍 Show workflow trace"):
        st.write(
            {
                "workflow_ms": round(st.session_state.last_workflow_ms, 2),
                "generation_usage": st.session_state.last_gen_usage,
                "trace": st.session_state.last_trace,
            }
        )

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    _render_chat_row("user", prompt)

    left_col, _right_sp = st.columns([2, 1])
    with left_col:
        answer = ""
        sources: list[str] = []
        trace: list[dict[str, Any]] = []
        try:
            t0 = time.perf_counter()
            graph = compile_app_graph()
            initial: AgentState = {
                "user_query": prompt,
                "chat_history": _chat_history_tuples(),
                "replan_count": 0,
                "replan_note": "",
                "trace": [],
                "workflow_started_at": t0,
            }
            spin = (
                "Looking through your documents and writing an answer. "
                "This usually takes under two minutes; thanks for your patience."
            )
            timeout_sec = config.graph_invoke_timeout_sec()
            with st.spinner(spin):
                with ThreadPoolExecutor(max_workers=1) as pool:
                    fut = pool.submit(graph.invoke, initial)
                    try:
                        final = fut.result(timeout=timeout_sec)
                    except FuturesTimeout as exc:
                        raise TimeoutError(
                            f"The workflow did not finish within {timeout_sec:.0f}s. "
                            "Check your network, Gemini/OpenAI status, and API keys—or raise "
                            "`DOCUBOT_GRAPH_TIMEOUT_SEC` in `.env`."
                        ) from exc
            st.session_state.last_workflow_ms = (time.perf_counter() - t0) * 1000.0
            msgs = final.get("generation_messages") or []
            sources = list(final.get("sources") or [])
            trace = list(final.get("trace") or [])
            gen_usage: dict[str, Any] = {}

            def token_stream():
                for piece, meta in stream_messages(msgs):
                    if meta:
                        gen_usage.update(meta)
                    if piece:
                        yield piece

            with st.container(border=True):
                answer = str(st.write_stream(token_stream()))
            if not answer.strip():
                answer = (
                    "*No text was returned from the model.* Check the workflow trace, API quotas, "
                    "and whether `DOCUBOT_CHAT_MODEL` is valid for your Gemini account."
                )
            st.session_state.last_gen_usage = gen_usage
            trace = trace + [
                {
                    "node": "generator_stream",
                    "detail": {"usage": gen_usage},
                    "usage": {},
                    "ts": time.time(),
                }
            ]
        except Exception as exc:  # noqa: BLE001
            answer = _format_chat_error(exc)
            trace = [{"node": "error", "detail": {"message": str(exc)}}]
            with st.container(border=True):
                st.markdown(answer)
        st.session_state.last_trace = trace
        if sources:
            _render_sources_block(sources)
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
    st.rerun()
