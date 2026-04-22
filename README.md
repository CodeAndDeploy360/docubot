# DocuBot

DocuBot is a training project: a single-page **Streamlit** app where you upload **PDF, TXT, or CSV** files, index them into **ChromaDB**, and ask questions in a chat UI. Answers are produced by a **LangGraph** workflow (Planner → Retriever → Grader → Generator prompt + streamed LLM) with **hybrid retrieval** (vector + BM25) and **source citations**.

**Project direction:** chat uses **Google Gemini** (free tier / AI Studio); **embeddings** use **OpenAI**; **web search** is a **real** integration (DuckDuckGo by default, optional **Tavily** for production); the repo is **deployment-ready** (Docker + env conventions).

## Quick start

1. **Python 3.11+** recommended.
2. Create a virtual environment and install dependencies:

   ```bash
   cd docubot
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set:
   - **`GEMINI_API_KEY`** (or **`GOOGLE_API_KEY`**) — [Google AI Studio](https://aistudio.google.com/apikey)
   - **`OPENAI_API_KEY`** — required for **embeddings** (`text-embedding-3-small`)

4. Run the app:

   ```bash
   PYTHONPATH=. streamlit run app.py
   ```

5. Open the URL Streamlit prints (usually http://localhost:8501).

## Configuration

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini chat (default provider). |
| `OPENAI_API_KEY` | **Embeddings** (and OpenAI chat if you switch provider). |
| `DOCUBOT_LLM_PROVIDER` | `gemini` (default), `openai`, or `ollama`. |
| `DOCUBOT_CHAT_MODEL` | e.g. `gemini-2.5-flash-lite`, `gemini-2.5-flash`, or `gemini-2.0-flash` (see [Gemini models](https://ai.google.dev/gemini-api/docs/models); `gemini-1.5-flash` often returns **404** on current Developer API). |
| `DOCUBOT_EMBEDDING_MODEL` | Default `text-embedding-3-small`. |
| `DOCUBOT_EMBEDDING_PROVIDER` | `openai` (default). |
| `DOCUBOT_WEB_SEARCH_BACKEND` | `duckduckgo` (default, live HTML search) or `tavily` (needs `TAVILY_API_KEY`). |
| `TAVILY_API_KEY` | Required when `DOCUBOT_WEB_SEARCH_BACKEND=tavily`. |
| `DOCUBOT_OLLAMA_BASE_URL` | When using Ollama for chat. |
| `DOCUBOT_CHROMA_PATH` | Chroma persistence directory (default `./vector_db`; Docker image uses `/data/chroma`). |
| `DOCUBOT_PII_REDACTION_ENABLED` | `true` / `false` for regex PII masking during cleaning. |
| `DOCUBOT_MAX_SOURCES_DISPLAY` | Max source lines under each answer (default **3**). |
| `DOCUBOT_RETRIEVER_K` | Chunks retrieved per question for hybrid search (default **8**). |

### Time-related settings (`config.py`, `services/llm.py`, `app.py`, `agent/tools.py`)

| Variable | Default | Where it applies |
|----------|---------|------------------|
| `DOCUBOT_LLM_TIMEOUT_SEC` | `120` | **HTTP timeout** for a **single** chat completion request to Gemini / OpenAI / Ollama (`build_chat_model` in `services/llm.py`). Each planner, grader, streamed final answer, etc. uses this per call. |
| `DOCUBOT_GRAPH_TIMEOUT_SEC` | `900` | **Wall-clock cap** for the **entire** `graph.invoke` in the UI (`ThreadPoolExecutor` + `fut.result(timeout=…)` in `app.py`). Covers planner → retriever → grader → replans → generator **prep**; streaming the final answer runs **after** this returns. Increase if you hit the limit on slow networks or when **429 retries** add sleep time (below). |
| `DOCUBOT_WEB_SEARCH_TIMEOUT_SEC` | `25` | Max wait for the **web_search** tool’s HTTP work (`agent/tools.py`); on timeout the tool returns a JSON `error` string instead of hanging. |
| `DOCUBOT_LLM_RETRY_ATTEMPTS` | `5` | Max **attempts** (including the first) for **rate-limit / quota** style LLM failures in `invoke_with_rate_limit_retry` and streaming retries (`services/llm.py`). Other errors are **not** retried. |

### Time limits, retries, and waits (behavior)

- **Per-request LLM timeout:** Each `generateContent`-style call uses `DOCUBOT_LLM_TIMEOUT_SEC`. If the provider hangs, the client raises a timeout (shown in the chat via `_format_chat_error`).
- **Whole LangGraph run:** The Streamlit app runs `graph.invoke` in a background thread and aborts after `DOCUBOT_GRAPH_TIMEOUT_SEC`. That is independent of per-call HTTP timeouts: one turn can issue **several** LLM calls (planner, grader, …), each subject to `DOCUBOT_LLM_TIMEOUT_SEC`, but the **sum** of work must finish within the graph cap.
- **Rate limits (429, `RESOURCE_EXHAUSTED`, quota messages):** `services/llm.py` retries up to `DOCUBOT_LLM_RETRY_ATTEMPTS`. Between attempts it **sleeps** using `sleep_rate_limit_backoff`: prefers the API’s **`retry in Ns`** hint when present, otherwise about **15s**, capped at **~120s** plus a small random jitter. This **wait time counts toward** wall time, so a busy session can approach `DOCUBOT_GRAPH_TIMEOUT_SEC` even when each HTTP call is “fast.”
- **Streaming:** If streaming yields no tokens, the code can fall back to a non-streaming `invoke` (also behind rate-limit retry). Streaming retries **restart the stream** on rate-limit, not token-by-token.
- **Web search:** Only the tool path uses `DOCUBOT_WEB_SEARCH_TIMEOUT_SEC`; RAG retrieval is local (Chroma + BM25) and not governed by these HTTP knobs.
- **Embeddings (indexing / retrieval):** OpenAI embedding calls do **not** set an explicit timeout in this repo; behavior follows LangChain / OpenAI client defaults unless you extend `rag/embedder.py`.

**Security:** never commit `.env` or paste real API keys into `.env.example`. If a key was ever committed, **rotate it** in the provider console.

## Deployment

### Docker

Build and run (mount a volume so indexed documents survive restarts):

```bash
docker build -t docubot .
docker run --rm -p 8501:8501 \
  --env-file .env \
  -v docubot_chroma:/data/chroma \
  docubot
```

The image binds Streamlit to `0.0.0.0` and uses **`PORT`** if set (common on PaaS), otherwise **`STREAMLIT_SERVER_PORT`**, otherwise **8501**.

### Checklist for a hosted environment

- Set **secrets** / env vars: `GEMINI_API_KEY`, `OPENAI_API_KEY`, and optionally `TAVILY_API_KEY`.
- Provide a **persistent disk** or volume for `DOCUBOT_CHROMA_PATH` (or accept re-indexing on each deploy).
- Ensure outbound HTTPS to Google, OpenAI, and your chosen web-search backend.
- Size the instance for **Chroma + Streamlit** (small CPU/RAM is usually enough for demos).

## Architecture (high level)

1. **Upload** → `pipeline/ingestor.py` extracts text (PDF via pdfplumber / pypdf, CSV normalized, TXT with encoding detection).
2. **Clean** → `pipeline/cleaner.py` five stages (in order): **encoding fix** (UTF-8 / chardet / BOM), **text cleanup** (control chars, whitespace), **PII redaction** (regex, optional), **deduplication** (exact lines), **quality check** (score + audit). Per-stage logs are stored on the cleaning report.
3. **Chunk** → `pipeline/chunker.py` splits on **token** boundaries (defaults **500** tokens, **50** overlap via `chunk_text_by_tokens()`; adjust in code / call sites — not env-driven today).
4. **Index** → `rag/embedder.py` embeds chunks (**OpenAI**); `rag/store.py` writes to Chroma with metadata (`source`, `page`, `chunk_index`, `doc_id`).
5. **Retrieve** → `rag/retriever.py` runs **cosine similarity** on embeddings and **BM25** over the same corpus, then **RRF**-style fusion of ranks.
6. **Agent** → `agent/graph.py`: **Planner** (structured plan), **Retriever** (tools), **Grader**, up to **two** re-plan loops, then **Generator** prep. The UI **streams** the final answer via `services/llm.py` (**Gemini** by default).
7. **UI** → `app.py`: sidebar upload + document status + delete; chat; **workflow trace**; **Sources** under assistant replies when chunks exist.

## Training spec deliverables (checklist)

| # | Requirement | Covered? | Where |
|---|-------------|------------|--------|
| 1 | Data cleaning pipeline — **5 stages** | Yes | `pipeline/cleaner.py` |
| 2 | File parser — **PDF, TXT, CSV** | Yes | `pipeline/ingestor.py` (pdfplumber / pypdf, chardet, CSV normalization) |
| 3 | Chunking — **size + overlap** | Yes (defaults 500 / 50 tokens) | `pipeline/chunker.py` · wired in `rag/ingest.py` |
| 4 | **ChromaDB** + embeddings + indexing | Yes | `rag/embedder.py`, `rag/store.py`, `rag/ingest.py` |
| 5 | **Hybrid** retrieval (vector + **BM25**) | Yes | `rag/retriever.py` (RRF-style fusion) |
| 6 | **LLM** wrapper — prompts, **streaming**, errors | Yes | `services/llm.py`, prompts in `agent/prompts.py`, stream in `app.py` |
| 7 | **LangChain tools** — RAG + web search | Yes | `agent/tools.py` (`knowledge_base_search`, `web_search`) |
| 8 | **LangGraph** — planner → retriever → grader → generator + **re-plan** | Yes | `agent/graph.py` (max 2 replan loops) |
| 9 | **Streamlit** — sidebar, chat, **workflow trace** | Yes | `app.py`, `.streamlit/config.toml` |
| 10 | **Source citations** on agent responses | Yes | `format_unique_source_lines` + Sources UI (`app.py`); inline `[S1]`-style in generator prompt |
| 11 | **Unit tests** — cleaner **≥80%** | Yes (measured ~90% on `pipeline/cleaner.py`) | `tests/test_cleaner.py` — run `pytest tests/test_cleaner.py --cov=pipeline.cleaner` |
| 12 | **README** — setup + architecture | Yes | This file (Quick start, Configuration, Deployment, Architecture, Tests) |

## Test data

Under `test_data/`:

- `sample.pdf` — short revenue snippet (committed binary for demos; recreate locally if you replace it).
- `quarterly_report.txt`, `messy_data.csv`, `pii_test.txt` — demo scenarios.

## Tests

The training spec expects three test modules (plus shared fixtures in `tests/conftest.py`):

| File | Scope |
|------|--------|
| `tests/test_cleaner.py` | Cleaning pipeline (PII includes **`test_data/pii_test.txt`**) |
| `tests/test_retriever.py` | Hybrid retrieval, store CRUD, **`test_data`** demo fixtures |
| `tests/test_workflow.py` | LangGraph routing, compile, **re-plan trace (stubbed)**, prompts, source-line helpers |

```bash
PYTHONPATH=. pytest tests/ -v --cov=pipeline.cleaner --cov-report=term-missing
```

Cleaner module is targeted for **≥80%** line coverage in `test_cleaner.py`.

### Review demo scenarios (manual UI)

| # | Try this | Automated overlap |
|---|----------|-------------------|
| 1 | Upload `test_data/sample.pdf` or `quarterly_report.txt`, ask *“What was the total revenue?”* | `test_retriever.test_hybrid_search_ranks_chunks`, `test_retriever.test_demo_data_fixtures_present` |
| 2 | Upload `pii_test.txt`, ask about contact info | `test_cleaner.test_pii_test_txt_fixture_redacts` |
| 3 | Upload `messy_data.csv`, ask about the rows | `test_retriever.test_demo_data_fixtures_present` |
| 4 | Ask a **vague or multi-part** question → **workflow trace** should show **`replan`** (grader may loop back to the planner) | `test_workflow.test_replan_loop_trace_with_stubbed_nodes` |
| 5 | Unrelated question vs indexed docs only | `test_workflow.test_generator_prompts_discourage_hallucination_when_context_missing` |

Full **LangGraph + Gemini** in the UI needs valid **`.env`** keys; automated tests avoid live LLM calls except where stubbed.

## Local embeddings (optional)

The stack uses **OpenAI embeddings**. A `local` embedding mode is possible with extra installs; see older notes in git history or extend `rag/embedder.py`.

## Notes

- **Console tracebacks mentioning `torchvision` / `transformers`:** These are **not** DocuBot bugs. Streamlit’s **file watcher** inspects loaded packages; optional `transformers` submodules expect `torchvision`, which this project does not install. The app still runs. The repo sets **`fileWatcherType = "none"`** in `.streamlit/config.toml` to keep the console quiet (you lose automatic reload-on-save for code changes; refresh the browser or switch to `auto` if you prefer reload and can ignore the noise).

- **Gemini** structured outputs are used for Planner / Grader via LangChain’s `with_structured_output`; if a model ID errors, try `gemini-1.5-flash` or check current model names in Google AI Studio.
- **Streaming** uses the active chat provider (**Gemini** or **OpenAI**) for the final generation call in `app.py`.
- **Re-plan cap**: at most **two** `bump_replan` cycles before the graph proceeds to generation.
