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

3. Copy **`.env.example`** to **`.env`** and set at least:
   - **`GEMINI_API_KEY`** (or **`GOOGLE_API_KEY`**) — [Google AI Studio](https://aistudio.google.com/apikey)
   - **`OPENAI_API_KEY`** — **embeddings** and hybrid RAG (`text-embedding-3-small` by default)

   All other keys in `.env.example` are optional; omit them or use the commented examples. Defaults for app behavior are defined in **`config.py`**.

4. Run the app:

   ```bash
   PYTHONPATH=. streamlit run app.py
   ```

5. Open the URL Streamlit prints (usually http://127.0.0.1:8501), then **Create account** or **Sign in**. Each user gets a private document index under **`DOCUBOT_DATA_DIR`** (default **`.docubot/`**).

## Configuration

`app.py` calls **`load_dotenv()`** so variables can live in a **`.env`** file next to the project. The template **`.env.example`** lists supported keys. Optional settings default as implemented in **`config.py`** (see the functions that read each `DOCUBOT_*` name).

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini chat (default provider). |
| `OPENAI_API_KEY` | **Embeddings** (and OpenAI chat if you switch provider). |
| `DOCUBOT_LLM_PROVIDER` | `gemini` (default), `openai`, or `ollama`. |
| `DOCUBOT_CHAT_MODEL` | e.g. `gemini-2.5-flash` or `gemini-2.0-flash` (see [Gemini models](https://ai.google.dev/gemini-api/docs/models); older ids such as `gemini-1.5-flash` may return **404** on the current Developer API). |
| `DOCUBOT_EMBEDDING_MODEL` | Default `text-embedding-3-small`. |
| `DOCUBOT_EMBEDDING_PROVIDER` | `openai` (default). |
| `OLLAMA_API_KEY` | Passed to the Ollama OpenAI-compatible client (often the literal `ollama`). |
| `DOCUBOT_OLLAMA_BASE_URL` | When using Ollama for chat (e.g. `http://localhost:11434/v1`). |
| `DOCUBOT_WEB_SEARCH_BACKEND` | `duckduckgo` (default) or `tavily` (requires `TAVILY_API_KEY`). |
| `TAVILY_API_KEY` | Required when `DOCUBOT_WEB_SEARCH_BACKEND=tavily`. |
| `DOCUBOT_DATA_DIR` | App data root: SQLite users DB (`users.sqlite3`) and per-user Chroma under `users/<id>/chroma` (default **`.docubot`** locally; Docker sets **`/data`**). |
| `DOCUBOT_PUBLIC_URL` | Base URL (no trailing slash) for **verification** and **password-reset** links in emails. Must match the URL users open in production. Default `http://127.0.0.1:8501`. |
| `DOCUBOT_SMTP_*` | Optional SMTP for transactional mail (`DOCUBOT_SMTP_HOST`, `DOCUBOT_SMTP_PORT`, `DOCUBOT_SMTP_USER`, `DOCUBOT_SMTP_PASSWORD`, `DOCUBOT_SMTP_FROM`, `DOCUBOT_SMTP_USE_TLS`). If unset, **verification and password-reset links are shown in the app** (no email sent). |
| `DOCUBOT_SESSION_SECRET` | Optional HMAC key for signed “stay signed in” cookies; if unset, a key is derived from `DOCUBOT_DATA_DIR` (fine for single-machine dev). |
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

## Accounts, authentication, and data

- **Sign in / Create account** in the app: **email + password** only. Passwords are stored as **bcrypt** hashes in **`DOCUBOT_DATA_DIR/users.sqlite3`**. To reset the account store during development, delete `users.sqlite3` and restart the app.
- **Code layout:** `auth/service.py` (register, login, verification, password reset), `auth/db.py` (SQLite users + `user_sessions`), `auth/mail.py` (optional SMTP), `auth/session_token.py` (signed cookie payload), and **`streamlit_session.py`** (restore / persist login across browser refresh — see below).
- **Email verification** and **password reset** use one-time token links: `?verify=...` and `?reset=...` on your **`DOCUBOT_PUBLIC_URL`**. Set that variable in production to your real HTTPS app URL.
- **Outgoing email:** optional **SMTP** (see `.env.example`). If SMTP is not set, the app does not send email; **one-time verification and password-reset links appear on the page** so you can complete sign-up and reset flows in the browser.
- **Staying signed in after refresh:** Streamlit’s **`session_state`** alone does not survive F5. The app adds a **server session id** in the query string (`docubot_sid`, rows in **`user_sessions`** in SQLite), plus a **signed browser cookie**, and a one-run guard in `session_state` so widget reruns do not keep re-applying the URL session. **`DOCUBOT_SESSION_SECRET`** sets the HMAC key for cookies; if unset, a key is derived from `DOCUBOT_DATA_DIR`. **Sign out** clears the DB session, URL param, and cookie (see `streamlit_session.clear_all_session_persistence` and the logout path in `app.py`).
- **Data isolation:** each user has a **separate Chroma index** at `DOCUBOT_DATA_DIR/users/<user_id>/chroma/`. Retrieval and uploads apply only to the logged-in account.

## Deployment

### Docker

Build and run (mount a volume at **`/data`** so the user database and all per-user indexes survive restarts):

```bash
docker build -t docubot .
docker run --name docubot --rm -p 8501:8501 \
  --env-file .env \
  -e DOCUBOT_DATA_DIR=/data \
  -v docubot_data:/data \
  docubot
```

**Why the `-e DOCUBOT_DATA_DIR=/data` line?** A typical local **`.env`** has `DOCUBOT_DATA_DIR=.docubot`. When you use **`--env-file .env`**, that value is injected into the container and **overrides** the image default, so the app would write under `/app/.docubot` **inside the container** — not on the mounted volume. Anything not on the volume is lost when the container is removed. The extra `-e` forces data onto **`/data`**, which is the same path as **`-v docubot_data:/data`**, so your SQLite user DB and Chroma files persist in the `docubot_data` volume.

**Why `docker start` said “no such container”?** You used **`--rm`**: Docker **deletes** the container as soon as it stops, so the old name (`optimistic_ishizaka`) is gone. That is normal. Your data is supposed to live in the **volume**, not in the container; start a **new** container (same `docker run` or `docker compose up`) and with **`DOCUBOT_DATA_DIR=/data`** your user account will still be there.

The image sets **`DOCUBOT_DATA_DIR=/data`** in the Dockerfile, but **host `.env` wins** unless you override as above. Streamlit binds to `0.0.0.0` and uses **`PORT`** if set (common on PaaS), otherwise **`STREAMLIT_SERVER_PORT`**, otherwise **8501`.

### Checklist for a hosted environment

- Set **secrets** / env vars: `GEMINI_API_KEY`, `OPENAI_API_KEY`, and optionally `TAVILY_API_KEY`.
- Set **`DOCUBOT_PUBLIC_URL`** to the public app URL (for links in emails and for in-app copy when SMTP is off). For production, configure **SMTP** (or a provider-compatible relay) so users receive links by email; without SMTP, links are only shown in the UI.
- Set **`DOCUBOT_SESSION_SECRET`** to a long random value if you use multiple app replicas or need stable cookie validation across restarts.
- Provide a **persistent disk** or volume for **`DOCUBOT_DATA_DIR`** (or accounts and indexes are lost on redeploy).
- Ensure outbound HTTPS to Google, OpenAI, your **SMTP** host, and your chosen web-search backend.
- Size the instance for **Chroma + Streamlit** (small CPU/RAM is usually enough for demos).

## Architecture (high level)

1. **Upload** → `pipeline/ingestor.py` extracts text (PDF via pdfplumber / pypdf, CSV normalized, TXT with encoding detection).
2. **Clean** → `pipeline/cleaner.py` five stages (in order): **encoding fix** (UTF-8 / chardet / BOM), **text cleanup** (control chars, whitespace), **PII redaction** (regex, optional), **deduplication** (exact lines), **quality check** (score + audit). Per-stage logs are stored on the cleaning report.
3. **Chunk** → `pipeline/chunker.py` splits on **token** boundaries (defaults **500** tokens, **50** overlap via `chunk_text_by_tokens()`; adjust in code / call sites — not env-driven today).
4. **Index** → `rag/embedder.py` embeds chunks (**OpenAI**); `rag/store.py` writes to Chroma with metadata (`source`, `page`, `chunk_index`, `doc_id`).
5. **Retrieve** → `rag/retriever.py` runs **cosine similarity** on embeddings and **BM25** over the same corpus, then **RRF**-style fusion of ranks.
6. **Agent** → `agent/graph.py`: **Planner** (structured plan), **Retriever** (tools), **Grader**, up to **two** re-plan loops, then **Generator** prep. The UI **streams** the final answer via `services/llm.py` (**Gemini** by default).
7. **UI** → `app.py`: **auth wall** (tabs: sign-in + create account; password reset and resend-verification in expanders; `?reset=` flow); call **`streamlit_session.try_restore_user_session()`** right after **`st.set_page_config`**; sidebar **upload**, indexed **document list** (per-file delete, **Remove all documents** clears the user index), **chat** with pinned input; **workflow trace**; **Sources** under assistant replies when chunks exist.

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

Automated tests focus on the **ingestion / cleaning / RAG / LangGraph** stack (see table below). **Authentication and Streamlit session persistence** are **not** covered by this repo’s test suite; exercise sign-in, email links, and password reset manually or in your own environment.

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
| 6 | **Auth:** create account, open the on-page verification link, sign in; optional forgot-password / resend; confirm you stay signed in after refresh | *No pytest coverage for `auth/` or `streamlit_session.py`* |

Full **LangGraph + Gemini** in the UI needs valid **`.env`** keys; automated tests avoid live LLM calls except where stubbed.

## Local embeddings (optional)

The stack uses **OpenAI embeddings**. A `local` embedding mode is possible with extra installs; see older notes in git history or extend `rag/embedder.py`.

## Notes

- **Console tracebacks mentioning `torchvision` / `transformers`:** These are **not** DocuBot bugs. Streamlit’s **file watcher** inspects loaded packages; optional `transformers` submodules expect `torchvision`, which this project does not install. The app still runs. The repo sets **`fileWatcherType = "none"`** in `.streamlit/config.toml` to keep the console quiet (you lose automatic reload-on-save for code changes; refresh the browser or switch to `auto` if you prefer reload and can ignore the noise).

- **Gemini** structured outputs are used for Planner / Grader via LangChain’s `with_structured_output`. If a model id errors, pick a current id from [Google AI Studio](https://aistudio.google.com/) and set **`DOCUBOT_CHAT_MODEL`** (avoid outdated ids that return **404** on the current API).
- **Streaming** uses the active chat provider (**Gemini** or **OpenAI**) for the final generation call in `app.py`.
- **Re-plan cap**: at most **two** `bump_replan` cycles before the graph proceeds to generation.
