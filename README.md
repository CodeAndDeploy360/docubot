# DocuBot

DocuBot is a training project: a single-page **Streamlit** app where you upload **PDF, TXT, or CSV** files, index them into **ChromaDB**, and ask questions in a chat UI. Answers are produced by a **LangGraph** workflow (Planner → Retriever → Grader → Generator prompt + streamed LLM) with **hybrid retrieval** (vector + BM25) and **source citations**.

**Project direction (PM):** chat uses **Google Gemini** (free tier / AI Studio); **embeddings** use **OpenAI**; **web search** is a **real** integration (DuckDuckGo by default, optional **Tavily** for production); the repo is **deployment-ready** (Docker + env conventions).

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
| `DOCUBOT_AGENT_MODE` | **Default (Gemini, env unset):** `fast` — one hybrid search + **one** Gemini call (fits free-tier limits). **`full`:** LangGraph planner / retriever / grader / re-plan (**3+** API calls per question). Set `DOCUBOT_AGENT_MODE=full` when you have quota or billing. |
| `TAVILY_API_KEY` | Required when `DOCUBOT_WEB_SEARCH_BACKEND=tavily`. |
| `DOCUBOT_OLLAMA_BASE_URL` | When using Ollama for chat. |
| `DOCUBOT_CHROMA_PATH` | Chroma persistence directory (default `./vector_db`; Docker image uses `/data/chroma`). |
| `DOCUBOT_PII_REDACTION_ENABLED` | `true` / `false` for regex PII masking during cleaning. |

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
2. **Clean** → `pipeline/cleaner.py` runs encoding normalization, cleanup, optional PII redaction, optional line dedupe, and a simple quality score with per-stage audit metadata.
3. **Chunk** → `pipeline/chunker.py` splits on **token** boundaries (~500 tokens, 50 overlap) using `tiktoken`.
4. **Index** → `rag/embedder.py` embeds chunks (**OpenAI**); `rag/store.py` writes to Chroma with metadata (`source`, `page`, `chunk_index`, `doc_id`).
5. **Retrieve** → `rag/retriever.py` runs **cosine similarity** on embeddings and **BM25** over the same corpus, then **RRF**-style fusion of ranks.
6. **Agent** → `agent/graph.py`: **Planner** (structured plan), **Retriever** (tools), **Grader**, up to **two** re-plan loops, then **Generator** prep. The UI **streams** the final answer via `services/llm.py` (**Gemini** by default).
7. **UI** → `app.py`: sidebar upload + document status + delete; chat; **workflow trace**.

## Test data

Under `test_data/`:

- `sample.pdf` — short revenue snippet (regenerate with `python scripts/gen_sample_pdf.py` if needed).
- `quarterly_report.txt`, `messy_data.csv`, `pii_test.txt` — demo scenarios.

## Tests

```bash
PYTHONPATH=. pytest tests/ -v --cov=pipeline.cleaner --cov-report=term-missing
```

Cleaner module is targeted for **≥80%** line coverage in `test_cleaner.py`.

## Local embeddings (optional)

The PM stack uses **OpenAI embeddings**. A `local` embedding mode is possible with extra installs; see older notes in git history or extend `rag/embedder.py`.

## Notes

- **Console tracebacks mentioning `torchvision` / `transformers`:** These are **not** DocuBot bugs. Streamlit’s **file watcher** inspects loaded packages; optional `transformers` submodules expect `torchvision`, which this project does not install. The app still runs. The repo sets **`fileWatcherType = "none"`** in `.streamlit/config.toml` to keep the console quiet (you lose automatic reload-on-save for code changes; refresh the browser or switch to `auto` if you prefer reload and can ignore the noise).

- **Gemini** structured outputs are used for Planner / Grader via LangChain’s `with_structured_output`; if a model ID errors, try `gemini-1.5-flash` or check current model names in Google AI Studio.
- **Streaming** uses the active chat provider (**Gemini** or **OpenAI**) for the final generation call in `app.py`.
- **Re-plan cap**: at most **two** `bump_replan` cycles before the graph proceeds to generation.
