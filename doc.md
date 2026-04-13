# Legal RAG API — codebase guide

This document explains **what each part of the project does**, **how data flows**, and **how to read the code** in a sensible order.

---

## 1. Big picture

The service is a **RAG (Retrieval-Augmented Generation)** pipeline for **Arabic legal questions**:

1. **Offline (once):** Split markdown documents into chunks, embed each chunk with a sentence-transformer model, store vectors + metadata in **Qdrant**.
2. **Online (each request):** Embed the user’s question, search Qdrant for similar chunks, optionally call an **OpenAI-compatible LLM** to write an answer grounded in those chunks, return **answer + source list**.

You do **not** need an LLM key to run retrieval: the API can return raw retrieved text with a clear metadata flag.

---

## 2. Where to start reading (recommended order)

Read in this order to follow **runtime behavior** first, then **indexing**:

| Step | File(s) | Why |
|------|---------|-----|
| 1 | `main.py` | App entry: lifespan, routers, global `Embedder` + `Retriever`. |
| 2 | `configs/config.py` | All env-driven settings in one place. |
| 3 | `interfaces/api/chat_route.py` | HTTP `POST /chat` → calls pipeline. |
| 4 | `core/pipeline.py` | The full RAG logic: embed → search → LLM or fallback. |
| 5 | `services/embedder.py` | How text becomes vectors (E5 prefixes). |
| 6 | `services/retriever.py` | Qdrant: collection, upsert, search, health. |
| 7 | `services/llm_service.py` | OpenAI-style `chat/completions` call. |
| 8 | `scripts/index_corpus.py` | CLI that fills Qdrant from disk. |
| 9 | `services/indexer.py` | Chunking rules + batch embed + upsert. |

After that, skim **`interfaces/schemas/chat.py`**, **`interfaces/api/dependencies.py`**, **`interfaces/api/health_route.py`**, **`configs/logging.py`**, **`configs/exceptions.py`**, and **`docker-compose.yml`**.

---

## 3. Architecture (layers)

```
┌─────────────────────────────────────────────────────────────┐
│  main.py          FastAPI app, CORS, exception handler       │
│  interfaces/api/* HTTP routes + Depends() wiring             │
│  interfaces/schemas/*  Pydantic request/response models      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  core/pipeline.py     answer(): embed → retrieve → LLM/fallback│
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
┌──────────────▼──────────────┐   ┌──────────▼──────────────┐
│  services/embedder.py       │   │  services/retriever.py   │
│  SentenceTransformer + E5   │   │  QdrantClient            │
│  prefixes (passage/query)   │   │  search / upsert / count │
└─────────────────────────────┘   └──────────────────────────┘
                                           ▲
┌──────────────────────────────────────────┴──────────────────┐
│  services/indexer.py + scripts/index_corpus.py               │
│  (chunk markdown → embed passages → upsert)                  │
└─────────────────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│  services/llm_service.py  (optional synthesis)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Lifecycle A — starting the API (`uvicorn main:app`)

1. **`main.py` — `lifespan`**
   - Logs startup.
   - Builds **`Embedder`** (loads `settings.EMBEDDING_MODEL`; first time may download from Hugging Face).
   - Builds **`Retriever`** (Qdrant client; uses `embedder.dim` as vector size).
   - Stores both on **`app.state`** for request handlers.
   - Checks Qdrant health and logs indexed point count.

2. **Routers**
   - **`/chat`** → `chat_route.chat`.
   - **`/health`** → `health_route.health`.

3. **Shutdown**
   - Clears `app.state` references (models may be freed by the process exit).

---

## 5. Lifecycle B — one `POST /chat` request

1. **`interfaces/schemas/chat.py` — `ChatRequest`**
   - Validates `question` length (3–2000 chars).

2. **`interfaces/api/dependencies.py`**
   - `get_embedder` / `get_retriever` read from `request.app.state` (set in lifespan).

3. **`interfaces/api/chat_route.py`**
   - Calls **`core.pipeline.answer(...)`** with question, embedder, retriever, `settings`.
   - Maps **`RetrievalError`** → HTTP 503 JSON.
   - Maps other **`LegalRAGException`** → HTTP 500 JSON.
   - Success → **`ChatResponse`**: `answer`, `sources`, optional `error`, `meta`.

4. **`core/pipeline.py` — `answer`**
   - **Embed query:** `embedder.embed_query(question)` → one vector.
   - **Search:** `retriever.search(query_vec, top_k=settings.TOP_K)` → list of hits with `text`, `title`, `file`, `score`, etc.
   - **No hits:** Arabic message, empty sources, `error: "NO_RESULTS"`.
   - **LLM not configured** (`settings.llm_configured()` is false): concatenates truncated chunk text into `answer`, sets `error: "LLM_NOT_CONFIGURED"`.
   - **LLM configured:** `await call_llm(...)`; on failure, same style of fallback as above with an error code (`LLM_TIMEOUT`, etc.).
   - **Success:** LLM-generated `answer` + `sources` + `meta`.

5. **`services/llm_service.py` — `call_llm`**
   - Builds Arabic **system** + **user** messages from question + retrieved chunks.
   - `POST {api_base}/chat/completions` with Bearer key.
   - Parses `choices[0].message.content`; handles timeouts and HTTP errors.

---

## 6. Lifecycle C — indexing (`python scripts/index_corpus.py`)

1. **`scripts/index_corpus.py`**
   - Adds project root to `sys.path` so imports work when run as a script.
   - Parses `--recreate` (passed to indexer to drop collection if it exists).
   - Instantiates **`Embedder`** and **`Retriever`** (same pattern as `main.py`).
   - Calls **`services.indexer.run_indexing`**.

2. **`services/indexer.py` — `run_indexing`**
   - **`retriever.ensure_collection(recreate)`** — creates `legal_ar` (or skips / drops per flag).
   - Walks **`settings.DATA_DIR`** (default `data/markdown`) via **`SOURCE_FOLDERS`**:
     - `boe_laws_detail` → base source `boe`
     - `qanoniah_blog` → `qanoniah`
     - `sahl_blog` → `sahl`
   - Only files matching **`Name__{16 hex}.md`** are indexed.
   - **Chunking:**
     - **BOE laws:** split on headings that look like **article markers** (`#`…`####` + `المادة`).
     - **Blogs:** split on markdown headings `##` / `###` at line start; long sections split into **1000**-char windows with **100**-char overlap.
   - Batches of **`BATCH_SIZE` (32)** chunks: **`embedder.embed_passages`** (adds `passage: ` prefix), then **`retriever.upsert`**.
   - Logs `indexed_batch` and finally `indexing_complete`.

**Important:** Query-time uses **`query: `** prefix; indexing uses **`passage: `**. That pairing is required for **E5** models.

---

## 7. File-by-file reference

### Root

| File | Role |
|------|------|
| `main.py` | FastAPI app factory: lifespan, CORS, exception handler, includes routers. |
| `requirements.txt` | Pinned Python dependencies (FastAPI, torch, sentence-transformers, qdrant-client, etc.). |
| `Dockerfile` | Slim Python 3.12 image: install deps, copy app, run uvicorn on 8000. |
| `docker-compose.yml` | **qdrant** service + **app** build; mounts `./data` and a Hugging Face cache volume; sets `QDRANT_HOST=qdrant` inside the app container. |
| `.env` / `.env.example` | Runtime secrets and overrides (not committed if `.env` is gitignored). Copy example → `.env` and edit. |
| `doc.md` | This guide. |

### `configs/`

| File | Role |
|------|------|
| `config.py` | Loads `.env` via `load_dotenv()`, exposes **`Settings`** (Qdrant, model name, data dir, LLM URL/key/model, `TOP_K`). Singleton **`settings`**. |
| `logging.py` | **`setup_logging()`** — structlog JSON to stdout; **`get_logger(name)`**. |
| `exceptions.py` | **`LegalRAGException`** base with `to_dict()`; **`RetrievalError`**, **`LLMError`**, and an indexing-related type (used for consistent API errors). |

### `interfaces/`

| File | Role |
|------|------|
| `schemas/chat.py` | **`ChatRequest`**, **`ChatResponse`**, **`SourceItem`** (Pydantic models for OpenAPI + validation). |
| `api/dependencies.py` | FastAPI **`Depends`** helpers to inject **`Embedder`** / **`Retriever`** from `app.state`. |
| `api/chat_route.py` | **`POST /chat`** implementation. |
| `api/health_route.py` | **`GET /health`** — Qdrant connectivity + **`docs_indexed`** (point count). |

### `core/`

| File | Role |
|------|------|
| `pipeline.py` | **`answer()`** — single place for RAG orchestration and LLM vs no-LLM behavior. |

### `services/`

| File | Role |
|------|------|
| `embedder.py` | **`Embedder`** wrapper: `embed_passages`, `embed_query`, E5 prefixes, normalized vectors. |
| `retriever.py` | **`Retriever`**: `ensure_collection`, `upsert`, `search`, `count`, `is_healthy`. |
| `indexer.py` | Chunking + **`run_indexing`** (used by script and conceptually the “write path” mirror of search). |
| `llm_service.py` | **`call_llm`** — httpx async POST to OpenAI-compatible chat API; Arabic prompts. |

### `scripts/`

| File | Role |
|------|------|
| `index_corpus.py` | Entrypoint to build/update the Qdrant index from `data/markdown`. |

### `data/`

| Path | Role |
|------|------|
| `data/markdown/boe_laws_detail/` | BOE-style legal markdown; article-based chunking. |
| `data/markdown/qanoniah_blog/` | Blog markdown; heading + max-length chunking. |
| `data/markdown/sahl_blog/` | Same chunking style as qanoniah. |

Filenames are expected like **`SomeTitle__a1b2c3d4e5f67890.md`** (16 hex chars before `.md`).

### `tests/`

Placeholder for automated tests (add pytest tests here as the project grows).

---

## 8. Configuration (environment variables)

See **`.env.example`** for the full list. Main concepts:

- **Qdrant:** `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_COLLECTION`
- **Embeddings:** `EMBEDDING_MODEL` (must match what you used at index time)
- **Corpus path:** `DATA_DIR` (indexing only)
- **LLM:** `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL`, timeouts / `max_tokens`
- **Retrieval:** `TOP_K`

If **`LLM_API_KEY`** is empty, the pipeline **skips** the LLM and returns retrieved excerpts.

---

## 9. How to “think like the codebase”

1. **Separate two modes:** **write path** (indexer script → Qdrant) vs **read path** (API → pipeline → Qdrant search).
2. **Same vector space:** Indexing and queries must use the **same** embedding model and the same **E5** `passage:` / `query:` convention.
3. **Qdrant stores payloads:** Search returns scores plus **`text`**, **`title`**, **`file`**, **`base_source`** — those drive both the LLM context and the citation list in the API response.
4. **Failures are layered:** Embedding/search errors → **503** retrieval errors; LLM failure → still **200** with fallback text + `error` field in body (see `pipeline.py`).

---

## 10. Quick command recap

```bash
# Qdrant (example: Docker / Colima)
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant

# Index (from project root, venv active)
python scripts/index_corpus.py

# API
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000/docs`** for interactive API documentation.
