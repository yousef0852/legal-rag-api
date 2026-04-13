# Legal RAG API

FastAPI service for an Arabic legal question-answering flow: embed the query, retrieve chunks from [Qdrant](https://qdrant.tech/), optionally call an OpenAI-compatible LLM, and return an answer with source citations.

## Requirements

- **Python 3.12** (recommended; pinned `torch` may not install on newer Python versions)
- **Qdrant** reachable at the host/port you configure (default `localhost:6333`)
- Markdown corpus under `data/markdown/` for indexing

## Setup

1. Clone the repository and enter the project directory.

2. Create a virtual environment and install dependencies:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy environment template and edit as needed:

   ```bash
   cp .env.example .env
   ```

   Set `LLM_API_KEY` if you want the LLM to synthesize answers. If it is unset, the API still returns retrieved context with a clear `LLM_NOT_CONFIGURED` flag in the response metadata.

4. Start Qdrant (for example with Docker):

   ```bash
   docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

   Or use [Colima](https://github.com/abiosoft/colima) or another runtime if you prefer not to use Docker Desktop.

5. Index the corpus (downloads the embedding model on first run):

   ```bash
   python scripts/index_corpus.py
   ```

   Use `--recreate` to drop and rebuild the collection.

6. Run the API:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## Docker Compose

From the project root:

```bash
docker compose up --build
```

The `app` service expects a `.env` file (see `.env.example`). Qdrant is started as a sibling service.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness and Qdrant status, indexed document count |
| POST | `/chat` | JSON body: `{"question": "<Arabic text, 3–2000 chars>"}` |

OpenAPI UI: `http://localhost:8000/docs`

### Example

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"ما هي حقوق العامل في نظام العمل؟"}'
```

## Configuration

All variables are documented in `.env.example` (Qdrant, embedding model, LLM endpoint, retrieval `TOP_K`).

## License

Specify your license here if applicable.
