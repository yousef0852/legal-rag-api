from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health(request: Request):
    retriever = request.app.state.retriever
    qdrant_ok = retriever.is_healthy() if retriever else False
    docs_indexed = retriever.count() if (retriever and qdrant_ok) else 0

    return {
        "status": "ok",
        "qdrant": "connected" if qdrant_ok else "unavailable",
        "docs_indexed": docs_indexed,
    }
