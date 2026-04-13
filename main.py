from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from configs.config import settings
from configs.exceptions import LegalRAGException
from configs.logging import get_logger, setup_logging
from interfaces.api.chat_route import router as chat_router
from interfaces.api.health_route import router as health_router
from services.embedder import Embedder
from services.retriever import Retriever

setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", model=settings.EMBEDDING_MODEL, collection=settings.QDRANT_COLLECTION)

    embedder = Embedder(settings.EMBEDDING_MODEL)
    retriever = Retriever(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        collection=settings.QDRANT_COLLECTION,
        vector_size=embedder.dim,
    )

    app.state.embedder = embedder
    app.state.retriever = retriever

    qdrant_ok = retriever.is_healthy()
    docs = retriever.count() if qdrant_ok else 0
    logger.info("startup_complete", qdrant_ok=qdrant_ok, docs_indexed=docs)

    yield

    app.state.embedder = None
    app.state.retriever = None
    logger.info("shutdown")


app = FastAPI(
    title="Legal RAG API",
    description="Arabic legal chatbot — RAG over Saudi legal corpus",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(LegalRAGException)
async def legal_exception_handler(request, exc: LegalRAGException):
    return JSONResponse(status_code=500, content=exc.to_dict())


app.include_router(chat_router)
app.include_router(health_router)
