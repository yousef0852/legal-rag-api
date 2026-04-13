from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from configs.config import settings
from configs.exceptions import LegalRAGException, RetrievalError
from configs.logging import get_logger
from core.pipeline import answer
from interfaces.api.dependencies import get_embedder, get_retriever
from interfaces.schemas.chat import ChatRequest, ChatResponse, SourceItem
from services.embedder import Embedder
from services.retriever import Retriever

router = APIRouter()
logger = get_logger("chat_route")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    embedder: Embedder = Depends(get_embedder),
    retriever: Retriever = Depends(get_retriever),
):
    try:
        result = await answer(
            body.question,
            embedder=embedder,
            retriever=retriever,
            settings=settings,
        )
    except RetrievalError as exc:
        logger.error("retrieval_error", error=str(exc))
        return JSONResponse(status_code=503, content=exc.to_dict())
    except LegalRAGException as exc:
        logger.error("pipeline_error", error=str(exc))
        return JSONResponse(status_code=500, content=exc.to_dict())

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        error=result.get("error"),
        meta=result.get("llm_meta"),
    )
