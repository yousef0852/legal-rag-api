"""
Full RAG pipeline: embed query → Qdrant search → build context → LLM → return answer + sources.
"""
from __future__ import annotations

from typing import Any, Optional

from configs.config import Settings
from configs.exceptions import LLMError, RetrievalError
from configs.logging import get_logger
from services.embedder import Embedder
from services.llm_service import call_llm
from services.retriever import Retriever

logger = get_logger("pipeline")


async def answer(
    question: str,
    *,
    embedder: Embedder,
    retriever: Retriever,
    settings: Settings,
) -> dict[str, Any]:
    """
    Returns:
      {
        "answer": str,
        "sources": [{"title", "base_source", "source", "file", "score"}],
        "llm_meta": {...},
        "error": str | None,
      }
    """
    # 1. Embed the query
    try:
        query_vec = embedder.embed_query(question)
    except Exception as exc:
        logger.error("embed_query_failed", exc_info=True)
        raise RetrievalError(f"embedding failed: {exc}") from exc

    # 2. Retrieve top-k chunks
    try:
        hits = retriever.search(query_vec, top_k=settings.TOP_K)
    except Exception as exc:
        logger.error("qdrant_search_failed", exc_info=True)
        raise RetrievalError(f"Qdrant search failed: {exc}") from exc

    if not hits:
        return {
            "answer": "لم أجد نصوصاً قانونية ذات صلة بسؤالك في قاعدة البيانات.",
            "sources": [],
            "llm_meta": {},
            "error": "NO_RESULTS",
        }

    sources = [
        {
            "title": h.get("title") or "",
            "base_source": h.get("base_source") or "",
            "source": h.get("source") or "",
            "file": h.get("file") or "",
            "score": round(h.get("score", 0.0), 4),
        }
        for h in hits
    ]

    # 3. Call LLM (optional — if not configured, return retrieved context directly)
    if not settings.llm_configured():
        logger.info("llm_not_configured_returning_context")
        fallback = "\n\n".join(
            f"**{h.get('title') or h.get('file')}**\n{h.get('text', '')[:600]}"
            for h in hits
        )
        return {
            "answer": fallback,
            "sources": sources,
            "llm_meta": {"note": "LLM not configured — showing raw retrieved context"},
            "error": "LLM_NOT_CONFIGURED",
        }

    answer_text, llm_meta, error_code = await call_llm(
        question=question,
        hits=hits,
        api_base=settings.LLM_API_BASE,
        api_key=settings.LLM_API_KEY,
        model=settings.LLM_MODEL,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
        max_tokens=settings.LLM_MAX_TOKENS,
    )

    if error_code:
        logger.warning("llm_fallback", error_code=error_code)
        fallback = "\n\n".join(
            f"**{h.get('title') or h.get('file')}**\n{h.get('text', '')[:600]}"
            for h in hits
        )
        return {
            "answer": fallback,
            "sources": sources,
            "llm_meta": llm_meta,
            "error": error_code,
        }

    logger.info("pipeline_ok", question_preview=question[:60], n_sources=len(sources))
    return {
        "answer": answer_text,
        "sources": sources,
        "llm_meta": llm_meta,
        "error": None,
    }
