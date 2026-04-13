"""
OpenAI-compatible LLM call for Arabic legal Q&A.
Returns (answer_str, meta_dict, error_code_or_None).
"""
from __future__ import annotations

import time
from typing import Any, Optional

import httpx

from configs.logging import get_logger

logger = get_logger("llm_service")

SYSTEM_PROMPT = """أنت مساعد قانوني متخصص في الأنظمة والقوانين السعودية.
مهمتك الإجابة على أسئلة المستخدمين بدقة واحترافية بناءً على النصوص القانونية المقدمة إليك فقط.

قواعد إلزامية:
- أجب فقط بناءً على المعلومات الواردة في النصوص المقدمة. لا تختلق معلومات.
- إذا لم تجد الإجابة في النصوص المقدمة، قل ذلك صراحةً.
- اذكر المصدر (اسم النظام أو المقال) عند الإجابة.
- أجب بالعربية الفصحى الواضحة.
- كن دقيقاً وموجزاً."""

USER_TEMPLATE = """السؤال: {question}

النصوص القانونية ذات الصلة:
---
{context}
---

أجب على السؤال بناءً على النصوص أعلاه فقط."""


def _build_context(hits: list[dict[str, Any]]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        source_label = {
            "boe": "هيئة الخبراء — نص نظام",
            "sahl": "مدونة سهل القانونية",
            "qanoniah": "مدونة قانونية",
        }.get(h.get("base_source", ""), "مصدر قانوني")

        title = h.get("title") or ""
        text = h.get("text") or ""
        parts.append(f"[{i}] المصدر: {source_label} — {title}\n{text}")
    return "\n\n".join(parts)


async def call_llm(
    question: str,
    hits: list[dict[str, Any]],
    *,
    api_base: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_tokens: int,
) -> tuple[Optional[str], dict[str, Any], Optional[str]]:
    context = _build_context(hits)
    user_msg = USER_TEMPLATE.format(question=question, context=context)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    timeout = httpx.Timeout(timeout_seconds, connect=10.0)
    started = time.perf_counter()
    meta: dict[str, Any] = {"model": model}

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning("llm_timeout", latency_ms=round(elapsed_ms, 2))
        meta["latency_ms"] = round(elapsed_ms, 2)
        return None, meta, "LLM_TIMEOUT"
    except httpx.RequestError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning("llm_request_error", error=str(exc))
        meta["latency_ms"] = round(elapsed_ms, 2)
        return None, meta, "LLM_HTTP_ERROR"

    elapsed_ms = (time.perf_counter() - started) * 1000
    meta["latency_ms"] = round(elapsed_ms, 2)
    meta["http_status"] = r.status_code

    if r.status_code >= 400:
        logger.warning("llm_http_error", status=r.status_code, body=r.text[:300])
        return None, meta, "LLM_HTTP_ERROR"

    try:
        answer = r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.warning("llm_bad_envelope", body=r.text[:300])
        return None, meta, "LLM_INVALID_RESPONSE"

    return answer.strip(), meta, None
