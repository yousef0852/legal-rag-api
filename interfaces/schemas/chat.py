from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="السؤال القانوني بالعربية")


class SourceItem(BaseModel):
    title: str
    base_source: str  # "boe" | "sahl" | "qanoniah"
    source: str       # folder name
    file: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    error: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
