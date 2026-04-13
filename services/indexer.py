"""
Chunking + embedding + Qdrant upsert.

Chunking strategy (per plan):
  - boe_laws_detail: split on article boundaries (#### المادة)
  - qanoniah_blog / sahl_blog: split on ## / ### headers, hard-cap at 1000 chars with 100-char overlap
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Generator

from configs.logging import get_logger
from services.embedder import Embedder
from services.retriever import Retriever

logger = get_logger("indexer")

BATCH_SIZE = 32  # chunks per upsert call
MAX_BLOG_CHUNK = 1000
BLOG_OVERLAP = 100

_ARTICLE_RE = re.compile(r"(?=#{1,4}\s*المادة)", re.MULTILINE)
_HEADING_RE = re.compile(r"(?=^#{1,3} )", re.MULTILINE)
_FILENAME_RE = re.compile(r"^(?P<source>.+)__(?P<doc_id>[a-f0-9]{16})\.md$", re.I)

SOURCE_FOLDERS = {
    "boe_laws_detail": "boe",
    "qanoniah_blog": "qanoniah",
    "sahl_blog": "sahl",
}


def _extract_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def _chunk_boe(text: str) -> list[str]:
    """Split on article boundary; keep anything before first article as intro chunk."""
    parts = _ARTICLE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _chunk_blog(text: str) -> list[str]:
    """Split on ## / ### headings, then hard-cap long sections."""
    sections = _HEADING_RE.split(text)
    chunks: list[str] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= MAX_BLOG_CHUNK:
            chunks.append(sec)
        else:
            # Hard-cap with overlap
            start = 0
            while start < len(sec):
                end = min(start + MAX_BLOG_CHUNK, len(sec))
                chunks.append(sec[start:end].strip())
                if end >= len(sec):
                    break
                start = end - BLOG_OVERLAP
    return [c for c in chunks if c]


def _iter_docs(data_dir: Path) -> Generator[dict, None, None]:
    for folder_name, base_source in SOURCE_FOLDERS.items():
        folder = data_dir / folder_name
        if not folder.is_dir():
            logger.warning("folder_missing", folder=str(folder))
            continue
        for md_file in sorted(folder.glob("*.md")):
            if md_file.name.startswith("_"):
                continue
            m = _FILENAME_RE.match(md_file.name)
            if not m:
                continue
            doc_id = m.group("doc_id")
            text = md_file.read_text(encoding="utf-8").strip()
            if not text:
                continue
            yield {
                "doc_id": doc_id,
                "source": folder_name,
                "base_source": base_source,
                "file": f"{folder_name}/{md_file.name}",
                "title": _extract_title(text),
                "text": text,
            }


def run_indexing(
    data_dir: Path,
    embedder: Embedder,
    retriever: Retriever,
    recreate: bool = False,
) -> int:
    """Chunk all docs, embed, upsert to Qdrant. Returns total chunks inserted."""
    retriever.ensure_collection(recreate=recreate)

    point_id = 0
    batch: list[dict] = []
    total_chunks = 0

    def _flush(batch: list[dict]) -> None:
        texts = [b["payload"]["text"] for b in batch]
        vectors = embedder.embed_passages(texts)
        for b, vec in zip(batch, vectors):
            b["vector"] = vec
        retriever.upsert(batch)

    for doc in _iter_docs(data_dir):
        if doc["source"] == "boe_laws_detail":
            chunks = _chunk_boe(doc["text"])
        else:
            chunks = _chunk_blog(doc["text"])

        for i, chunk_text in enumerate(chunks):
            payload = {
                "doc_id": doc["doc_id"],
                "chunk_index": i,
                "source": doc["source"],
                "base_source": doc["base_source"],
                "file": doc["file"],
                "title": doc["title"],
                "text": chunk_text,
            }
            batch.append({"id": point_id, "payload": payload})
            point_id += 1
            total_chunks += 1

            if len(batch) >= BATCH_SIZE:
                _flush(batch)
                logger.info("indexed_batch", chunks_so_far=total_chunks)
                batch = []

    if batch:
        _flush(batch)

    logger.info("indexing_complete", total_chunks=total_chunks)
    return total_chunks
