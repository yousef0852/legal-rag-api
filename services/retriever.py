"""
Qdrant client wrapper — create collection, upsert vectors, search.
"""
from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from configs.logging import get_logger

logger = get_logger("retriever")


class Retriever:
    def __init__(self, host: str, port: int, collection: str, vector_size: int):
        self._client = QdrantClient(host=host, port=port)
        self._collection = collection
        self._vector_size = vector_size

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, recreate: bool = False) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection in existing:
            if recreate:
                logger.info("dropping_collection", collection=self._collection)
                self._client.delete_collection(self._collection)
            else:
                logger.info("collection_exists", collection=self._collection)
                return
        logger.info("creating_collection", collection=self._collection, size=self._vector_size)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
        )

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert(self, points: list[dict[str, Any]]) -> None:
        """
        Each item must have: id (int), vector (list[float]), payload (dict).
        """
        structs = [
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in points
        ]
        self._client.upsert(collection_name=self._collection, points=structs)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "score": h.score,
                "doc_id": h.payload.get("doc_id"),
                "chunk_index": h.payload.get("chunk_index"),
                "source": h.payload.get("source"),
                "base_source": h.payload.get("base_source"),
                "file": h.payload.get("file"),
                "title": h.payload.get("title"),
                "text": h.payload.get("text"),
            }
            for h in hits
        ]

    def is_healthy(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False
