"""
Thin wrapper around SentenceTransformer.
E5 models require a task prefix:
  - documents → "passage: <text>"
  - queries   → "query: <text>"
"""
from __future__ import annotations

from typing import Union

from sentence_transformers import SentenceTransformer

from configs.logging import get_logger

logger = get_logger("embedder")


class Embedder:
    def __init__(self, model_name: str):
        logger.info("loading_embedding_model", model=model_name)
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("embedding_model_loaded", model=model_name, dim=self._dim)

    @property
    def dim(self) -> int:
        return self._dim

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self._model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefixed = f"query: {text}"
        vec = self._model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return vec.tolist()
