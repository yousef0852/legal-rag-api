#!/usr/bin/env python3
"""
Run once to embed all documents and load them into Qdrant.

  python scripts/index_corpus.py
  python scripts/index_corpus.py --recreate   # drop and rebuild collection
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.config import settings
from configs.logging import setup_logging, get_logger
from services.embedder import Embedder
from services.indexer import run_indexing
from services.retriever import Retriever

setup_logging()
logger = get_logger("index_corpus")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild the Qdrant collection")
    args = parser.parse_args()

    logger.info("starting_indexer", data_dir=str(settings.DATA_DIR), model=settings.EMBEDDING_MODEL)

    embedder = Embedder(settings.EMBEDDING_MODEL)
    retriever = Retriever(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        collection=settings.QDRANT_COLLECTION,
        vector_size=embedder.dim,
    )

    total = run_indexing(
        data_dir=settings.DATA_DIR,
        embedder=embedder,
        retriever=retriever,
        recreate=args.recreate,
    )

    logger.info("done", total_chunks=total, collection=settings.QDRANT_COLLECTION)
    print(f"\nIndexed {total} chunks into collection '{settings.QDRANT_COLLECTION}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
