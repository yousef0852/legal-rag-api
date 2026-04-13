from fastapi import Request

from services.embedder import Embedder
from services.retriever import Retriever


def get_embedder(request: Request) -> Embedder:
    return request.app.state.embedder


def get_retriever(request: Request) -> Retriever:
    return request.app.state.retriever
