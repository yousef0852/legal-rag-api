from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path
from typing import Optional


class Settings:
    # Qdrant
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "legal_ar")

    # Embedding model (sentence-transformers)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

    # Data
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data" / "markdown")))

    # LLM (OpenAI-compatible)
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").rstrip("/")
    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    def llm_configured(self) -> bool:
        key = self.LLM_API_KEY
        return bool(key and str(key).strip())


settings = Settings()
