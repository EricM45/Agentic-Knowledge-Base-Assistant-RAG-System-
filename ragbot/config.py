"""Runtime settings loaded from the environment and an optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _default_index_dir() -> Path:
    return ROOT / "data" / "index"


def _default_sample_dir() -> Path:
    return ROOT / "sample_documents"


def _optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _integer(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _path_value(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


@dataclass(frozen=True)
class Settings:
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 140
    top_k: int = 4
    min_dense_score: float = 0.34
    min_rerank_score: float = 0.15
    min_idf: float = 1.15
    min_coverage: float = 0.72
    use_reranker: bool = True
    backend: str = "auto"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    ollama_model: str | None = None
    vector_store: str = "auto"
    pinecone_api_key: str | None = None
    pinecone_index: str = "rag-chatbot"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    index_dir: Path = field(default_factory=_default_index_dir)
    sample_dir: Path = field(default_factory=_default_sample_dir)

    @classmethod
    def from_env(cls) -> "Settings":
        try:
            from dotenv import load_dotenv

            load_dotenv(ROOT / ".env")
        except ImportError:
            pass
        flag = os.getenv("USE_RERANKER", "1").strip().lower()
        return cls(
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model).strip() or cls.embedding_model,
            reranker_model=os.getenv("RERANKER_MODEL", cls.reranker_model).strip() or cls.reranker_model,
            chunk_size=_integer("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_integer("CHUNK_OVERLAP", cls.chunk_overlap),
            top_k=_integer("TOP_K", cls.top_k),
            use_reranker=flag not in {"0", "false", "no"},
            openai_api_key=_optional("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", cls.openai_model).strip() or cls.openai_model,
            openai_base_url=_optional("OPENAI_BASE_URL"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", cls.ollama_base_url).strip() or cls.ollama_base_url,
            ollama_model=_optional("OLLAMA_MODEL"),
            vector_store=os.getenv("VECTOR_STORE", "auto").strip().lower() or "auto",
            pinecone_api_key=_optional("PINECONE_API_KEY"),
            pinecone_index=os.getenv("PINECONE_INDEX", "rag-chatbot").strip() or "rag-chatbot",
            pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip() or "aws",
            pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip() or "us-east-1",
            index_dir=_path_value("INDEX_DIR", _default_index_dir()),
            sample_dir=_path_value("SAMPLE_DIR", _default_sample_dir()),
        )
