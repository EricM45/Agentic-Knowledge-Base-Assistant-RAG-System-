"""Shared data objects for indexing and answers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    text: str
    source: str
    section: str
    page: int | None


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    source: str
    section: str
    page: int | None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "source": self.source,
            "section": self.section,
            "page": self.page,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Chunk":
        return cls(
            index=int(payload["index"]),
            text=str(payload["text"]),
            source=str(payload["source"]),
            section=str(payload.get("section", "")),
            page=payload.get("page"),
        )


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    dense_score: float
    sparse_score: float
    fused_score: float
    rerank_score: float | None = None


@dataclass(frozen=True)
class Source:
    number: int
    source: str
    section: str
    page: int | None
    text: str
    dense_score: float
    sparse_score: float
    fused_score: float
    rerank_score: float | None


@dataclass(frozen=True)
class Answer:
    text: str
    backend: str
    refused: bool
    standalone_question: str
    sources: tuple[Source, ...]
    warning: str | None = None
    vector_store: str = "local"
    query_terms: tuple[str, ...] = ()
    expanded_terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Generated:
    text: str
    backend: str
    refused: bool
    warning: str | None = None


@dataclass(frozen=True)
class IndexStats:
    files: int
    sections: int
    chunks: int
    skipped: tuple[str, ...] = ()
