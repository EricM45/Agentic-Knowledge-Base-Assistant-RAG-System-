"""Index documents and answer questions from the retrieved passages."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path

from ragbot.chunker import chunk_documents
from ragbot.config import Settings
from ragbot.embeddings import CrossEncoderReranker, SentenceTransformerEmbedder
from ragbot.generator import REFUSAL_TEXT, GroundedGenerator
from ragbot.loader import collect_files, load_paths
from ragbot.nlp import analyze_query, lexical_match, standalone_query
from ragbot.pinecone_store import (
    PineconeVectorStore,
    read_sync_record,
    sync_matches,
    write_sync_record,
)
from ragbot.retriever import CorpusIndex, dedupe_hits
from ragbot.store import corpus_signature, load_index, save_index
from ragbot.types import Answer, Hit, IndexStats, Source

logger = logging.getLogger("ragbot.pipeline")


class RagPipeline:
    def __init__(self, settings: Settings | None = None, embedder=None, generator=None, vector_store=None) -> None:
        self.settings = settings or Settings.from_env()
        self.embedder = embedder or SentenceTransformerEmbedder(self.settings.embedding_model)
        self.generator = generator or GroundedGenerator(self.embedder)
        self.vector_store = vector_store
        self.index: CorpusIndex | None = None
        self.stats = IndexStats(0, 0, 0)
        self.signature = ""
        self.vector_store_error: str | None = None
        self._reranker_model: CrossEncoderReranker | None = None
        self._reranker_failed = False
        self._ollama_probe: tuple | None = None
        self._pinecone_client: PineconeVectorStore | None = None
        self._pinecone_ready = False

    def configure(self, **changes) -> None:
        self.settings = replace(self.settings, **changes)
        self._ollama_probe = None
        self._pinecone_client = None
        self._pinecone_ready = False

    def index_paths(self, paths: list[Path], persist: bool = True) -> IndexStats:
        files = collect_files(paths)
        documents, skipped = load_paths(files)
        chunks = chunk_documents(documents, self.settings.chunk_size, self.settings.chunk_overlap)
        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        self.index = CorpusIndex(chunks, embeddings)
        self.stats = IndexStats(
            files=len({document.source for document in documents}),
            sections=len(documents),
            chunks=len(chunks),
            skipped=skipped,
        )
        self.signature = corpus_signature(files, self.settings, self.embedder.model_name)
        if persist and files:
            save_index(self.index, self.stats, self.settings.index_dir, self.signature, self.embedder.model_name)
        if self.uses_pinecone():
            self.ensure_pinecone()
        logger.info("Indexed %s files into %s passages", self.stats.files, self.stats.chunks)
        return self.stats

    def load_cached(self, paths: list[Path]) -> bool:
        files = collect_files(paths)
        if not files:
            return False
        signature = corpus_signature(files, self.settings, self.embedder.model_name)
        loaded = load_index(self.settings.index_dir, signature, self.embedder.model_name)
        if loaded is None:
            return False
        self.index, self.stats = loaded
        self.signature = signature
        return True

    def warm(self) -> None:
        if self.settings.use_reranker:
            self._reranker()

    def uses_pinecone(self) -> bool:
        choice = self.settings.vector_store
        if choice == "local":
            return False
        if choice == "pinecone":
            return self.vector_store is not None or bool(self.settings.pinecone_api_key)
        return bool(self.settings.pinecone_api_key)

    def needs_pinecone_sync(self) -> bool:
        if not self.uses_pinecone() or self.index is None or not self.signature:
            return False
        record = read_sync_record(self.settings.index_dir)
        return not sync_matches(record, self.signature, self.settings.pinecone_index)

    def ensure_pinecone(self) -> None:
        """Upload passage vectors when Pinecone is selected and this corpus is not already there."""
        if not self.uses_pinecone() or self.index is None or not self.index.chunks or not self.signature:
            return
        record = read_sync_record(self.settings.index_dir)
        if sync_matches(record, self.signature, self.settings.pinecone_index):
            self._pinecone_ready = True
            self.vector_store_error = None
            return
        previous = record.get("namespace") if record and record.get("index_name") == self.settings.pinecone_index else None
        try:
            self._vector_client().sync(
                self.index.chunks,
                self.index.embeddings,
                self.signature,
                previous_namespace=previous,
            )
            write_sync_record(
                self.settings.index_dir,
                signature=self.signature,
                index_name=self.settings.pinecone_index,
                namespace=self.signature,
            )
            self._pinecone_ready = True
            self.vector_store_error = None
            logger.info("Stored %s passage vectors in Pinecone index %s", len(self.index.chunks), self.settings.pinecone_index)
        except Exception:
            logger.warning("Pinecone sync failed: %s", _safe_error_name())
            self._pinecone_ready = False
            self.vector_store_error = (
                "Pinecone could not store the passages, so search is using the local vector index."
            )

    def vector_store_label(self) -> str:
        if self.uses_pinecone() and self._pinecone_ready and not self.vector_store_error:
            return "Pinecone"
        return "Local vectors"

    def ask(self, question: str, history: list[dict] | None = None, top_k: int | None = None) -> Answer:
        """Rewrite follow-ups, retrieve a hybrid pool, and answer from grounded passages."""
        question = " ".join((question or "").split())
        if len(question) > 2000:
            question = question[:2000]
        history = list(history or [])[-8:]
        if not question:
            return self._empty("Enter a question about the indexed documents.", question)
        if self.index is None or not self.index.chunks:
            return self._empty("Add documents before asking a question.", question)

        standalone = standalone_query(question, history)
        query_terms, expanded_terms = analyze_query(standalone)
        limit = top_k or self.settings.top_k
        query_vector = self.embedder.embed_query(standalone)
        pool_limit = max(limit * 3, 8)
        dense_hits, store_name = self._dense_hits(query_vector, pool_limit)
        pool = self.index.search(standalone, query_vector, limit=pool_limit, dense_hits=dense_hits)
        pool = self._apply_reranker(standalone, pool)
        pool = dedupe_hits(pool, self.index.embeddings)[:limit]
        grounded = [hit for hit in pool if self._is_grounded(standalone, hit)]
        if not grounded:
            warning = self.vector_store_error if store_name == "local" and self.uses_pinecone() else None
            return Answer(
                text=REFUSAL_TEXT,
                backend="none",
                refused=True,
                standalone_question=standalone,
                sources=self._to_sources(pool),
                warning=warning,
                vector_store=store_name,
                query_terms=query_terms,
                expanded_terms=expanded_terms,
            )
        generated = self.generator.generate(
            question,
            grounded,
            history,
            self.settings,
            self.index.bm25.idf,
            self.resolve_backend(),
        )
        warning = generated.warning
        if store_name == "local" and self.vector_store_error and self.uses_pinecone():
            warning = self.vector_store_error
        return Answer(
            text=generated.text,
            backend=generated.backend,
            refused=generated.refused,
            standalone_question=standalone,
            sources=self._to_sources(grounded),
            warning=warning,
            vector_store=store_name,
            query_terms=query_terms,
            expanded_terms=expanded_terms,
        )

    def resolve_backend(self) -> str:
        requested = self.settings.backend
        if requested == "local-nlp":
            return "local-nlp"
        if requested == "openai":
            return "openai" if self.settings.openai_api_key else "local-nlp"
        if requested == "ollama":
            return "ollama" if self.settings.ollama_model and self.ollama_ready() else "local-nlp"
        if self.settings.openai_api_key:
            return "openai"
        if self.settings.ollama_model and self.ollama_ready():
            return "ollama"
        return "local-nlp"

    def backend_label(self) -> str:
        labels = {
            "openai": f"OpenAI ({self.settings.openai_model})",
            "ollama": f"Ollama ({self.settings.ollama_model})",
            "local-nlp": "Local NLP synthesis",
        }
        active = self.resolve_backend()
        label = labels.get(active, active)
        if self.settings.backend not in {"auto", active}:
            label += " — the selected model is unavailable"
        return label

    def ollama_ready(self) -> bool:
        model = self.settings.ollama_model
        if not model:
            return False
        cache_key = (self.settings.ollama_base_url, model)
        now = time.monotonic()
        cached = self._ollama_probe
        if cached and cached[0] == cache_key and now - cached[1] < 20:
            return bool(cached[2])
        ready = _probe_ollama(self.settings.ollama_base_url, model)
        self._ollama_probe = (cache_key, now, ready)
        return ready

    def _is_grounded(self, question: str, hit: Hit) -> bool:
        if self.index is None:
            return False
        if lexical_match(
            question,
            hit.chunk.text,
            self.index.bm25.idf,
            min_idf=self.settings.min_idf,
            min_coverage=self.settings.min_coverage,
            sparse_score=hit.sparse_score,
        ):
            return True
        if hit.rerank_score is not None:
            return hit.rerank_score >= self.settings.min_rerank_score
        return hit.dense_score >= self.settings.min_dense_score

    def _apply_reranker(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits or not self.settings.use_reranker:
            return hits
        reranker = self._reranker()
        if reranker is None:
            return hits
        scores = reranker.score(query, [hit.chunk.text for hit in hits])
        ranked = [replace(hit, rerank_score=score) for hit, score in zip(hits, scores)]
        ranked.sort(key=lambda hit: hit.rerank_score if hit.rerank_score is not None else -1e9, reverse=True)
        return ranked

    def _reranker(self) -> CrossEncoderReranker | None:
        if self._reranker_failed:
            return None
        if self._reranker_model is None:
            try:
                self._reranker_model = CrossEncoderReranker(self.settings.reranker_model)
            except Exception:
                logger.warning("Cross-encoder reranker is unavailable", exc_info=True)
                self._reranker_failed = True
                return None
        return self._reranker_model

    def _to_sources(self, hits: list[Hit]) -> tuple[Source, ...]:
        sources: list[Source] = []
        for number, hit in enumerate(hits, start=1):
            sources.append(
                Source(
                    number=number,
                    source=hit.chunk.source,
                    section=hit.chunk.section,
                    page=hit.chunk.page,
                    text=hit.chunk.text,
                    dense_score=hit.dense_score,
                    sparse_score=hit.sparse_score,
                    fused_score=hit.fused_score,
                    rerank_score=hit.rerank_score,
                )
            )
        return tuple(sources)

    def _dense_hits(self, query_vector, pool_limit: int) -> tuple[list[tuple[int, float]] | None, str]:
        if not self.uses_pinecone() or self.index is None:
            return None, "local"
        try:
            self.ensure_pinecone()
            if not self._pinecone_ready:
                return None, "local"
            top_k = min(len(self.index.chunks), max(pool_limit * 5, 20))
            hits = self._vector_client().query(query_vector, self.signature, top_k)
            return hits, "pinecone"
        except Exception:
            logger.warning("Pinecone query failed: %s", _safe_error_name())
            self.vector_store_error = (
                "Pinecone could not be reached, so this search used the local vector index."
            )
            return None, "local"

    def _vector_client(self):
        if self.vector_store is not None:
            return self.vector_store
        if self._pinecone_client is None:
            self._pinecone_client = PineconeVectorStore(self.settings)
        return self._pinecone_client

    def _empty(self, text: str, question: str) -> Answer:
        return Answer(text=text, backend="none", refused=True, standalone_question=question, sources=())


def _safe_error_name() -> str:
    import sys

    exc = sys.exc_info()[1]
    return type(exc).__name__ if exc is not None else "Error"


def _probe_ollama(base_url: str, model: str) -> bool:
    import requests

    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    try:
        response = requests.get(root + "/api/tags", timeout=0.5)
        response.raise_for_status()
        names = [item.get("name", "") for item in response.json().get("models", [])]
    except Exception:
        return False
    if any(name == model or name.startswith(model + ":") or name.split(":")[0] == model for name in names):
        return True
    lowered = model.lower()
    return lowered.endswith("-cloud") or lowered.endswith(":cloud")
