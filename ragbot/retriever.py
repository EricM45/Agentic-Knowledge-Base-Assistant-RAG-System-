"""Hybrid retrieval: cosine similarity, BM25, and reciprocal rank fusion."""

from __future__ import annotations

import numpy as np

from ragbot.bm25 import BM25Index
from ragbot.types import Chunk, Hit


class CorpusIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if len(chunks) != len(matrix):
            raise ValueError("Embedding rows do not match the number of chunks.")
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        self.chunks = list(chunks)
        self.embeddings = _normalize_rows(matrix)
        self.bm25 = BM25Index([chunk.text for chunk in self.chunks])

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        limit: int,
        dense_hits: list[tuple[int, float]] | None = None,
    ) -> list[Hit]:
        if not self.chunks or limit <= 0:
            return []
        query_vector = _normalize_vector(query_vector)
        if dense_hits is None:
            dense = self.embeddings @ query_vector
            dense_rank = np.argsort(-dense).astype(int).tolist()
        else:
            dense = np.zeros(len(self.chunks), dtype=np.float32)
            dense_rank = []
            for row, score in dense_hits:
                if 0 <= row < len(self.chunks):
                    dense[row] = score
                    dense_rank.append(row)
        sparse = self.bm25.score(query)
        sparse_rank = [int(index) for index in np.argsort(-sparse) if sparse[index] > 0]
        rankings = [dense_rank]
        if sparse_rank:
            rankings.append(sparse_rank)
        fused = dict(reciprocal_rank_fusion(rankings))
        order = sorted(fused, key=lambda doc_id: fused[doc_id], reverse=True)
        hits: list[Hit] = []
        for row in order[:limit]:
            hits.append(
                Hit(
                    chunk=self.chunks[row],
                    dense_score=float(dense[row]),
                    sparse_score=float(sparse[row]),
                    fused_score=float(fused[row]),
                )
            )
        return hits


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Fuse ranked lists. Rank 0 is the best item in each list."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def maximal_marginal_relevance(
    vectors: np.ndarray,
    relevance: np.ndarray,
    top_k: int,
    lambda_mult: float = 0.7,
) -> list[int]:
    """Pick relevant passages while penalizing near-duplicates."""
    count = len(relevance)
    if count == 0 or top_k <= 0:
        return []
    selected: list[int] = []
    remaining = set(range(count))
    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda index: float(relevance[index]))
        else:

            def score(index: int) -> float:
                redundancy = max(float(vectors[index] @ vectors[chosen]) for chosen in selected)
                return lambda_mult * float(relevance[index]) - (1.0 - lambda_mult) * redundancy

            best = max(remaining, key=score)
        selected.append(best)
        remaining.remove(best)
    return selected


def dedupe_hits(hits: list[Hit], embeddings: np.ndarray, threshold: float = 0.92) -> list[Hit]:
    kept: list[Hit] = []
    for hit in hits:
        vector = embeddings[hit.chunk.index]
        if any(float(vector @ embeddings[other.chunk.index]) >= threshold for other in kept):
            continue
        kept.append(hit)
    return kept


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        return vector
    return vector / norm
