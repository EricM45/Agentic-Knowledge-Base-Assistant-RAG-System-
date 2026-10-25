"""Dense embeddings and an optional cross-encoder reranker.

The neural libraries are imported only when a model is constructed, so unit
tests can run with a stand-in embedder.
"""

from __future__ import annotations

import os

import numpy as np


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        dimension_fn = getattr(self._model, "get_embedding_dimension", None)
        dimension = dimension_fn() if callable(dimension_fn) else None
        if not isinstance(dimension, int):
            raise RuntimeError("The embedding model did not report a vector size.")
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw = self._model.predict([(query, text) for text in texts], show_progress_bar=False)
        return [float(value) for value in np.atleast_1d(raw)]
