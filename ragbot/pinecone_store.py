"""Pinecone-backed dense retrieval.

Chunk text and the BM25 index stay on disk. Pinecone stores the embedding
vectors and answers the nearest-neighbor query.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from ragbot.config import Settings

logger = logging.getLogger("ragbot.pinecone")

SYNC_FILE = "pinecone.json"


class PineconeVectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None
        self._index = None
        self._dimension: int | None = None

    def sync(
        self,
        chunks,
        embeddings: np.ndarray,
        namespace: str,
        previous_namespace: str | None = None,
    ) -> None:
        if not len(chunks):
            return
        matrix = np.asarray(embeddings, dtype=np.float32)
        index = self._connect(int(matrix.shape[1]))
        if previous_namespace and previous_namespace != namespace:
            _delete_namespace(index, previous_namespace)
        _delete_namespace(index, namespace)
        vectors = [(str(chunk.index), matrix[chunk.index].tolist()) for chunk in chunks]
        response = index.upsert(
            vectors=vectors,
            namespace=namespace,
            batch_size=100,
            show_progress=False,
        )
        if getattr(response, "has_errors", False):
            raise RuntimeError("Pinecone did not accept every passage vector.")
        _wait_until_counted(index, namespace, len(vectors))

    def query(self, vector: np.ndarray, namespace: str, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        values = np.asarray(vector, dtype=np.float32).tolist()
        dimension = len(values)
        index = self._connect(dimension)
        response = index.query(top_k=top_k, vector=values, namespace=namespace, include_metadata=False)
        hits: list[tuple[int, float]] = []
        for match in getattr(response, "matches", []) or []:
            raw_id = match["id"] if isinstance(match, dict) else match.id
            score = match["score"] if isinstance(match, dict) else match.score
            try:
                row = int(raw_id)
            except (TypeError, ValueError):
                continue
            hits.append((row, float(score)))
        return hits

    def _connect(self, dimension: int):
        if self._index is not None and self._dimension == dimension:
            return self._index
        if not self.settings.pinecone_api_key:
            raise RuntimeError("Pinecone API key is missing.")
        from pinecone import Pinecone, ServerlessSpec

        client = Pinecone(api_key=self.settings.pinecone_api_key)
        name = self.settings.pinecone_index
        if not client.has_index(name):
            client.create_index(
                name=name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
        description = client.describe_index(name)
        actual_dimension = _index_dimension(description)
        actual_metric = _index_metric(description)
        if actual_dimension is not None and actual_dimension != dimension:
            raise RuntimeError(
                f"Pinecone index '{name}' stores {actual_dimension}-dimensional vectors, "
                f"but this embedding model uses {dimension}. Set PINECONE_INDEX to a new name."
            )
        if actual_metric and actual_metric.lower() not in {"cosine", "dotproduct"}:
            raise RuntimeError(
                f"Pinecone index '{name}' uses the {actual_metric} metric. "
                "Use a cosine or dotproduct index for this chatbot."
            )
        self._client = client
        self._dimension = dimension
        self._index = client.index(name=name)
        return self._index


def read_sync_record(directory: Path) -> dict | None:
    path = directory / SYNC_FILE
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_sync_record(directory: Path, *, signature: str, index_name: str, namespace: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"signature": signature, "index_name": index_name, "namespace": namespace}
    (directory / SYNC_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sync_matches(record: dict | None, signature: str, index_name: str) -> bool:
    if not record:
        return False
    return record.get("signature") == signature and record.get("index_name") == index_name and bool(
        record.get("namespace")
    )


def _delete_namespace(index, namespace: str) -> None:
    from pinecone.errors.exceptions import NotFoundError

    try:
        index.delete(delete_all=True, namespace=namespace)
    except NotFoundError:
        return
    except Exception as exc:
        message = str(exc).lower()
        if "not found" in message or "namespace" in message and "404" in message:
            return
        raise


def _wait_until_counted(index, namespace: str, expected: int) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if _namespace_count(index.describe_index_stats(), namespace) >= expected:
            return
        time.sleep(1)
    logger.warning("Pinecone has not reported %s vectors in namespace %s yet", expected, namespace)


def _namespace_count(stats, namespace: str) -> int:
    namespaces = getattr(stats, "namespaces", None)
    if namespaces is None and isinstance(stats, dict):
        namespaces = stats.get("namespaces", {})
    if not namespaces:
        return 0
    entry = namespaces.get(namespace) if hasattr(namespaces, "get") else None
    if entry is None:
        return 0
    if isinstance(entry, dict):
        return int(entry.get("vector_count", 0))
    return int(getattr(entry, "vector_count", 0))


def _index_dimension(description) -> int | None:
    try:
        value = description.dimension
        if value:
            return int(value)
    except Exception:
        pass
    return _schema_value(description, "dimension")


def _index_metric(description) -> str | None:
    try:
        value = description.metric
        if value:
            return str(value)
    except Exception:
        pass
    metric = _schema_value(description, "metric")
    return str(metric) if metric else None


def _schema_value(description, name: str):
    schema = getattr(description, "schema", None)
    fields = getattr(schema, "fields", None) or {}
    if not isinstance(fields, dict):
        return None
    for field in fields.values():
        value = field.get(name) if isinstance(field, dict) else getattr(field, name, None)
        if value:
            return value
    return None
