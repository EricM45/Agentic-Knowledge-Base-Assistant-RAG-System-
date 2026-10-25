"""Persist a corpus index so the next launch skips re-embedding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ragbot.config import Settings
from ragbot.retriever import CorpusIndex
from ragbot.types import Chunk, IndexStats


def corpus_signature(paths: list[Path], settings: Settings, model_name: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode())
    digest.update(str(settings.chunk_size).encode())
    digest.update(str(settings.chunk_overlap).encode())
    for path in sorted(paths, key=lambda item: str(item.resolve())):
        digest.update(str(path.resolve()).encode())
        file_hash = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 16), b""):
                file_hash.update(block)
        digest.update(file_hash.digest())
    return digest.hexdigest()


def save_index(
    index: CorpusIndex,
    stats: IndexStats,
    directory: Path,
    signature: str,
    model_name: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "embeddings.npy", index.embeddings)
    payload = {
        "version": 1,
        "signature": signature,
        "model_name": model_name,
        "files": stats.files,
        "sections": stats.sections,
        "skipped": list(stats.skipped),
        "chunks": [chunk.to_dict() for chunk in index.chunks],
    }
    (directory / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_index(directory: Path, signature: str, model_name: str) -> tuple[CorpusIndex, IndexStats] | None:
    manifest_path = directory / "manifest.json"
    matrix_path = directory / "embeddings.npy"
    if not manifest_path.exists() or not matrix_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            return None
        if payload.get("signature") != signature or payload.get("model_name") != model_name:
            return None
        chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        embeddings = np.load(matrix_path)
    except (OSError, ValueError, KeyError):
        return None
    if embeddings.shape[0] != len(chunks):
        return None
    stats = IndexStats(
        files=int(payload.get("files", 0)),
        sections=int(payload.get("sections", 0)),
        chunks=len(chunks),
        skipped=tuple(payload.get("skipped", ())),
    )
    return CorpusIndex(chunks, embeddings), stats
