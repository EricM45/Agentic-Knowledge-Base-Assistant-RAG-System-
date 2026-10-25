"""Split documents on sentence boundaries, carrying a short overlap forward."""

from __future__ import annotations

from ragbot.nlp import split_sentences
from ragbot.types import Chunk, Document


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        cleaned = " ".join(text.split())
        return [cleaned] if cleaned else []
    chunks: list[str] = []
    current: list[str] = []

    def emit(parts: list[str]) -> None:
        chunk = " ".join(parts).strip()
        if chunk and (not chunks or chunks[-1] != chunk):
            chunks.append(chunk)

    for sentence in sentences:
        sentence = " ".join(sentence.split())
        if not sentence:
            continue
        if len(sentence) > size:
            if current:
                emit(current)
                current = []
            chunks.extend(_split_long_sentence(sentence, size))
            continue
        if current and len(" ".join(current + [sentence])) > size:
            emit(current)
            current = _tail(current, overlap)
            if current and len(" ".join(current + [sentence])) > size:
                current = []
        current.append(sentence)
    if current:
        emit(current)
    return chunks


def _tail(parts: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return []
    kept: list[str] = []
    for sentence in reversed(parts):
        trial = [sentence, *kept]
        if kept and len(" ".join(trial)) > overlap:
            break
        kept.append(sentence)
    kept.reverse()
    return kept


def _split_long_sentence(sentence: str, size: int) -> list[str]:
    words = sentence.split()
    if not words:
        return []
    parts: list[str] = []
    current: list[str] = []
    for word in words:
        if current and len(" ".join(current + [word])) > size:
            parts.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_documents(documents: list[Document], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for piece in chunk_text(document.text, size, overlap):
            text = f"{document.section}\n{piece}".strip() if document.section else piece
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=text,
                    source=document.source,
                    section=document.section,
                    page=document.page,
                )
            )
    return chunks
