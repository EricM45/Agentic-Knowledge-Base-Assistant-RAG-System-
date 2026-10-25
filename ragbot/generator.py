"""Grounded answer generation.

OpenAI and Ollama write the reply when configured. Otherwise a query-focused
selector stitches the best retrieved sentences and cites them.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from ragbot.nlp import overlap_weight, split_sentences, tokenize
from ragbot.retriever import maximal_marginal_relevance
from ragbot.types import Generated, Hit

logger = logging.getLogger("ragbot.generator")

REFUSAL_TEXT = (
    "I couldn't find that in the indexed documents. "
    "Try the wording used in the source files, or add a document that covers the question."
)
MIN_SENTENCE_CHARS = 40
SYSTEM_PROMPT = (
    "You answer questions using only the numbered context passages from the user's documents. "
    "If the passages do not contain the answer, say you cannot find it in the documents. "
    "Cite the passages you use with their numbers, such as [1]. "
    "Prefer exact figures, dates, and names from the passages. "
    "Write plain sentences a person can read quickly in a chat."
)
_MODEL_REFUSAL = re.compile(
    r"cannot find|could not find|can't find|don't know|do not know|not in the (indexed )?documents",
    re.IGNORECASE,
)


class LocalSynthesizer:
    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def synthesize(self, question: str, hits: list[Hit], idf: dict[str, float]) -> str | None:
        rows: list[dict] = []
        query_tokens = list(dict.fromkeys(tokenize(question)))
        for number, hit in enumerate(hits, start=1):
            for sentence_index, sentence in enumerate(split_sentences(hit.chunk.text)):
                if len(sentence) < MIN_SENTENCE_CHARS:
                    continue
                rows.append(
                    {
                        "text": sentence,
                        "number": number,
                        "sentence_index": sentence_index,
                    }
                )
        if not rows:
            return None
        vectors = self.embedder.embed_documents([row["text"] for row in rows])
        query_vector = np.asarray(self.embedder.embed_query(question), dtype=np.float32)
        similarities = vectors @ query_vector
        bonuses = np.asarray(
            [
                sum(
                    overlap_weight(token, set(tokenize(row["text"])), idf)
                    for token in query_tokens
                )
                for row in rows
            ],
            dtype=np.float32,
        )
        relevance = np.asarray(similarities, dtype=np.float32) + 0.08 * bonuses
        chosen = maximal_marginal_relevance(vectors, relevance, top_k=min(3, len(rows)), lambda_mult=0.62)
        if not chosen:
            return None
        best_score = max(float(relevance[index]) for index in chosen)
        kept = [index for index in chosen if float(relevance[index]) >= best_score - 0.2]
        best_index = max(range(len(rows)), key=lambda index: float(relevance[index]))
        if best_index not in kept:
            kept.append(best_index)
        kept.sort(key=lambda index: (rows[index]["number"], rows[index]["sentence_index"]))
        return _cite(rows, kept)


class GroundedGenerator:
    def __init__(self, embedder) -> None:
        self.synthesizer = LocalSynthesizer(embedder)

    def generate(self, question, hits, history, settings, idf: dict[str, float], backend: str) -> Generated:
        warning = None
        if backend in {"openai", "ollama"}:
            try:
                text = self._complete(question, hits, history, settings, backend)
            except Exception:
                logger.warning("Generative model request failed", exc_info=True)
                warning = (
                    "The generative model did not respond, so this answer was written from the retrieved passages."
                )
                text = ""
            if text:
                return Generated(
                    text=text,
                    backend=backend,
                    refused=_MODEL_REFUSAL.search(text) is not None,
                    warning=None,
                )
        synthesized = self.synthesizer.synthesize(question, hits, idf)
        if not synthesized:
            return Generated(text=REFUSAL_TEXT, backend="none", refused=True, warning=warning)
        return Generated(text=synthesized, backend="local-nlp", refused=False, warning=warning)

    def _complete(self, question, hits, history, settings, backend: str) -> str:
        from openai import OpenAI

        if backend == "openai":
            client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=45.0,
            )
            model = settings.openai_model
        else:
            if not settings.ollama_model:
                raise RuntimeError("Set OLLAMA_MODEL before using the Ollama backend.")
            client = OpenAI(
                api_key="ollama",
                base_url=settings.ollama_base_url,
                timeout=180.0,
            )
            model = settings.ollama_model
        completion = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=500,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_context_prompt(question, hits, history)},
            ],
        )
        content = completion.choices[0].message.content or ""
        return content.strip()


def build_context_prompt(question: str, hits: list[Hit], history: list[dict] | None) -> str:
    blocks: list[str] = []
    history_text = _format_history(history or [])
    if history_text:
        blocks.append("Conversation:\n" + history_text)
    passages: list[str] = []
    for number, hit in enumerate(hits, start=1):
        location = hit.chunk.source
        if hit.chunk.section:
            location += f" | {hit.chunk.section}"
        if hit.chunk.page:
            location += f" | p. {hit.chunk.page}"
        text = hit.chunk.text.strip()
        if len(text) > 1200:
            text = text[:1200].rstrip() + "..."
        passages.append(f"[{number}] {location}\n{text}")
    blocks.append("Context passages:\n" + "\n\n".join(passages))
    blocks.append(f"Question: {question}\nAnswer using only the context passages. Cite passage numbers like [1].")
    return "\n\n".join(blocks)


def _format_history(history: list[dict]) -> str:
    lines: list[str] = []
    for item in history[-6:]:
        content = " ".join(str(item.get("content", "")).split())
        if not content:
            continue
        if len(content) > 500:
            content = content[:500] + "..."
        role = "User" if item.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _cite(rows: list[dict], indexes: list[int]) -> str:
    parts: list[str] = []
    buffer: list[str] = []
    current_number = None
    for index in indexes:
        number = rows[index]["number"]
        if current_number is None:
            current_number = number
        if number != current_number and buffer:
            parts.append(f"{' '.join(buffer)} [{current_number}]")
            buffer = []
            current_number = number
        buffer.append(rows[index]["text"])
    if buffer and current_number is not None:
        parts.append(f"{' '.join(buffer)} [{current_number}]")
    return " ".join(parts)
