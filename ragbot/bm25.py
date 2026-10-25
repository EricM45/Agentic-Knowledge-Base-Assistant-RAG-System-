"""Okapi BM25 over stemmed tokens.

score(q, d) = sum IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
"""

from __future__ import annotations

import math

import numpy as np

from ragbot.nlp import tokenize, tokenize_query


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(text) for text in texts]
        self.doc_len = [len(tokens) for tokens in self.docs]
        self.n = len(self.docs)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0
        self.tfs: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}
        for tokens in self.docs:
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.tfs.append(frequencies)
            for token in frequencies:
                document_frequency[token] = document_frequency.get(token, 0) + 1
        self.df = document_frequency
        self.idf = {token: self._idf(frequency) for token, frequency in self.df.items()}

    def _idf(self, document_frequency: int) -> float:
        return math.log(1.0 + (self.n - document_frequency + 0.5) / (document_frequency + 0.5))

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        tokens = tokenize_query(query)
        if not tokens or self.n == 0 or self.avgdl == 0:
            return scores
        for index, frequencies in enumerate(self.tfs):
            length = self.doc_len[index] or 1
            total = 0.0
            for token in tokens:
                frequency = frequencies.get(token, 0)
                if frequency == 0:
                    continue
                numerator = frequency * (self.k1 + 1.0)
                denominator = frequency + self.k1 * (1.0 - self.b + self.b * length / self.avgdl)
                total += self.idf.get(token, 0.0) * numerator / denominator
            scores[index] = total
        return scores
