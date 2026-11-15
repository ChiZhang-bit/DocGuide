"""BM25 scoring and stable ranking helpers."""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Sequence
from utils.data.records import SearchHit

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "with",
}

def tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(str(text))
        if token.lower() not in STOPWORDS
    ]

def finite_score(score: float) -> float:
    value = float(score)
    return value if math.isfinite(value) else float("-inf")

def rank_scores_descending(scores: Sequence[float]) -> list[int]:
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: (-finite_score(scores[index]), index),
    )
    ranks = [0] * len(scores)
    for position, index in enumerate(ranked_indices, start=1):
        ranks[index] = position
    return ranks

class BM25Retriever:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document) for document in self.documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        self.document_frequencies: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.document_frequencies.update(set(tokens))

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        if top_k <= 0 or not self.documents:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return [SearchHit(doc_id=index, score=0.0) for index in range(min(top_k, len(self.documents)))]

        scores = [self._score_doc(query_terms, doc_id) for doc_id in range(len(self.documents))]
        ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        return [SearchHit(doc_id=index, score=float(scores[index])) for index in ranked[:top_k]]

    def _score_doc(self, query_terms: list[str], doc_id: int) -> float:
        score = 0.0
        doc_len = self.doc_lengths[doc_id] or 1
        term_frequency = self.term_frequencies[doc_id]
        total_docs = len(self.documents)
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            df = self.document_frequencies.get(term, 0)
            idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
            denom = frequency + self.k1 * (1.0 - self.b + self.b * doc_len / (self.avgdl or 1.0))
            score += idf * frequency * (self.k1 + 1.0) / denom
        return score
