"""Text recall with independent dense/lexical candidates and bounded graph entry."""
from __future__ import annotations

from dataclasses import dataclass
from .bm25 import BM25Retriever
from .fusion import fuse_rankings

import numpy as np

from utils.common import require


@dataclass(frozen=True)
class SearchHit:
    doc_id: int
    score: float


class TextRetriever:
    def __init__(self, documents, *, method="hybrid", encoder=None, batch_size=16,
                 candidate_k=20, rrf_k=60):
        require(method in {"hybrid", "embedding", "bm25"}, "Unknown text retrieval method")
        require(type(candidate_k) is int and candidate_k > 0, "Invalid text candidate_k")
        require(type(rrf_k) is int and rrf_k > 0, "Invalid RRF constant")
        self.documents, self.method, self.encoder = list(documents), method, encoder
        self.candidate_k, self.rrf_k = candidate_k, rrf_k
        self.last_search = {}
        self.lexical = (BM25Retriever(self.documents)
                        if method in {"hybrid", "bm25"} else None)
        self.embeddings = None
        if method in {"hybrid", "embedding"}:
            require(encoder is not None, "Dense retrieval requires an encoder")
            self.embeddings = encoder.encode(self.documents, batch_size=batch_size, role="corpus")

    def search(self, query, top_k):
        require(isinstance(query, str) and query.strip(), "Missing retrieval query")
        require(type(top_k) is int and top_k > 0, "top_k must be positive")
        count = min(len(self.documents), max(self.candidate_k, top_k))
        dense, lexical = [], []
        if count and self.embeddings is not None:
            vector = self.encoder.encode([query], batch_size=1, role="query")[0]
            scores = self.embeddings @ vector
            dense = [SearchHit(int(i), float(scores[i])) for i in np.argsort(-scores, kind="stable")[:count]]
        if count and self.lexical is not None:
            lexical = [SearchHit(h.doc_id, h.score) for h in self.lexical.search(query, count) if h.score > 0]
        dense_ranks = {h.doc_id: i for i, h in enumerate(dense, 1)}
        lexical_ranks = {h.doc_id: i for i, h in enumerate(lexical, 1)}
        if self.method == "hybrid":
            ids = set(dense_ranks) | set(lexical_ranks)
            fusion = fuse_rankings((dense_ranks, lexical_ranks), self.rrf_k)
            ranked = sorted(ids, key=lambda i: (-fusion[i], i))
            hits = [SearchHit(i, fusion[i]) for i in ranked[:top_k]]
        else:
            hits = (dense if self.method == "embedding" else lexical)[:top_k]
        self.last_search = dict(method=self.method, candidate_k=count, rrf_k=self.rrf_k,
                                preserve_route_leaders=False,
                                hits=[dict(doc_id=h.doc_id, score=h.score,
                                           dense_rank=dense_ranks.get(h.doc_id),
                                           bm25_rank=lexical_ranks.get(h.doc_id)) for h in hits])
        return hits
