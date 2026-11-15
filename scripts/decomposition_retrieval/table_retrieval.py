"""Full-row dense/BM25 retrieval, fused with reciprocal rank fusion."""
from __future__ import annotations

import math
import numpy as np

from utils.common import require
from .bm25 import BM25Retriever, finite_score, rank_scores_descending
from utils.data.records import TableHit


class TableRetriever:
    def __init__(self, rows, *, embedding_model_path="", embedding_device="cpu",
                 embedding_batch_size=16, embedding_max_length=512, encoder=None,
                 rank_weights=(0.5, 0.5, 0.0, 0.0), rank_methods=(1, 1, 0, 0), rrf_k=60):
        self.rows = rows
        require(type(rrf_k) is int and rrf_k > 0, "Invalid RRF constant")
        self.rrf_k, self.last_search = rrf_k, {}
        self.embedding_batch_size = embedding_batch_size
        self.rank_methods = tuple(rank_methods)
        self.rank_weights = tuple(float(w) for w in rank_weights)
        require(self.rank_methods in ((1, 1, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0)),
                "Only full-row dense and BM25 retrieval are supported")
        require(len(self.rank_weights) == 4 and all(math.isfinite(w) and w >= 0 for w in self.rank_weights),
                "Invalid rank weights")
        enabled = [w if active else 0.0 for w, active in zip(self.rank_weights, self.rank_methods)]
        require(sum(enabled) > 0, "Enabled rank weights must be positive")
        self.effective_rank_weights = tuple(w / sum(enabled) for w in enabled)
        self.use_row_embedding, self.use_row_bm25 = map(bool, self.rank_methods[:2])
        self.use_cell_embedding = self.use_cell_match = False
        self.cell_locations = []
        self.cell_embeddings = np.empty((0, 0), dtype=np.float32)
        self.encoder = None
        if self.use_row_embedding:
            if encoder is None:
                from utils.model.embedding import get_shared_encoder
                encoder = get_shared_encoder(embedding_model_path, embedding_device, embedding_max_length)
            self.encoder = encoder
        self.row_embedding_documents = [self._row_embedding_search_text(r) for r in rows] if self.use_row_embedding else []
        self.row_bm25_documents = [self._row_bm25_search_text(r) for r in rows] if self.use_row_bm25 else []
        self.row_bm25_retriever = BM25Retriever(self.row_bm25_documents) if self.use_row_bm25 else None
        self.row_embeddings = (self.encoder.encode(self.row_embedding_documents,
            batch_size=embedding_batch_size, instruction="") if self.use_row_embedding
            else np.empty((len(rows), 0), dtype=np.float32))

    @staticmethod
    def _row_embedding_search_text(row):
        return " ".join(row.description.split()).strip() or row.text

    @staticmethod
    def _row_bm25_search_text(row):
        return " | ".join(f"{column}: {value}" for column, value in row.values.items() if value)

    def search(self, query, top_k, original_question=""):
        if top_k <= 0 or not self.rows:
            return []
        dense_scores = [0.0] * len(self.rows)
        if self.use_row_embedding:
            vector = self.encoder.encode([query], batch_size=1,
                instruction="Represent this sentence for searching relevant passages: ")
            if self.row_embeddings.size and vector.size:
                dense_scores = [finite_score(s) for s in self.row_embeddings @ vector[0]]
        lexical_scores = [0.0] * len(self.rows)
        if self.use_row_bm25:
            for hit in self.row_bm25_retriever.search(query, len(self.rows)):
                lexical_scores[hit.doc_id] = hit.score
        dense_ranks = rank_scores_descending(dense_scores) if self.use_row_embedding else [0] * len(self.rows)
        lexical_ranks = [0] * len(self.rows)
        if self.use_row_bm25:
            lexical_order = sorted((i for i, score in enumerate(lexical_scores) if score > 0),
                                   key=lambda i: (-lexical_scores[i], i))
            for rank, i in enumerate(lexical_order, 1):
                lexical_ranks[i] = rank
        hits = []
        for i, row in enumerate(self.rows):
            score = ((1.0 / (self.rrf_k + dense_ranks[i]) if self.use_row_embedding else 0.0) +
                     (1.0 / (self.rrf_k + lexical_ranks[i])
                      if self.use_row_bm25 and lexical_scores[i] > 0 else 0.0))
            rank_score = 1.0 / score if score else 1e30
            hits.append(TableHit(row=row, score=score, rank_score=rank_score,
                row_embedding_score=dense_scores[i], row_bm25_score=lexical_scores[i],
                row_embedding_rank=dense_ranks[i], row_bm25_rank=lexical_ranks[i],
                cell_match_query_source="disabled"))
        hits.sort(key=lambda hit: (-hit.score, hit.row.row_index))
        selected = hits[:min(top_k, len(hits))]
        self.last_search = dict(method="rrf" if all(self.rank_methods[:2]) else "single_route",
            rrf_k=self.rrf_k, hits=[dict(row_index=h.row.row_index, score=h.score,
                dense_rank=h.row_embedding_rank if self.use_row_embedding else None,
                bm25_rank=h.row_bm25_rank if self.use_row_bm25 and h.row_bm25_score > 0 else None) for h in selected])
        return selected
