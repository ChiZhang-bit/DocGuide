"""Reciprocal-rank fusion of independently ranked retrieval routes."""
from __future__ import annotations


def reciprocal_rank_score(ranks, rrf_k):
    """Only ranks from routes that actually contain a candidate contribute."""
    return sum(1.0 / (rrf_k + rank) for rank in ranks if rank is not None)


def fuse_rankings(rankings, rrf_k):
    """Return candidate scores, leaving deterministic tie ordering to the caller."""
    ids = set().union(*(set(ranking) for ranking in rankings))
    return {item: reciprocal_rank_score((ranks.get(item) for ranks in rankings), rrf_k)
            for item in ids}
