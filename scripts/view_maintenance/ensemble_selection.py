"""Combine logical and semantic relevance decisions without discarding disagreement."""
from __future__ import annotations

from utils.common import require, stable_id
from scripts.view_maintenance.logical_selection import logical_grade
from scripts.view_maintenance.semantic_selection import (
    score_semantic_pairs, semantic_assessments, serialize_node,
)

class DualRouteSelector:
    """Retain a node when either route finds it useful, or either vote is unknown."""

    def __init__(self, config, scorer=None, row_formatter=None):
        require(config.reranker_threshold is not None, "Selection needs an explicit reranker threshold")
        self.config, self.scorer = config, scorer
        self.row_formatter = row_formatter

    def select(self, question, subquestions, nodes, llm_grades, pinned_ids=()):
        demands = [dict(id="original", query=question)]
        demands.extend(dict(id=q["id"], query=q["question"]) for q in subquestions)
        node_ids = sorted(nodes)
        texts = {nid: serialize_node(nodes[nid], self.row_formatter) for nid in node_ids}
        pairs = [(d["query"], texts[nid]) for nid in node_ids for d in demands]
        threshold = self.config.reranker_threshold
        raw_scores, errors = score_semantic_pairs(pairs, self.config, self.scorer)
        semantic, semantic_errors = semantic_assessments(node_ids, demands, raw_scores, threshold)
        errors.extend(semantic_errors)
        assessments, grades = {}, {}
        pinned = set(pinned_ids)
        for nid in node_ids:
            scores = semantic[nid]["scores"]
            reranker_bit = semantic[nid]["grade"]
            llm_bit = logical_grade(llm_grades.get(nid))
            disagreement = (llm_bit != reranker_bit) if llm_bit is not None and reranker_bit is not None else None
            keep = nid in pinned or llm_bit != 0 or reranker_bit != 0
            grades[nid] = int(keep)
            reason = ("pinned" if nid in pinned else "unknown_keep" if None in (llm_bit, reranker_bit)
                      else "disagreement_keep" if disagreement else "agree_keep" if keep else "agree_dormant")
            assessments[nid] = dict(llm_grade=llm_bit, reranker_grade=reranker_bit,
                disagreement=disagreement, scores=scores, final_grade=int(keep), reason=reason)
        return dict(enabled=True, policy="llm_or_reranker", comparison_scope="node_usefulness_over_demands",
            backend=getattr(self.scorer, "backend_name", "transformers_sequence_classification"),
            model=self.config.reranker_model_path, threshold=threshold, transform="sigmoid(raw_logit)",
            demands=demands, demand_snapshot_id=stable_id("D_", demands),
            node_snapshot_id=stable_id("N_", texts), assessments=assessments, grades=grades,
            selected_ids=[nid for nid in node_ids if grades[nid]], pinned_ids=sorted(pinned),
            disagreement_ids=[nid for nid, a in assessments.items() if a["disagreement"] is True],
            unknown_ids=[nid for nid, a in assessments.items() if a["disagreement"] is None],
            errors=errors, pair_count=len(pairs))
