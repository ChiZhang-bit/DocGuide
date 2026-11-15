"""Node serialization, local semantic scoring, and threshold decisions."""
from __future__ import annotations

import math
from utils.common import canonical, require
from utils.model.reranker import get_shared_reranker

def sigmoid(raw_score):
    score = float(raw_score)
    require(math.isfinite(score), "Non-finite reranker score")
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exp_score = math.exp(score)
    return exp_score / (1.0 + exp_score)


def serialize_node(node, row_formatter=None):
    """Original content and claim text only, without grades, reasons or scores."""
    kind = node["type"]
    if kind == "table_record":
        if row_formatter:
            return row_formatter(node["fields"], node.get("row_index", 0) + 1)
        return "Table row: " + canonical(node["fields"])
    if kind == "text_record":
        return node["raw_text"]
    require(kind == "derived_fact", "Unsupported selection node type")
    return "Question: " + str(node["predicate"]) + "\nAnswer: " + canonical(node["value"]) + \
           "\nQualifiers: " + canonical(node.get("qualifiers", {}))


def score_semantic_pairs(pairs, config, scorer=None):
    """Score all node/demand pairs in one local reranker batch sequence."""
    errors, raw_scores = [], [None] * len(pairs)
    if pairs:
        try:
            scorer = scorer if scorer is not None else get_shared_reranker(config)
            values = list(scorer.score_pairs(pairs, batch_size=config.reranker_batch_size))
            require(len(values) == len(pairs), "Reranker score count mismatch")
            raw_scores = values
        except Exception as exc:
            errors.append(f"Reranker unavailable: {exc}")
    return raw_scores, errors


def semantic_assessments(node_ids, demands, raw_scores, threshold):
    """Threshold each raw logit into a per-demand relevance decision."""
    assessments, errors = {}, []
    for index, nid in enumerate(node_ids):
        scores = []
        for j, demand in enumerate(demands):
            raw = raw_scores[index * len(demands) + j]
            raw_value = normalized = local_bit = None
            if raw is not None:
                try:
                    normalized = sigmoid(raw)
                    raw_value = float(raw)
                    local_bit = int(normalized >= threshold)
                except (TypeError, ValueError, OverflowError) as exc:
                    errors.append(f"{nid}/{demand['id']}: {exc}")
            scores.append(dict(demand_id=demand["id"], raw_score=raw_value,
                               sigmoid_score=normalized, grade=local_bit))
        local_bits = [s["grade"] for s in scores]
        reranker_bit = 1 if 1 in local_bits else None if None in local_bits else 0
        assessments[nid] = dict(scores=scores, grade=reranker_bit)
    return assessments, errors
