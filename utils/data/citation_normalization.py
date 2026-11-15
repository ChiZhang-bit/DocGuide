"""Recover exact citation spans from visible source text."""
from __future__ import annotations

import copy
import re

from .references import resolve_reference


def _tokens(text):
    return list(re.finditer(r"\w+|[^\w\s]", text))


def _valid(ref, nodes):
    try:
        resolved = resolve_reference(ref, nodes)
        return all(ref[k] == resolved.get(k) for k in ("value", "original_span") if k in ref)
    except (ValueError, KeyError, TypeError):
        return False


def _recover(ref, nodes):
    if not isinstance(ref, dict) or _valid(ref, nodes):
        return None
    rid = ref.get("node_id")
    node = nodes.get(rid) if isinstance(rid, str) else None
    if node is None or node.get("status") in {"invalidated", "rejected", "pending_revalidation"}:
        return None
    replacement = dict(ref)
    if node["type"] == "table_record":
        col = ref.get("column")
        if not isinstance(col, str) or col in node["fields"]:
            return None
        tokens = [m.group() for m in _tokens(col)]
        matches = [c for c in node["fields"] if tokens and [m.group() for m in _tokens(c)] == tokens]
        if len(matches) != 1:
            return None
        replacement["column"] = matches[0]
        kind = "column_whitespace"
    elif node["type"] == "text_record":
        quote = ref.get("quote")
        if not isinstance(quote, str) or not quote.strip():
            return None
        if "occurrence" in ref and (type(ref["occurrence"]) is not int or ref["occurrence"] != 0):
            return None
        wanted = [m.group() for m in _tokens(quote)]
        if not wanted:
            return None
        candidates = []
        for sentence in node["sentences"]:
            spans = _tokens(sentence["text"])
            values = [m.group() for m in spans]
            for start in range(len(values) - len(wanted) + 1):
                if values[start:start + len(wanted)] == wanted:
                    end = start + len(wanted) - 1
                    exact = sentence["text"][spans[start].start():spans[end].end()]
                    candidates.append((sentence["id"], exact))
        located = [c for c in candidates if c[0] == ref.get("sentence_id")]
        candidates = located or candidates
        if len(candidates) != 1:
            return None
        replacement.update(sentence_id=candidates[0][0], quote=candidates[0][1])
        kind = "quote_whitespace" if replacement["sentence_id"] == ref.get("sentence_id") else "sentence_locator"
    else:
        return None
    return (replacement, kind) if _valid(replacement, nodes) else None


def normalize_candidate_references(prediction, working_view):
    """Return a copy, with exact recovered references and an audit of each edit."""
    result = copy.deepcopy(prediction)
    refs = result.get("evidence_refs")
    if not isinstance(refs, list):
        return result
    nodes = {n["id"]: n for n in working_view.get("nodes", [])}
    edits = []
    for index, ref in enumerate(refs):
        recovered = _recover(ref, nodes)
        if recovered is not None:
            replacement, kind = recovered
            edits.append(dict(index=index, kind=kind, original=copy.deepcopy(ref), normalized=copy.deepcopy(replacement)))
            refs[index] = replacement
    if edits:
        result["citation_normalizations"] = edits
    return result
