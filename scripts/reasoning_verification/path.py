"""Extract the candidate's full upstream reasoning and alignment evidence path."""
from __future__ import annotations

import copy

def extract_reasoning_path(prediction, working_view):
    """Select the candidate's full upstream closure, including alignment support."""
    nodes = {n["id"]: n for n in working_view.get("nodes", [])}
    alignments = {a["id"]: a for a in working_view.get("alignments", [])}
    root = prediction.get("candidate_node_id")
    roots = [root] if root in nodes else list(prediction.get("used_node_ids", []))
    selected, selected_alignments, receipt_ids = set(), set(), set()
    pending = list(roots)

    def use_alignment(aid):
        if aid in selected_alignments or aid not in alignments:
            return
        selected_alignments.add(aid)
        alignment = alignments[aid]
        refs = [alignment.get("left", {}), alignment.get("right", {}), *alignment.get("evidence_refs", [])]
        pending.extend(r.get("node_id") for r in refs if isinstance(r, dict))
        receipt_ids.update(alignment.get("access_receipt_ids", []))

    for aid in prediction.get("used_alignment_ids", []):
        use_alignment(aid)
    while pending:
        nid = pending.pop()
        if nid in selected or nid not in nodes:
            continue
        selected.add(nid)
        node = nodes[nid]
        pending.extend(node.get("inputs", []))
        pending.extend(r.get("node_id") for r in node.get("evidence_refs", []) if isinstance(r, dict))
        receipt_ids.update(node.get("access_receipt_ids", []))
        for aid in node.get("used_alignment_ids", []):
            use_alignment(aid)
    result = dict(nodes=[copy.deepcopy(n) for n in working_view.get("nodes", []) if n["id"] in selected],
                  alignments=[copy.deepcopy(a) for a in working_view.get("alignments", []) if a["id"] in selected_alignments],
                  receipts=[copy.deepcopy(r) for r in working_view.get("receipts", []) if r["id"] in receipt_ids])
    if "derive_edges" in working_view:
        result["derive_edges"] = [copy.deepcopy(e) for e in working_view["derive_edges"]
            if e.get("source_node_id") in selected and e.get("target_fact_id") in selected]
    return result
