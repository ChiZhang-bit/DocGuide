"""Compact prompt views; full provenance and original records remain in storage."""
from __future__ import annotations

import copy


def compact_receipt(receipt):
    """Summarize executed queries and coverage for prompts."""
    if not isinstance(receipt, dict):
        return copy.deepcopy(receipt)
    result = {k:copy.deepcopy(v) for k,v in receipt.items()
              if k not in {"id", "source_version", "returned_ids", "code_attempt"}}
    if "returned_ids" in receipt:
        result["returned_count"] = len(receipt["returned_ids"])
    if "code_attempt" in receipt:
        result["code_attempt"] = compact_receipt(receipt["code_attempt"])
    return result


def compact_retrieval_result(summary):
    """Keep one executed-query record, with the requested query when it differs."""
    result = copy.deepcopy(summary)
    receipt = result.get("receipt")
    if isinstance(receipt, dict):
        result["receipt"] = receipt = compact_receipt(receipt)
        if result.get("executed_query") == receipt.get("executed_query"):
            result.pop("executed_query", None)
        if result.get("requested_query") == receipt.get("executed_query"):
            result.pop("requested_query", None)
        if result.get("requested_kind") == receipt.get("mode"):
            result.pop("requested_kind", None)
        if result.get("returned_count") == receipt.get("returned_count"):
            result.pop("returned_count", None)
    elif result.get("requested_query") == result.get("executed_query"):
        result.pop("requested_query", None)
    result.pop("new_node_count", None)
    return result


def compact_ref(ref):
    if not isinstance(ref, dict):
        return ref
    return {k:copy.deepcopy(ref[k]) for k in ("node_id", "column", "sentence_id", "quote", "occurrence") if k in ref}


def compact_graph(snapshot):
    result = dict(nodes=[], alignments=[], receipts=[])
    for node in snapshot.get("nodes", []):
        if node["type"] == "table_record":
            keys = ("id", "type", "table_id", "row_id", "fields")
        elif node["type"] == "text_record":
            keys = ("id", "type", "document_id")
        else:
            keys = ("id", "type", "operation", "predicate", "value", "datatype", "qualifiers", "status",
                    "inputs", "used_alignment_ids", "answer_status")
        item = {k:copy.deepcopy(node[k]) for k in keys if k in node}
        if node["type"] == "text_record":
            item["sentences"] = [{"id":s["id"], "text":s["text"]} for s in node["sentences"]]
        for key in ("subject_ref", "attribute_subject_ref"):
            if node.get(key):
                item[key] = compact_ref(node[key])
        if "evidence_refs" in node:
            item["evidence_refs"] = [compact_ref(r) for r in node["evidence_refs"]]
        result["nodes"].append(item)
    for alignment in snapshot.get("alignments", []):
        item = {k:copy.deepcopy(alignment[k]) for k in ("id", "status", "basis", "reason") if k in alignment}
        item.update(left=compact_ref(alignment["left"]), right=compact_ref(alignment["right"]),
                    evidence_refs=[compact_ref(r) for r in alignment["evidence_refs"]])
        result["alignments"].append(item)
    for receipt in snapshot.get("receipts", []):
        result["receipts"].append(compact_receipt(receipt))
    if "view_states" in snapshot:
        visible = {n["id"] for n in result["nodes"]}
        result["view_states"] = {k:dict(state=v["state"], reason=v["reason"])
                                 for k,v in snapshot["view_states"].items() if k in visible}
    return result


def compact_tasks(tasks):
    keys = ("id", "kind", "goal", "status", "depends_on", "input_bindings", "query_spec",
            "completion_criteria", "coverage_requirement", "output_refs", "unresolved")
    return [{k:copy.deepcopy(t[k]) for k in keys if k in t} for t in tasks]


def compact_task_dag(task_dag):
    """Keep the complete task plan, without execution logs or repeated edges."""
    if task_dag is None:
        return dict(nodes=[])
    if hasattr(task_dag, "snapshot"):
        task_dag = task_dag.snapshot()
    tasks = task_dag.get("nodes", []) if isinstance(task_dag, dict) else task_dag
    keys = ("id", "question", "depends_on", "status", "answer_fact_id", "round",
            "replaces_task_ids")
    return dict(nodes=[{k:copy.deepcopy(t[k]) for k in keys if k in t} for t in tasks])


def compact_payload(stage, payload):
    result = dict(payload)
    current_receipt_ids = {s["receipt"]["id"] for s in result.get("retrieval_results", [])
        if isinstance(s.get("receipt"), dict) and "id" in s["receipt"]} if stage == "integrate" else set()
    for key in ("graph", "working_view", "proof"):
        if isinstance(result.get(key), dict) and "nodes" in result[key]:
            snapshot = result[key]
            if stage == "integrate" and key == "working_view" and current_receipt_ids:
                snapshot = dict(snapshot, receipts=[r for r in snapshot.get("receipts", [])
                    if r.get("id") not in current_receipt_ids])
            result[key] = compact_graph(snapshot)
    if isinstance(result.get("retrieval_results"), list):
        result["retrieval_results"] = [compact_retrieval_result(s) for s in result["retrieval_results"]]
    if stage == "integrate":
        result.pop("retrieved_node_ids", None)
        result.pop("new_node_ids", None)
    if isinstance(result.get("subquestion_history"), list):
        result["subquestion_history"] = copy.deepcopy(result["subquestion_history"])
        for entry in result["subquestion_history"]:
            if isinstance(entry.get("retrievals"), list):
                entry["retrievals"] = [compact_retrieval_result(s) for s in entry["retrievals"]]
    if stage == "verify" and isinstance(result.get("working_view"), dict):
        nodes = result["working_view"].get("nodes", [])
        for node in nodes:
            for key in ("support_assessment", "produced_by_task_ids", "revision", "alignment_revisions", "invalidation_reason"):
                node.pop(key, None)
            if node.get("operation") == "final_answer":
                node.pop("predicate", None)
        candidate = result.get("candidate", {})
        if any(n["id"] == candidate.get("candidate_node_id") and n.get("operation") == "final_answer" for n in nodes):
            result["candidate"] = {k:copy.deepcopy(candidate[k]) for k in ("answer", "candidate_node_id") if k in candidate}
        result["local_issues"] = [copy.deepcopy(i) for i in result.get("local_issues", [])
            if i.get("code") not in {"self_reported_insufficient", "unsupported_answer_basis", "incomplete_fact"}]
        if "task_dag" in result:
            result["task_dag"] = compact_task_dag(result["task_dag"])
    if isinstance(result.get("tasks"), list):
        result["tasks"] = compact_tasks(result["tasks"])
    if isinstance(result.get("demands"), dict):
        result["demands"] = copy.deepcopy(result["demands"])
        for demand in result["demands"]["demands"]:
            demand.pop("input_bindings", None)
            demand.pop("output_refs", None)
    return result
