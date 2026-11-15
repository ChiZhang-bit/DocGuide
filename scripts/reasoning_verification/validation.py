"""Validate all proposed repair targets and continuation tasks before mutation."""
from __future__ import annotations

import copy
from utils.common import require
from utils.prompt_builder import compact_task_dag
from scripts.decomposition_retrieval.task_dag import TaskDAG
from scripts.reasoning_verification.response import CHECKS, EDIT_FIELDS

def _validate_subquestions(proposals, tasks, nodes, invalid_tasks):
    """Validate the whole continuation before deleting any existing result."""
    for q in proposals:
        require(isinstance(q, dict) and isinstance(q.get("question"), str) and q["question"].strip(), "Repair needs a subquestion")
        require(isinstance(q.get("retrievals", []), list), "Repair retrievals must be a list")
        revisions = q.get("replaces_task_ids", [])
        require(isinstance(revisions, list) and all(isinstance(t, str) and t in tasks for t in revisions), "Unknown revised task")
        for request in q.get("retrievals", []):
            require(isinstance(request, dict) and request.get("kind") in {"table_query", "table_retrieve", "text_retrieve", "text_read"}, "Invalid repair retrieval")
            spec = request.get("query_spec", {})
            require(isinstance(spec, dict), "Invalid repair query_spec")
            if request["kind"] in {"table_retrieve", "text_retrieve"}:
                require(isinstance(spec.get("query", ""), str), "Invalid repair query")
            if request["kind"] == "table_query":
                require(isinstance(spec.get("filters", []), list) and isinstance(spec.get("order_by", []), list), "Invalid repair table query")
                for condition in spec.get("filters", []):
                    require(isinstance(condition, dict) and isinstance(condition.get("column"), str) and
                            condition.get("operator") in {"eq", "ne", "in", "contains", "gt", "ge", "lt", "le"} and
                            condition.get("datatype", "text") in {"text", "number", "date"}, "Invalid repair filter")
                for ordering in spec.get("order_by", []):
                    require(isinstance(ordering, dict) and isinstance(ordering.get("column"), str) and
                            ordering.get("direction", "asc") in {"asc", "desc"} and
                            ordering.get("datatype", "text") in {"text", "number", "date"}, "Invalid repair ordering")
                require(spec.get("return_mode", "all_matches") in {"all_matches", "top_k", "exact"}, "Invalid repair return_mode")
            if request["kind"] == "text_read":
                available = {n["document_id"] for n in nodes.values() if n.get("document_id")}
                require(isinstance(spec.get("document_ids"), list) and spec["document_ids"] and
                        all(isinstance(d, str) and d in available for d in spec["document_ids"]), "Unknown repair document")
    if not proposals:
        return
    staged = TaskDAG()
    staged.tasks = copy.deepcopy(tasks)
    for task in staged.tasks.values():
        task.setdefault("depends_on", [])
        task.setdefault("status", "unanswered")
        task.setdefault("answer_fact_id", None)
        if task["id"] in invalid_tasks:
            task.update(status="invalidated", answer_fact_id=None)
    next_round = max((int(t.get("round", 0)) for t in tasks.values()), default=0) + 1
    while any(f"r{next_round}q{i}" in tasks for i in range(1, len(proposals)+1)):
        next_round += 1
    staged.stage(proposals, next_round)


def validate_semantic_response(response, working_view, task_dag=None):
    """Validate every edit target and continuation before any graph mutation."""
    require(isinstance(response, dict), "Verification response must be an object")
    for key in CHECKS:
        require(key in response and (type(response[key]) is bool or response[key] is None), f"Invalid {key}: use boolean or null")
    nodes = {n["id"]: n for n in working_view["nodes"]}
    alignments = {a["id"] for a in working_view.get("alignments", [])}
    tasks = {t["id"]: t for t in compact_task_dag(task_dag)["nodes"]}
    issues = response.get("issues", [])
    require(type(response.get("repair_required", False)) is bool, "repair_required must be a boolean")
    require(isinstance(issues, list), "Verification issues must be a list")
    for issue in issues:
        require(isinstance(issue, dict) and isinstance(issue.get("reason"), str) and issue["reason"].strip(), "Missing issue reason")
        if "kind" in issue:
            require(issue["kind"] in {"reasoning_error", "alignment_error", "path_error", "missing_step", "missing_evidence", "answer_error"}, "Unknown issue kind")
        require(issue.get("severity", "uncertain") in {"clear", "uncertain"}, "Unknown semantic issue severity")
        targets = issue.get("target_ids", [])
        require(isinstance(targets, list) and all(isinstance(t, str) and (t in nodes or t in alignments or t in tasks or t == "answer") for t in targets), "Unknown issue target")
    plan = {k: copy.deepcopy(response.get(k, [])) for k in EDIT_FIELDS}
    require(all(isinstance(v, list) for v in plan.values()), "Repair edits must be arrays")
    revised_tasks = []
    for proposal in plan["subquestions"]:
        require(isinstance(proposal, dict), "Repair needs a subquestion object")
        revisions = proposal.get("replaces_task_ids", [])
        require(isinstance(revisions, list) and all(isinstance(t, str) and t in tasks for t in revisions),
                "Unknown revised task")
        revised_tasks.extend(revisions)
    plan["invalid_task_ids"] = list(dict.fromkeys(plan["invalid_task_ids"] + revised_tasks))
    require(all(isinstance(t, str) and t in tasks for t in plan["invalid_task_ids"]), "Unknown task edit target")
    for nid in plan["invalid_node_ids"]:
        require(isinstance(nid, str) and nid in nodes and nodes[nid]["type"] == "derived_fact", "Only visible derived results may be invalidated")
    require(all(isinstance(a, str) and a in alignments for a in plan["invalid_alignment_ids"]), "Unknown alignment edit target")
    aliases = set()
    for correction in plan["corrections"]:
        require(isinstance(correction, dict), "Correction must be an object")
        cid = correction.get("id")
        require(isinstance(cid, str) and cid and cid not in nodes and cid not in aliases, "Invalid correction alias")
        require(isinstance(correction.get("question"), str) and correction["question"].strip(), "Correction needs a subquestion")
        inputs = correction.get("inputs", [])
        require(isinstance(inputs, list) and inputs and all(isinstance(n, str) and (n in nodes or n in aliases) for n in inputs), "Invalid correction inputs/order")
        require(isinstance(correction.get("evidence_refs", []), list) and isinstance(correction.get("used_alignment_ids", []), list), "Invalid correction citations")
        aliases.add(cid)
    invalid_tasks, invalid_nodes = set(plan["invalid_task_ids"]), set(plan["invalid_node_ids"])
    changed = True
    while changed:
        before = (len(invalid_tasks), len(invalid_nodes))
        invalid_nodes.update(n["id"] for n in nodes.values() if n.get("type") == "derived_fact" and
            (set(n.get("inputs", [])) & invalid_nodes or set(n.get("used_alignment_ids", [])) & set(plan["invalid_alignment_ids"])))
        invalid_tasks.update(t["id"] for t in tasks.values() if t.get("answer_fact_id") in invalid_nodes or
                             set(t.get("depends_on", [])) & invalid_tasks)
        invalid_nodes.update(t["answer_fact_id"] for t in tasks.values() if t["id"] in invalid_tasks and t.get("answer_fact_id"))
        changed = before != (len(invalid_tasks), len(invalid_nodes))
    local_ids = {q.get("id") for q in plan["subquestions"] if isinstance(q.get("id"), str)}
    for proposal in plan["subquestions"]:
        parents = proposal.get("depends_on", [])
        require(isinstance(parents, list) and all(isinstance(p, str) for p in parents),
                "Repair dependencies must be task IDs")
        converted = []
        for parent in parents:
            if parent in tasks or parent in local_ids:
                converted.append(parent)
                continue
            if parent in nodes and nodes[parent]["type"] == "derived_fact":
                owners = [t["id"] for t in tasks.values() if t.get("answer_fact_id") == parent and
                          t.get("status") in {"answered", "partial"} and t["id"] not in invalid_tasks]
                require(parent not in invalid_nodes and len(owners) == 1,
                        "Repair fact dependency needs exactly one surviving task owner")
                parent = owners[0]
            converted.append(parent)
        proposal["depends_on"] = list(dict.fromkeys(converted))
    _validate_subquestions(plan["subquestions"], tasks, nodes, invalid_tasks)
    passed = all(response[k] is True for k in CHECKS)
    require(not passed or (not issues and not any(plan.values()) and not response.get("repair_required", False)), "Passing verdict conflicts with issues/repair edits")
    require(passed or issues, "Failed/unknown verification must identify an issue")
    return plan
