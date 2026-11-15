"""Invalidate faulty branches atomically and retain reusable results for continuation."""
from __future__ import annotations

import copy
from utils.common import require
from scripts.decomposition_retrieval.task_dag import TaskDAG
from scripts.reasoning_verification.invalidation import _repair_closure, invalidate_history

def repair_feedback(report):
    issues = report.get("issues", [])
    if report.get("repair_policy"):
        issues = [i for i in issues if i.get("severity") == "clear"]
    return [(i.get("reason") or f"{i.get('code')}: {i.get('target')}: {i.get('detail')}")[:240]
            for i in issues[:2]]


def apply_repair(graph, report, prediction, allow_corrections=True, task_dag=None, next_round=None):
    """Validate repair targets and continuation tasks, then update the working subgraph."""
    plan = report.get("repair_plan", {})
    require(isinstance(plan, dict), "Repair plan must be an object")
    bad_nodes = plan.get("invalid_node_ids", [])
    bad_alignments = plan.get("invalid_alignment_ids", [])
    bad_tasks = plan.get("invalid_task_ids", [])
    subquestions = plan.get("subquestions", [])
    require(all(isinstance(v, list) for v in (bad_nodes, bad_alignments, bad_tasks, subquestions)),
            "Repair targets and subquestions must be arrays")
    visible = set(graph.visible)
    for nid in bad_nodes:
        require(isinstance(nid, str) and nid in visible and graph.usable(nid)
                and graph.nodes[nid]["type"] == "derived_fact", "Repair cannot delete raw/dormant evidence")
    for aid in bad_alignments:
        require(isinstance(aid, str) and aid in {a["id"] for a in graph.view()["alignments"]},
                "Repair cannot edit an invisible alignment")
    require(not bad_tasks or task_dag is not None, "Task repair needs a task DAG")
    for qid in bad_tasks:
        require(isinstance(qid, str) and qid in task_dag.tasks
                and task_dag.tasks[qid]["status"] != "invalidated", "Unknown/inactive task repair target")
    reason = "; ".join(repair_feedback(report))
    candidate = prediction.get("candidate_node_id")
    if candidate in graph.nodes:
        require(graph.nodes[candidate]["type"] == "derived_fact", "Final candidate must be a derived result")
    roots = set(bad_nodes) | ({candidate} if candidate in graph.nodes and graph.usable(candidate) else set())
    affected_tasks, affected_nodes = _repair_closure(graph, task_dag, bad_tasks, roots, bad_alignments)

    prospective = copy.deepcopy(task_dag) if task_dag is not None else TaskDAG()
    prospective.invalidate(affected_tasks, reason)
    prospective.sync(graph)
    if next_round is None:
        next_round = max((t.get("round", 0) for t in prospective.tasks.values()), default=0) + 1
    require(type(next_round) is int and next_round > 0, "Invalid repair round")
    if subquestions:
        prospective.stage(subquestions, next_round)

    affected = set()
    for aid in bad_alignments:
        affected.update(graph.invalidate_alignment(aid, reason))
    for nid in sorted(affected_nodes):
        if not graph.usable(nid):
            affected.add(nid)
            continue
        affected.update(graph.invalidate(nid, reason))
    if task_dag is not None:
        task_dag.invalidate(affected_tasks, reason)
        task_dag.sync(graph)
    graph.maintain_view()
    aliases, corrections, errors = {}, [], []
    if allow_corrections and plan.get("corrections"):
        staged = copy.deepcopy(graph)
        try:
            for index, proposal in enumerate(plan["corrections"], 1):
                fid = staged._add_answer(dict(proposal, status="answered"),
                    dict(id=f"repair{staged.revision}q{index}", question=proposal["question"]),
                    set(staged.visible), aliases, list(staged.receipts), set())
                aliases[proposal["id"]] = fid
                staged.visible.add(fid)
                staged.grades[fid] = 1
                corrections.append(dict(id=fid, question=proposal["question"], value=staged.nodes[fid]["value"]))
            staged.maintain_view()
            for key in ("nodes", "alignments", "receipts", "visible", "pinned", "grades", "revision", "events"):
                setattr(graph, key, getattr(staged, key))
        except (ValueError, KeyError, TypeError) as exc:
            aliases, corrections = {}, []
            errors.append(f"Rejected correction: {exc}")
    return dict(invalidated_node_ids=sorted(affected), invalidated_alignment_ids=list(bad_alignments),
        invalidated_task_ids=sorted(affected_tasks), next_round=next_round,
        corrections=corrections, correction_aliases=aliases, errors=errors,
        subquestions=copy.deepcopy(subquestions), repaired_working_view=graph.view())
