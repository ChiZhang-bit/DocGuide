"""Close task/evidence dependencies and discard obsolete planning history."""
from __future__ import annotations



def _repair_closure(graph, task_dag, task_ids, node_ids, alignment_ids):
    """Close both dependency graphs; either graph may contain the missing edge."""
    tasks, nodes = set(task_ids), set(node_ids)
    while True:
        old = (set(tasks), set(nodes))
        if task_dag is not None:
            tasks = task_dag.descendants(tasks)
            nodes.update(t["answer_fact_id"] for qid, t in task_dag.tasks.items()
                         if qid in tasks and t.get("answer_fact_id"))
        for fid, fact in graph.nodes.items():
            if fact["type"] != "derived_fact" or not graph.usable(fid):
                continue
            producers = set(fact.get("produced_by_task_ids", []))
            if (tasks.intersection(producers) or nodes.intersection(fact.get("inputs", []))
                    or set(alignment_ids).intersection(fact.get("used_alignment_ids", []))):
                nodes.add(fid)
            if fid in nodes and task_dag is not None:
                tasks.update(producers.intersection(task_dag.tasks))
        if task_dag is not None:
            tasks.update(qid for qid, task in task_dag.tasks.items()
                         if task.get("answer_fact_id") in nodes)
        if old == (tasks, nodes):
            return tasks, nodes


def invalidate_history(history, affected, affected_task_ids=()):
    """Keep audit traces elsewhere; do not feed obsolete answers/bindings to Router."""
    affected = set(affected)
    affected_task_ids = set(affected_task_ids)
    for entry in history:
        changed = False
        for q in entry.get("subquestions", []):
            if q.get("answer_fact_id") in affected or q.get("id") in affected_task_ids:
                old_id = q.get("answer_fact_id")
                q.update(status="unanswered", answer=None, answer_fact_id=None,
                         invalidated_answer_fact_id=old_id, missing="Recompute: previous result or its dependency failed verification")
                changed = True
        if changed:
            entry["retrievals"] = []
            entry["can_answer_original"] = False
