"""Task state, dependency synchronization, and reusable answers across rounds."""
from __future__ import annotations

import copy
from utils.common import ContractError, require
from scripts.decomposition_retrieval.dag_validation import TaskPlanValidation

class TaskDAG(TaskPlanValidation):
    def __init__(self):
        self.tasks = {}
        self.events = []


    def descendants(self, task_ids):
        """Return the dependency closure, including its root tasks."""
        affected = set(task_ids)
        require(affected <= set(self.tasks), "Unknown task invalidation target")
        while True:
            expanded = affected | {qid for qid, task in self.tasks.items()
                                   if affected.intersection(task["depends_on"])}
            if expanded == affected:
                return affected
            affected = expanded


    def invalidate(self, task_ids, reason="Verification repair"):
        """Keep obsolete tasks for audit, excluding them from live dependency use."""
        affected = self.descendants(task_ids)
        for qid in affected:
            task = self.tasks[qid]
            fid = task.get("answer_fact_id")
            if fid:
                task["invalidated_answer_fact_id"] = fid
            task.update(status="invalidated", answer_fact_id=None, invalidation_reason=reason)
        if affected:
            self.events.append(dict(event="invalidation", task_ids=sorted(affected), reason=reason))
        return affected


    def active_order(self):
        return [qid for qid in self._ordered(self.tasks)
                if self.tasks[qid]["status"] != "invalidated"]


    def commit(self, transaction):
        self.tasks = transaction["tasks"]
        return [{k: copy.deepcopy(q[k]) for k in ("id", "question", "retrievals", "depends_on", "replaces_task_ids") if k in q}
                for q in transaction["questions"]]


    def sync(self, graph):
        invalid = set()
        for task in self.tasks.values():
            if task["status"] == "invalidated":
                continue
            fid = task.get("answer_fact_id")
            if fid and not graph.usable(fid):
                if graph.nodes.get(fid, {}).get("status") == "invalidated":
                    invalid.add(task["id"])
                else:
                    task.update(status="unanswered", answer_fact_id=None)
        if invalid:
            self.invalidate(invalid, "Task result or its dependency was invalidated")


    def outputs(self):
        return {qid: t["answer_fact_id"] for qid, t in self.tasks.items()
                if t.get("answer_fact_id") and t["status"] in {"answered", "partial"}}


    def record_outcomes(self, outcomes, graph):
        for outcome in outcomes:
            qid = outcome["id"]
            if qid in self.tasks and self.tasks[qid]["status"] != "invalidated":
                self.tasks[qid].update(status=outcome["status"], answer_fact_id=outcome.get("answer_fact_id"))
        for outcome in outcomes:
            qid, fid = outcome["id"], outcome.get("answer_fact_id")
            if (qid not in self.tasks or self.tasks[qid]["status"] == "invalidated"
                    or not fid or not graph.usable(fid)):
                continue
            for parent_fid in graph.nodes[fid].get("inputs", []):
                for parent in graph.nodes[parent_fid].get("produced_by_task_ids", []):
                    if (parent not in self.tasks or self.tasks[parent]["status"] == "invalidated"
                            or parent == qid or parent in self.tasks[qid]["depends_on"]):
                        continue
                    self.tasks[qid]["depends_on"].append(parent)
                    try:
                        self._ordered(self.tasks)
                    except ContractError:
                        self.tasks[qid]["depends_on"].remove(parent)
                        self.events.append(dict(event="inferred_edge_rejected", source=parent, target=qid,
                                                fact_id=fid, input_fact_id=parent_fid))
                    else:
                        self.tasks[qid]["dependency_sources"][parent] = "fact_input"
        self.sync(graph)
        for outcome in outcomes:
            parents = self.tasks.get(outcome["id"], {}).get("depends_on", [])
            if parents:
                outcome["depends_on"] = list(parents)


    def snapshot(self):
        return dict(nodes=copy.deepcopy(list(self.tasks.values())),
                    edges=[dict(source=p, target=qid, basis=t["dependency_sources"].get(p, "router"))
                           for qid, t in self.tasks.items() for p in t["depends_on"]],
                    active_task_ids=self.active_order(),
                    active_edges=[dict(source=p, target=qid, basis=t["dependency_sources"].get(p, "router"))
                                  for qid, t in self.tasks.items() if t["status"] != "invalidated"
                                  for p in t["depends_on"] if self.tasks[p]["status"] != "invalidated"],
                    events=copy.deepcopy(self.events))
