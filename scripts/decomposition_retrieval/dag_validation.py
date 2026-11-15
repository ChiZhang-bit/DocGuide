"""Validate dependency references, execution requests, and acyclic task plans transactionally."""
from __future__ import annotations

import copy
import re
from utils.common import require

class TaskPlanValidation:
    @staticmethod
    def _ordered(tasks):
        pending = list(tasks)
        done, ordered = set(), []
        while pending:
            ready = [qid for qid in pending if set(tasks[qid]["depends_on"]) <= done]
            require(ready, "Task dependency cycle")
            for qid in ready:
                pending.remove(qid)
                done.add(qid)
                ordered.append(qid)
        return ordered


    def stage(self, proposals, round_id):
        """Validate on a copy and return a transaction without changing the DAG."""
        require(isinstance(proposals, list) and proposals, "A retrieval round needs subquestions")
        tasks, aliases, current = copy.deepcopy(self.tasks), {}, []
        replaced = set()
        for i, proposal in enumerate(proposals, 1):
            require(isinstance(proposal, dict), "Subquestion must be an object")
            qid = f"r{round_id}q{i}"
            require(qid not in tasks, "Task ID already exists")
            alias = proposal.get("id", qid)
            require(isinstance(alias, str) and alias and alias not in aliases and alias not in self.tasks,
                    "Duplicate/invalid current task alias")
            aliases[alias] = qid
            current.append(qid)
        for qid in current:
            require(qid not in aliases or aliases[qid] == qid, "Task alias shadows a canonical ID")
            aliases[qid] = qid
        for i, qid in enumerate(current, 1):
            aliases.setdefault(f"q{i}", qid)
        for qid, proposal in zip(current, proposals):
            require(isinstance(proposal.get("question"), str) and proposal["question"].strip(), "Missing subquestion")
            retrievals = proposal.get("retrievals", [])
            require(isinstance(retrievals, list), "retrievals must be a list")
            for request in retrievals:
                require(isinstance(request, dict) and request.get("kind") in
                        {"table_query", "table_retrieve", "text_retrieve", "text_read"}, "Invalid task retrieval kind")
                require(isinstance(request.get("query_spec", {}), dict), "Invalid task query_spec")
                spec = request.get("query_spec", {})
                if "top_k" in spec:
                    require(type(spec["top_k"]) is int and spec["top_k"] > 0, "Invalid task top_k")
                if request["kind"] == "table_query":
                    require(isinstance(spec.get("filters", []), list) and isinstance(spec.get("order_by", []), list),
                            "Invalid table query execution plan")
                if request["kind"] == "text_read":
                    require(isinstance(spec.get("document_ids"), list) and spec["document_ids"],
                            "text_read needs document_ids")
            parents = proposal.get("depends_on", [])
            require(isinstance(parents, list) and all(isinstance(p, str) for p in parents), "depends_on must be task IDs")
            parents = list(dict.fromkeys(aliases.get(p, p) for p in parents))
            require(qid not in parents, "Task cannot depend on itself")
            require(all(p in self.tasks or p in current for p in parents), "Unknown task dependency")
            for parent in parents:
                if parent in self.tasks:
                    require(self.tasks[parent]["status"] in {"answered", "partial"} and
                            self.tasks[parent].get("answer_fact_id"), "Dependency result is not available")
            replacements = proposal.get("replaces_task_ids", [])
            require(isinstance(replacements, list) and all(isinstance(p, str) for p in replacements),
                    "replaces_task_ids must be task IDs")
            replacements = list(dict.fromkeys(replacements))
            require(all(p in self.tasks and self.tasks[p]["status"] == "invalidated" for p in replacements),
                    "Only invalidated historical tasks can be replaced")
            require(not replaced.intersection(replacements), "A historical task can be replaced only once per plan")
            replaced.update(replacements)
            if retrievals and any(p in current for p in parents):
                unbound = re.compile(r"\$\{[^{}]+\}|\{(?:r\d+)?q\d+(?:\.[\w]+)?\}|\$(?:r\d+)?q\d+(?:\.[\w]+)?")
                def has_binding(value):
                    if isinstance(value, str):
                        return bool(unbound.search(value))
                    if isinstance(value, dict):
                        return any(has_binding(v) for v in value.values())
                    if isinstance(value, list):
                        return any(has_binding(v) for v in value)
                    return False
                require(not has_binding(retrievals),
                        "A search needing an unbound parent answer must be planned next round")
            tasks[qid] = dict(id=qid, question=proposal["question"].strip(), depends_on=parents,
                              retrievals=copy.deepcopy(retrievals), status="planned", answer_fact_id=None,
                              round=round_id, dependency_sources={p: "router" for p in parents})
            if replacements:
                tasks[qid]["replaces_task_ids"] = replacements
                for old in replacements:
                    tasks[old].setdefault("replaced_by_task_ids", []).append(qid)
            if retrievals and any(p in current for p in parents):
                tasks[qid]["literal_retrieval_prefetch"] = True
        order = self._ordered(tasks)
        return dict(tasks=tasks, questions=[copy.deepcopy(tasks[q]) for q in order if q in current])
