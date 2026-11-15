"""Raw node insertion, candidate materialization, and derived-node invalidation."""
from __future__ import annotations

import copy
from utils.common import require

class NodeOperations:
    def upsert(self, result: dict, task_id: str):
        touched, added = [], []
        receipt = copy.deepcopy(result["receipt"])
        self.receipts[receipt["id"]] = receipt
        for record in result["records"]:
            rid = record["id"]
            require(record == self.sources.records[rid], "Attempt to overwrite raw source")
            if rid not in self.nodes:
                self.nodes[rid] = copy.deepcopy(record) | {"observed_task_ids": []}
                added.append(rid)
            if task_id not in self.nodes[rid]["observed_task_ids"]:
                self.nodes[rid]["observed_task_ids"].append(task_id)
            if self.preserve_dependencies:
                ids = self.nodes[rid].setdefault("access_receipt_ids", [])
                if receipt["id"] not in ids:
                    ids.append(receipt["id"])
            touched.append(rid)
        self.revision += bool(added)
        return dict(added=added, touched=touched)


    def invalidate(self, node_id, reason):
        require(node_id in self.nodes and self.nodes[node_id]["type"] == "derived_fact", "Only derived results can be invalidated")
        affected, queue = set(), [node_id]
        while queue:
            item = queue.pop()
            if item in affected:
                continue
            affected.add(item)
            node = self.nodes.get(item)
            if node and node["type"] == "derived_fact":
                node["status"] = "invalidated"
                node["invalidation_reason"] = reason
            queue.extend(k for k,v in self.nodes.items() if item in v.get("inputs", []))
        self.events.append(dict(event="invalidation", affected=sorted(affected), reason=reason))
        self.visible.difference_update(affected)
        self.pinned.difference_update(affected)
        self.grades.update({n: 0 for n in affected})
        self.revision += 1
        return affected


    def add_candidate(self, prediction):
        """Materialize the final composition using the same provenance contract."""
        proposal = dict(value=prediction["answer"], status="answered", inputs=prediction["used_node_ids"],
            evidence_refs=prediction.get("evidence_refs", []), used_alignment_ids=prediction.get("used_alignment_ids", []),
            reason=prediction.get("reason") or "Final answer composition")
        fid = self._add_answer(proposal, dict(id="final_answer", question=self.sources.question),
            set(self.visible), {}, list(self.receipts), set(), operation="final_answer")
        self.visible.add(fid)
        self.grades[fid] = 1
        self.maintain_view()
        return fid
