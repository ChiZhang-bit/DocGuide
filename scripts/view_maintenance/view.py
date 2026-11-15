"""Maintain the visible subview and expose its complete located support."""
from __future__ import annotations



class WorkingView:
    def maintain_view(self):
        """View maintenance, not the verifier, retains necessary support ancestors."""
        self.pinned = {n for n in self.pinned if self.usable(n)}
        self.visible = {n for n in self.visible | self.pinned | self.protected_ids() if self.usable(n)}
        added, queue = set(), list(self.visible) if self.preserve_dependencies else []
        seen = set()
        while queue:
            nid = queue.pop()
            if nid in seen:
                continue
            seen.add(nid)
            node = self.nodes[nid]
            needed = set(node.get("inputs", []))
            for aid in node.get("used_alignment_ids", []):
                a = self.alignments.get(aid)
                if a and a["status"] == "accepted":
                    needed.update(r["node_id"] for r in [a["left"], a["right"], *a["evidence_refs"]])
            for parent in needed:
                if self.usable(parent):
                    if parent not in self.visible:
                        self.visible.add(parent)
                        self.grades[parent] = 1
                        added.add(parent)
                    queue.append(parent)
        return sorted(added)


    def view(self, ids=None):
        ids = {n for n in (self.visible if ids is None else set(ids)) if self.usable(n)}
        alignments = [aid for aid,a in self.alignments.items()
                      if a["status"] == "accepted" and
                      {a["left"]["node_id"], a["right"]["node_id"]}.issubset(ids) and
                      all(r["node_id"] in ids for r in a["evidence_refs"])]
        receipt_ids = {qid for nid in ids for qid in self.nodes[nid].get("access_receipt_ids", [])}
        return self.snapshot(dict(node_ids=sorted(ids), alignment_ids=sorted(alignments), receipt_ids=sorted(receipt_ids)))
