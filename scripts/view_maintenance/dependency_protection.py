"""Retain live aggregate results and their recursive evidence dependencies."""
from __future__ import annotations



class DependencyProtection:
    def protected_ids(self):
        """All live aggregate results and their recursive evidence dependencies."""
        protected, queue = set(), [n for n, v in self.nodes.items()
                                   if v["type"] == "derived_fact" and self.usable(n)]
        while queue:
            nid = queue.pop()
            if nid in protected or not self.usable(nid):
                continue
            protected.add(nid)
            node = self.nodes[nid]
            queue.extend(node.get("inputs", []))
            for aid in node.get("used_alignment_ids", []):
                alignment = self.alignments.get(aid)
                if alignment and alignment["status"] == "accepted":
                    queue.extend(r["node_id"] for r in
                                 [alignment["left"], alignment["right"], *alignment["evidence_refs"]])
        return protected
