"""Located table/text entity alignments and alignment invalidation."""
from __future__ import annotations

import re
from utils.common import canonical, require, stable_id

class AlignmentOperations:
    def add_alignment(self, proposal):
        left, right = self.resolve_labeled(proposal["left"], "left"), self.resolve_labeled(proposal["right"], "right")
        require(left["node_id"] != right["node_id"], "Alignment requires different records")
        require(not any(re.fullmatch(r"[\d\s.,+%/:-]+", str(endpoint["value"]).strip()) for endpoint in [left,right]),
                "Numeric equality is not entity identity")
        refs = [self.resolve_labeled(r, f"evidence_refs[{i}]") for i,r in enumerate(proposal.get("evidence_refs", []))]
        status = proposal.get("status", "candidate")
        require(status in {"candidate", "accepted", "rejected", "invalidated"}, "Invalid alignment status")
        if status == "accepted":
            require(refs and proposal.get("basis") and proposal.get("reason"), "Accepted alignment needs located semantic evidence")
        endpoints = sorted([left, right], key=canonical)
        aid = stable_id("A_", endpoints)
        old = self.alignments.get(aid)
        if old and old["status"] == status and old["evidence_refs"] == refs:
            return aid
        if old:
            self.invalidate_alignment(aid, "alignment revised")
        self.alignments[aid] = dict(id=aid, type="ALIGN", left=left, right=right, status=status,
                                   basis=proposal.get("basis", ""), reason=proposal.get("reason", ""),
                                   evidence_refs=refs, revision=old["revision"]+1 if old else 1)
        self.events.append(dict(event="alignment", id=aid, status=status))
        self.revision += 1
        return aid


    def invalidate_alignment(self, aid, reason):
        require(aid in self.alignments, "Unknown alignment")
        self.alignments[aid]["status"] = "invalidated"
        affected = set()
        for fid in list(self.nodes):
            if aid in self.nodes[fid].get("used_alignment_ids", []):
                affected.update(self.invalidate(fid, reason))
        return affected
