"""Restrict pruning to unprotected usable raw evidence."""
from __future__ import annotations



class PruningCandidates:
    def pruning_candidates(self, ids):
        protected = self.protected_ids() | self.pinned
        return {n for n in ids if self.usable(n) and n not in protected and
                self.nodes[n]["type"] in {"table_record", "text_record"}}
