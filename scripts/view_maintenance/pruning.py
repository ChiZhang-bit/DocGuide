"""Apply local relevance grades while retaining aggregate and pinned support."""
from __future__ import annotations

from scripts.view_maintenance.logical_selection import logical_grade

class PruningOperations:
    def apply_relevance_pruning(self, response, allowed, aliases, no_fact_aliases, prune, errors):
        """Apply one reasoning response after aggregate dependencies are registered."""
        protected = self.protected_ids()
        candidates = self.pruning_candidates(allowed) if prune else set()
        ratings = response.get("grades", {})
        if not isinstance(ratings, dict):
            ratings = {}
            errors.append("grades must be an object")
        ratings = {aliases.get(k,k):v for k,v in ratings.items()}
        llm_grades = {}
        for nid in allowed:
            level = ratings.get(nid)
            llm_grades[nid] = logical_grade(level)
            if nid not in candidates:
                self.grades[nid] = 1
                continue
            if type(level) is not int or level not in (0,1):
                level = 1
                errors.append(f"Missing/invalid grade: {nid}; retained")
            self.grades[nid] = level
        for alias in response.get("original_relevant_ids", []):
            if alias in no_fact_aliases:
                continue
            nid = aliases.get(alias, alias)
            if nid in allowed:
                self.pinned.add(nid)
            else:
                errors.append(f"Unknown original-question relevant node: {nid}")
        for nid in self.pinned:
            self.grades[nid] = 1
        candidates.difference_update(self.pinned)
        self.visible = {nid for nid in allowed if self.grades[nid] == 1} | self.pinned | protected
        self.maintain_view()
        return protected, candidates, llm_grades

    def retain_on_error(self, window_ids):
        self.visible = set(window_ids) | self.pinned
        self.maintain_view()
        for nid in self.visible:
            self.grades[nid] = 1
