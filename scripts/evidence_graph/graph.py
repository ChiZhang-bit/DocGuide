"""One shared state for raw evidence, aggregate nodes, alignments, and the working subview."""
from __future__ import annotations

import copy
from scripts.evidence_graph.nodes import NodeOperations
from scripts.evidence_graph.alignment import AlignmentOperations
from scripts.evidence_graph.aggregation import AggregationOperations
from scripts.evidence_graph.references import GraphReferences
from scripts.view_maintenance.dependency_protection import DependencyProtection
from scripts.view_maintenance.pruning_candidates import PruningCandidates
from scripts.view_maintenance.pruning import PruningOperations
from scripts.view_maintenance.view import WorkingView

class WorkingGraph(NodeOperations, AlignmentOperations, AggregationOperations, GraphReferences,
                   DependencyProtection, PruningCandidates, PruningOperations, WorkingView):
    def __init__(self, sources, preserve_dependencies=True):
        self.sources = sources
        self.preserve_dependencies = preserve_dependencies
        self.nodes, self.alignments, self.receipts = {}, {}, {}
        self.events = []
        self.revision = 0
        self.visible, self.pinned, self.grades = set(), set(), {}


    def usable(self, nid):
        return nid in self.nodes and self.nodes[nid].get("status") not in {"invalidated", "pending_revalidation", "rejected"}


    def snapshot(self, selection=None):
        selected = selection or dict(node_ids=list(self.nodes), alignment_ids=list(self.alignments), receipt_ids=list(self.receipts))
        ids = set(selected["node_ids"])
        return dict(nodes=[copy.deepcopy(self.nodes[k]) for k in selected["node_ids"]],
                    alignments=[copy.deepcopy(self.alignments[k]) for k in selected["alignment_ids"]],
                    derive_edges=[dict(type="DERIVE", source_node_id=i, target_fact_id=k)
                                  for k in ids for i in self.nodes[k].get("inputs", []) if i in ids],
                    receipts=[copy.deepcopy(self.receipts[k]) for k in selected["receipt_ids"]])
