"""Local structural checks on the candidate's located upstream support."""
from __future__ import annotations

from utils.common import canonical, require
from scripts.reasoning_verification.answer import is_abstention
from utils.data.sources import resolve_reference

class LocalVerifier:
    """Check only the recorded ancestors of the final answer in its working view."""
    def verify(self, *, question, prediction, working_view):
        issues, visited, active, checked_alignments, coverage = [], set(), set(), set(), {}

        def issue(code, severity, target, detail):
            issues.append(dict(code=code, severity=severity, target=target, detail=detail))

        def index(items, kind):
            result = {}
            for item in items:
                key = item["id"]
                if key in result:
                    issue("duplicate_id", "invalid", key, kind)
                result[key] = item
            return result

        nodes = index(working_view.get("nodes", []), "node")
        alignments = index(working_view.get("alignments", []), "alignment")
        receipts = index(working_view.get("receipts", []), "receipt")

        def check_ref(ref, target):
            if ref.get("node_id") not in nodes:
                issue("reference_outside_view", "insufficient", target, str(ref.get("node_id")))
                return
            try:
                resolved = resolve_reference(ref, nodes)
                for field in ("value", "original_span"):
                    if field in ref:
                        require(ref[field] == resolved.get(field), f"Stale reference {field}")
            except (ValueError, KeyError, TypeError) as exc:
                issue("invalid_reference", "invalid", target, str(exc))

        roots = prediction.get("used_node_ids", [])
        if prediction.get("candidate_node_id") in nodes:
            roots = [prediction["candidate_node_id"]]
        require(isinstance(roots, list) and all(isinstance(x, str) for x in roots), "Invalid answer citations")
        if prediction.get("invalid_used_node_ids"):
            issue("invalid_answer_citations", "invalid", "answer", canonical(prediction["invalid_used_node_ids"]))
        if prediction.get("candidate_provenance_error"):
            issue("invalid_candidate_provenance", "invalid", "answer", prediction["candidate_provenance_error"])
        if not roots:
            issue("no_answer_evidence", "insufficient", "answer", "No recorded final-answer inputs")
        if is_abstention(prediction.get("answer", "")):
            issue("no_concrete_answer", "insufficient", "answer", "No concrete candidate to check")
        if prediction.get("basis") in {"model_prior", "cached_candidate"}:
            issue("unsupported_answer_basis", "insufficient", "answer", prediction["basis"])
        if prediction.get("evidence_sufficient") is not True:
            issue("self_reported_insufficient", "insufficient", "answer", "Answer did not report sufficient evidence")

        stack = [(nid, False) for nid in reversed(roots)]
        while stack:
            nid, exiting = stack.pop()
            if exiting:
                active.discard(nid)
                visited.add(nid)
                continue
            if nid in active:
                issue("dependency_cycle", "invalid", nid, "Cycle in recorded reasoning inputs")
                continue
            if nid in visited:
                continue
            if nid not in nodes:
                issue("dependency_outside_view", "insufficient", nid, "Required input absent from final working view")
                visited.add(nid)
                continue
            node = nodes[nid]
            active.add(nid)
            stack.append((nid, True))
            if node.get("status") in {"invalidated", "rejected", "pending_revalidation"}:
                issue("unusable_node", "invalid", nid, node["status"])
            if node["type"] in {"table_record", "text_record"}:
                continue
            require(node["type"] == "derived_fact", "Unknown evidence node type")
            inputs = node.get("inputs", [])
            require(isinstance(inputs, list) and all(isinstance(x, str) for x in inputs), "Invalid fact inputs")
            if not inputs:
                issue("no_fact_inputs", "insufficient", nid, "Derived result lacks recorded inputs")
            if node.get("answer_status") != "answered":
                issue("incomplete_fact", "insufficient", nid, str(node.get("answer_status")))
            refs = node.get("evidence_refs", [])
            located = {r.get("node_id") for r in refs}
            for ref in refs:
                check_ref(ref, nid)
                if ref.get("node_id") not in inputs:
                    issue("undeclared_reference_input", "invalid", nid, str(ref.get("node_id")))
            for parent in inputs:
                if parent in nodes:
                    other = nodes[parent]
                    if other.get("source_version") != node.get("source_version"):
                        issue("source_version_mismatch", "invalid", nid, parent)
                    if other["type"] != "derived_fact" and parent not in located:
                        issue("unlocated_raw_input", "insufficient", nid, parent)
            for aid in node.get("used_alignment_ids", []):
                alignment = alignments.get(aid)
                if alignment is None:
                    issue("alignment_outside_view", "insufficient", nid, aid)
                    continue
                if alignment.get("status") != "accepted":
                    issue("unusable_alignment", "invalid", nid, aid)
                if node.get("alignment_revisions", {}).get(aid) != alignment.get("revision"):
                    issue("stale_alignment_revision", "invalid", nid, aid)
                support = [alignment["left"], alignment["right"], *alignment.get("evidence_refs", [])]
                for ref in support:
                    if ref.get("node_id") not in inputs:
                        issue("undeclared_alignment_input", "invalid", nid, str(ref.get("node_id")))
                    if aid not in checked_alignments:
                        check_ref(ref, aid)
                checked_alignments.add(aid)
            for rid in node.get("access_receipt_ids", []):
                receipt = receipts.get(rid)
                if receipt is None:
                    issue("receipt_outside_view", "insufficient", nid, rid)
                    continue
                if receipt.get("source_version") != node.get("source_version"):
                    issue("receipt_version_mismatch", "invalid", nid, rid)
                coverage[rid] = {k: receipt[k] for k in ("coverage", "selection_limited", "truncated", "outcome") if k in receipt}
            stack.extend((parent, False) for parent in reversed(inputs))

        structural = "failed" if any(i["severity"] == "invalid" for i in issues) else "incomplete" if issues else "passed"
        return dict(backend="local_provenance_v1", structural_status=structural,
            semantic_status="not_checked", verified=False, issues=issues,
            checked_node_ids=sorted(visited & nodes.keys()), checked_alignment_ids=sorted(checked_alignments),
            receipt_coverage=coverage,
            unchecked=["semantic_support", "entity_identity", "original_question_path_and_branch_completeness",
                       "population_coverage", "arithmetic_and_units", "final_answer_located_references"])
