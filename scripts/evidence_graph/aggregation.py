"""Materialize aggregate answers and integrate one batched reasoning response."""
from __future__ import annotations

import copy
from utils.common import canonical, require, stable_id
from scripts.reasoning_verification.answer import is_abstention

class AggregationOperations:
    def _add_answer(self, proposal, question, allowed, aliases, receipt_ids, skipped_alignments, operation="subquestion_answer"):
        value = proposal.get("value")
        require(value is not None and not is_abstention(value), "Missing concrete subquestion answer")
        inputs = list(dict.fromkeys(aliases.get(x,x) for x in proposal.get("inputs", [])))
        refs = [self.resolve(r) for r in proposal.get("evidence_refs", [])]
        inputs.extend(r["node_id"] for r in refs)
        aids = list(dict.fromkeys(aliases.get(x,x) for x in proposal.get("used_alignment_ids", [])
                                 if x not in skipped_alignments))
        for aid in aids:
            require(aid in self.alignments and self.alignments[aid]["status"] == "accepted", "Unknown accepted alignment")
            a = self.alignments[aid]
            support = [a["left"], a["right"], *a["evidence_refs"]]
            require(all(r["node_id"] in allowed for r in support), "Alignment outside working evidence")
            refs.extend(support)
            inputs.extend(r["node_id"] for r in support)
        inputs = list(dict.fromkeys(inputs))
        refs = list({canonical(r):r for r in refs}.values())
        require(inputs and all(x in allowed for x in inputs), "Answer must depend on visible/new evidence")
        require(all(self.usable(x) for x in inputs), "Answer cannot depend on invalidated evidence")
        raw_inputs = {i for i in inputs if self.nodes[i]["type"] != "derived_fact"}
        require(raw_inputs <= {r["node_id"] for r in refs}, "Locate each directly used raw input")
        require(proposal.get("reason"), "Answer needs a brief evidence reason")
        fid = stable_id("F_", [self.sources.version, question["question"], value, sorted(inputs), sorted(aids)])
        if operation != "subquestion_answer":
            fid = stable_id("F_", [fid, operation])
        base, supersedes, revision = fid, None, 1
        while fid in self.nodes and not self.usable(fid):
            supersedes = fid
            revision += 1
            fid = stable_id("F_", [base, "revision", revision])
        used_receipts = [qid for qid in receipt_ids if set(inputs) & set(self.receipts[qid].get("returned_ids", []))]
        if fid not in self.nodes:
            self.nodes[fid] = dict(id=fid, type="derived_fact", source_version=self.sources.version,
                operation=operation, predicate=question["question"], value=copy.deepcopy(value),
                inputs=inputs, used_alignment_ids=aids,
                alignment_revisions={a:self.alignments[a]["revision"] for a in aids},
                evidence_refs=refs, access_receipt_ids=used_receipts, status="supported",
                answer_status=proposal["status"], subject_ref=None, qualifiers={},
                support_assessment=dict(method="llm", reason=proposal["reason"], verified=False),
                produced_by_task_ids=[question["id"]], revision=revision)
            if supersedes:
                self.nodes[fid]["supersedes"] = supersedes
            self.events.append(dict(event="subquestion_answer", id=fid, subquestion_id=question["id"]))
        return fid


    def integrate(self, response, questions, window_ids, receipt_ids=(), *, prune=True, task_outputs=None):
        require(isinstance(response, dict), "Integration must return an object")
        allowed = {n for n in window_ids if self.usable(n)}
        by_id = {q["id"]:q for q in questions}
        aliases, errors, outcomes = {}, [], {}
        skipped_alignments, no_fact_aliases = set(), set()
        answers = response.get("answers", [])
        require(isinstance(answers, list) and all(isinstance(p, dict) for p in answers), "answers must be objects")
        pending, answers = list(answers), []
        while pending:
            pending_aliases = {p.get("id") for p in pending if p.get("id")}
            pending_tasks = {p.get("subquestion_id") for p in pending}
            ready = next((p for p in pending if
                not (set(p.get("inputs", [])) & pending_aliases) and
                not (set(by_id.get(p.get("subquestion_id"), {}).get("depends_on", [])) & pending_tasks)), None)
            require(ready is not None, "Cyclic subquestion answer dependencies")
            pending.remove(ready)
            answers.append(ready)
        available_outputs = dict(task_outputs or {})
        useful_alignments = {a for p in answers if p.get("status") in {"answered", "partial"}
                             and p.get("value") is not None and not is_abstention(p["value"])
                             for a in p.get("used_alignment_ids", [])}
        final_answer = response.get("final_answer")
        if isinstance(final_answer, dict) and isinstance(final_answer.get("used_alignment_ids"), list):
            useful_alignments.update(a for a in final_answer["used_alignment_ids"] if isinstance(a, str))
        for p in response.get("alignments", []):
            try:
                require(p.get("id") and p["id"] not in aliases and p["id"] not in self.nodes, "Duplicate alignment alias")
                if p["id"] not in useful_alignments:
                    skipped_alignments.add(p["id"])
                    continue
                require(p.get("subquestion_id") in by_id, "Alignment must serve a current subquestion")
                refs = [p["left"], p["right"], *p.get("evidence_refs", [])]
                require(all(r.get("node_id") in allowed for r in refs), "Alignment outside working evidence")
                if p["left"]["node_id"] == p["right"]["node_id"]:
                    skipped_alignments.add(p["id"])
                    continue
                require({self.nodes[p[side]["node_id"]]["type"] for side in ("left", "right")} ==
                        {"table_record", "text_record"}, "Align a table entity with a text entity")
                aliases[p["id"]] = self.add_alignment(dict(p, status="accepted"))
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
        for p in answers:
            qid = p.get("subquestion_id")
            try:
                require(qid in by_id and qid not in outcomes, "Unknown/duplicate subquestion answer")
                state = p.get("status")
                require(state in {"answered", "partial", "unanswered"}, "Invalid answer status")
                if state != "unanswered" and (p.get("value") is None or is_abstention(p["value"])):
                    state = "unanswered"
                outcome = dict(id=qid, question=by_id[qid]["question"], status=state,
                               answer=None, answer_fact_id=None, missing=p.get("missing", ""))
                if state != "unanswered":
                    parents = by_id[qid].get("depends_on", [])
                    require(all(parent in available_outputs and self.usable(available_outputs[parent]) for parent in parents),
                            "Required task result is not available for reasoning")
                    require(p.get("id") and p["id"] not in aliases and p["id"] not in self.nodes, "Duplicate answer alias")
                    proposal = dict(p, inputs=list(dict.fromkeys(p.get("inputs", []) +
                        [available_outputs[parent] for parent in parents])))
                    fid = self._add_answer(proposal, by_id[qid], allowed, aliases, receipt_ids, skipped_alignments)
                    aliases[p["id"]] = fid
                    allowed.add(fid)
                    outcome.update(answer=copy.deepcopy(self.nodes[fid]["value"]), answer_fact_id=fid)
                    available_outputs[qid] = fid
                elif p.get("id"):
                    no_fact_aliases.add(p["id"])
                outcomes[qid] = outcome
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"{qid}: {exc}")
        for qid,q in by_id.items():
            outcomes.setdefault(qid, dict(id=qid, question=q["question"], status="unanswered",
                                         answer=None, answer_fact_id=None, missing="No valid answer derived in this round"))
        protected, candidates, llm_grades = self.apply_relevance_pruning(
            response, allowed, aliases, no_fact_aliases, prune, errors)
        can_answer = response.get("can_answer_original", False)
        if type(can_answer) is not bool:
            errors.append("can_answer_original must be a JSON boolean; continue retrieval")
        return dict(subquestions=list(outcomes.values()), errors=errors, aliases=aliases,
                    skipped_alignment_ids=sorted(skipped_alignments),
                    pruning_performed=prune, pruning_candidate_ids=sorted(candidates), protected_ids=sorted(protected),
                    llm_grades={n:llm_grades[n] for n in sorted(allowed)},
                    grades={n:self.grades[n] for n in sorted(allowed)}, pinned_ids=sorted(self.pinned),
                    can_answer_original=can_answer is True,
                    candidate_answer=response.get("candidate_answer", ""))
