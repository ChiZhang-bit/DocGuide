from __future__ import annotations

import copy
import re

from utils.common import canonical, require
from utils.prompt_loader import load_prompt

def render_answer(value):
    if isinstance(value, str):
        return value.strip()
    if value is not None and not isinstance(value, dict):
        return canonical(value)
    return ""

def is_abstention(answer):
    text = str(answer).strip().casefold()
    return (text in {"", "unknown", "insufficient", "n/a", "none", "null", "[]"}
            or bool(re.search(r"^(?:not (?:stated|provided|available|known|found|specified)|(?:cannot|can't|unable to) (?:determine|answer|infer)|insufficient (?:evidence|information)|no .{0,60} (?:found|available|provided))(?:\b|$)", text)))


def prepare_fused_answer(response, view, aliases=None, skipped_alignment_ids=()):
    """Parse a final composition with valid structure and visible dependencies."""
    require(isinstance(response, dict), "Fused final_answer must be an object")
    answer = render_answer(response.get("answer"))
    require(not is_abstention(answer), "Fused final_answer needs a concrete answer")
    aliases = aliases or {}
    known = {n["id"] for n in view["nodes"]}
    alignments = {a["id"] for a in view.get("alignments", []) if a.get("status") == "accepted"}

    def ids(field, allowed, skipped=()):
        values = response.get(field, [])
        require(isinstance(values, list) and all(isinstance(x, str) for x in values),
                "Invalid fused " + field)
        values = list(dict.fromkeys(aliases.get(x, x) for x in values if x not in skipped))
        require(all(x in allowed for x in values), "Unknown fused " + field)
        return values

    used = ids("used_node_ids", known)
    aids = ids("used_alignment_ids", alignments, set(skipped_alignment_ids))
    refs = response.get("evidence_refs", [])
    require(isinstance(refs, list) and all(isinstance(ref, dict) for ref in refs),
            "Invalid fused evidence_refs")
    refs = copy.deepcopy(refs)
    for ref in refs:
        rid = ref.get("node_id")
        require(isinstance(rid, str), "Invalid fused reference node")
        ref["node_id"] = aliases.get(rid, rid)
        require(ref["node_id"] in known, "Unknown fused reference node")
    require(used or refs or aids, "Fused final_answer needs supporting evidence")
    require(response.get("confidence") in {"low", "medium", "high"}, "Invalid fused confidence")
    require(response.get("basis") in {"retrieved_evidence", "inference"}, "Unsupported fused answer basis")
    require(type(response.get("evidence_sufficient")) is bool, "Invalid fused evidence assessment")
    require(isinstance(response.get("reason"), str) and response["reason"].strip(),
            "Fused final_answer needs an evidence reason")
    return dict(answer=answer, status="unverified", verification_performed=False,
                used_node_ids=used, invalid_used_node_ids=[], evidence_refs=refs,
                used_alignment_ids=aids, confidence=response["confidence"], basis=response["basis"],
                evidence_sufficient=response["evidence_sufficient"], reason=response["reason"], errors=[])


def answer_from_view(llm, question, view, max_tokens=800, candidate="", repair_feedback=None, max_attempts=2):
    payload = dict(question=question, working_view=view)
    if repair_feedback:
        payload["repair_feedback"] = repair_feedback
    known = {n["id"] for n in view["nodes"]}
    errors = []
    for _ in range(max_attempts):
        try:
            response = llm.ask("answer", load_prompt("answer"), payload, max_tokens=max_tokens)
            answer = render_answer(response.get("answer"))
            require(not is_abstention(answer), "Return a concrete best-effort answer")
            cited = response.get("used_node_ids", [])
            invalid_cited = ([x for x in cited if not isinstance(x, str) or x not in known]
                             if isinstance(cited, list) else [cited])
            cited = cited if isinstance(cited, list) else []
            return dict(answer=answer, status="unverified", verification_performed=False,
                        used_node_ids=list(dict.fromkeys(x for x in cited if isinstance(x,str) and x in known)),
                        invalid_used_node_ids=invalid_cited,
                        evidence_refs=response.get("evidence_refs", []),
                        used_alignment_ids=response.get("used_alignment_ids", []),
                        confidence=response.get("confidence", "low"), basis=response.get("basis", "inference"),
                        evidence_sufficient=bool(response.get("evidence_sufficient", False)),
                        reason=response.get("reason", ""), errors=errors)
        except Exception as exc:
            errors.append(str(exc))
            payload["format_feedback"] = "Give a concrete nonempty best-effort answer. " + str(exc)[:200]
    candidate = render_answer(candidate)
    if not is_abstention(candidate):
        return dict(answer=candidate, status="best_effort", verification_performed=False, used_node_ids=[],
                    confidence="low", basis="cached_candidate", evidence_sufficient=False, errors=errors)
    return dict(answer="UNKNOWN", status="answer_failed", verification_performed=False, used_node_ids=[],
                confidence="low", evidence_sufficient=False, errors=errors)
