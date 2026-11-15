"""Perform one semantic verification and return its compact repair decision."""
from __future__ import annotations

import copy
from typing import Protocol
from utils.common import Config, canonical, require
from utils.prompt_loader import load_prompt
from utils.prompt_builder import compact_task_dag
from scripts.reasoning_verification.local import LocalVerifier
from scripts.reasoning_verification.path import extract_reasoning_path
from scripts.reasoning_verification.response import CHECKS, normalize_semantic_response
from scripts.reasoning_verification.validation import validate_semantic_response
from scripts.reasoning_verification.policy import apply_repair_policy

class Verifier(Protocol):
    def verify(self, *, question: str, prediction: dict, working_view: dict) -> dict: ...


class LLMVerifier:
    def __init__(self, llm, max_tokens=Config.verification_tokens):
        self.llm, self.max_tokens = llm, max_tokens

    def verify(self, *, question, prediction, working_view, task_dag=None):
        local = LocalVerifier().verify(question=question, prediction=prediction, working_view=working_view)
        path = extract_reasoning_path(prediction, working_view)
        candidate = {k: prediction[k] for k in ("answer", "used_node_ids", "evidence_refs", "used_alignment_ids", "candidate_node_id") if k in prediction}
        response = self.llm.ask("verify", load_prompt("verify"), dict(question=question, candidate=candidate,
            task_dag=compact_task_dag(task_dag), working_view=path, local_issues=local["issues"]), max_tokens=self.max_tokens)
        response = normalize_semantic_response(response)
        plan = validate_semantic_response(response, working_view, task_dag)
        advisory = {"self_reported_insufficient", "unsupported_answer_basis", "incomplete_fact"}
        blocking = [i for i in local["issues"] if i["code"] not in advisory]
        structural = "failed" if any(i["severity"] == "invalid" for i in blocking) else "incomplete" if blocking else "passed"
        semantic = "passed" if all(response[k] is True for k in CHECKS) else "failed" if any(response[k] is False for k in CHECKS) else "unknown"
        return dict(backend="llm_path_v2", structural_status=structural, semantic_status=semantic,
            protocol_status=response.get("status"), checks={k: response[k] for k in CHECKS}, issues=response.get("issues", []) + blocking,
            semantic_issues=response.get("issues", []), repair_required=response.get("repair_required", False),
            local_check=local, repair_plan=plan, checked_path_node_ids=[n["id"] for n in path["nodes"]], verified=False)


def run_verification(mode, *, question, prediction, working_view, verifier=None, llm=None,
                     max_tokens=Config.verification_tokens, task_dag=None):
    """Isolate verifier failures and graph mutations."""
    if mode == "off":
        return dict(enabled=False, performed=False, backend="off", structural_status="not_checked",
                    semantic_status="not_checked", verified=False, issues=[])
    require(mode in {"local", "llm"}, "Unknown verification mode")
    try:
        backend = verifier if verifier is not None else LLMVerifier(llm, max_tokens) if mode == "llm" else LocalVerifier()
        p, v = (prediction, working_view) if verifier is None else (copy.deepcopy(prediction), copy.deepcopy(working_view))
        kwargs = dict(question=question, prediction=p, working_view=v)
        if verifier is None and mode == "llm":
            kwargs["task_dag"] = task_dag
        report = backend.verify(**kwargs)
        require(isinstance(report, dict), "Verifier must return an object")
        require(isinstance(report.get("backend"), str) and report["backend"], "Verifier must identify its backend")
        require(report.get("structural_status") in {"passed", "incomplete", "failed", "error"}, "Invalid structural verdict")
        require(report.get("semantic_status") in {"passed", "failed", "unknown", "not_checked"}, "Invalid semantic verdict")
        require(isinstance(report.get("issues"), list), "Verifier issues must be a list")
        if mode == "llm":
            if verifier is not None:
                response = dict(report.get("checks", {}), issues=report.get("semantic_issues", report["issues"]),
                    repair_required=report.get("repair_required", False), **report.get("repair_plan", {}))
                if report.get("protocol_status"):
                    response.update(status=report["protocol_status"], reason=report.get("reason"))
                response = normalize_semantic_response(response)
                report["repair_plan"] = validate_semantic_response(response, working_view, task_dag)
                report["checks"] = {k: response[k] for k in CHECKS}
            require(all(report.get("checks", {}).get(k) is True for k in CHECKS) or report["semantic_status"] != "passed", "Inconsistent semantic status")
            apply_repair_policy(report)
        report["verified"] = report["structural_status"] == report["semantic_status"] == "passed"
        canonical(report)
    except Exception as exc:
        report = dict(backend=type(verifier).__name__ if verifier is not None else "llm_path_v2" if mode == "llm" else "local_provenance_v1",
            structural_status="error", semantic_status="not_checked", verified=False,
            issues=[dict(code="verifier_error", severity="insufficient", target="verification", detail=str(exc))])
    return dict(report, enabled=True, performed=True)
