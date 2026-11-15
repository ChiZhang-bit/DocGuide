"""Normalize PASS, UNCERTAIN, and actionable REPAIR verdicts."""
from __future__ import annotations

import copy
from utils.common import require

CHECKS = ("reasoning_correct", "path_correct", "path_complete", "answer_supported")
EDIT_FIELDS = ("invalid_task_ids", "invalid_node_ids", "invalid_alignment_ids", "corrections", "subquestions")

def normalize_semantic_response(response):
    """Read compact verification verdicts."""
    if response == "PASS":
        return dict(status="PASS", **{k: True for k in CHECKS})
    if not isinstance(response, dict):
        return response
    response = copy.deepcopy(response)
    if "edits" in response:
        proposals = response.pop("edits")
        task_fields = {"id", "question", "depends_on", "retrievals", "replaces_task_ids"}
        require(isinstance(proposals, list) and all(isinstance(q, dict) and
                isinstance(q.get("question"), str) and q["question"].strip() and
                isinstance(q.get("retrievals", []), list) and set(q) <= task_fields
                for q in proposals), "edits must contain continuation task objects")
        require("subquestions" not in response or response["subquestions"] == proposals,
                "Conflicting edits and subquestions")
        response["subquestions"] = proposals
    status = response.get("status")
    if status is not None:
        require(status in {"PASS", "REPAIR", "UNCERTAIN"}, "Unknown verification status")
        if status == "PASS":
            require(all(response.get(k, True) is True for k in CHECKS), "PASS conflicts with checks")
            for key in CHECKS:
                response[key] = True
        elif status == "UNCERTAIN":
            reason = response.get("reason")
            if reason is None or (isinstance(reason, str) and not reason.strip()):
                reason = next((response[key] for key in ("focus", "check")
                               if isinstance(response.get(key), str) and response[key].strip()), None)
                if reason is not None:
                    response["reason"] = reason
            require(isinstance(reason, str) and reason.strip(), "UNCERTAIN needs a support reason")
            require(response.get("repair_required", False) is False and
                    not any(response.get(k) for k in EDIT_FIELDS), "UNCERTAIN cannot propose repairs")
            require(all(response.get(k) is None for k in CHECKS), "UNCERTAIN conflicts with checks")
            for key in CHECKS:
                response.setdefault(key, None)
            response["repair_required"] = False
            if "issues" not in response and "issue" not in response:
                response["issues"] = [dict(severity="uncertain", target_ids=["answer"], reason=reason.strip())]
        else:
            reason = response.get("reason")
            mismatch = response.get("mismatch")
            if isinstance(mismatch, str) and mismatch.strip():
                if reason is None or (isinstance(reason, str) and not reason.strip()):
                    response["reason"] = reason = mismatch
                elif isinstance(reason, str):
                    require(reason.strip() == mismatch.strip(), "Conflicting reason and mismatch")
            require(isinstance(reason, str) and reason.strip(), "REPAIR needs a material defect reason")
            require(response.get("repair_required", True) is True, "REPAIR conflicts with repair_required")
            for key in CHECKS:
                response.setdefault(key, False if key == "answer_supported" else None)
            response["repair_required"] = True
            if "issues" not in response and "issue" not in response:
                targets = [t for key in ("invalid_task_ids", "invalid_node_ids", "invalid_alignment_ids")
                           for t in response.get(key, [])] or ["answer"]
                response["issues"] = [dict(severity="clear", target_ids=targets, reason=reason.strip())]
    if "issue" in response:
        single = response.pop("issue")
        require(isinstance(single, dict), "issue must be an object")
        if "issues" not in response or response["issues"] == []:
            response["issues"] = [single]
        else:
            require(response["issues"] == [single], "Conflicting issue and issues fields")
    elif isinstance(response.get("issues"), dict):
        response["issues"] = [response["issues"]]
    return response
