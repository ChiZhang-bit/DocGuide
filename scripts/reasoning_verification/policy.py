"""Start repair for clear actionable verification failures."""
from __future__ import annotations

import copy
from utils.common import require
from scripts.reasoning_verification.response import CHECKS, EDIT_FIELDS

def apply_repair_policy(report):
    """Require an actionable defect to start repair."""
    plan = report.get("repair_plan", {})
    requested = report.get("repair_required", False)
    require(type(requested) is bool, "repair_required must be a boolean")
    clear = [i for i in report.get("semantic_issues", report["issues"]) if i.get("severity") == "clear"]
    actionable = (any(plan.get(k) for k in EDIT_FIELDS) or report.get("protocol_status") == "REPAIR" or
                  any(i.get("kind") == "answer_error" for i in clear))
    repair = (requested and bool(clear) and actionable and
              any(report.get("checks", {}).get(k) is False for k in CHECKS))
    retained = {k: copy.deepcopy(plan.get(k, [])) if repair else [] for k in EDIT_FIELDS}
    if retained != plan:
        report["proposed_repair_plan"] = copy.deepcopy(plan)
    report.update(repair_requested=requested, repair_required=bool(repair), repair_plan=retained,
        repair_policy="clear_material_issue_only_v2",
        repair_decision="clear_actionable_issue" if repair else "no_clear_actionable_issue")
