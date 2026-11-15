from __future__ import annotations

import copy
from dataclasses import asdict

from scripts.decomposition_retrieval.access import AccessExecutor
from scripts.reasoning_verification.answer import answer_from_view, prepare_fused_answer
from utils.data.citation_normalization import normalize_candidate_references
from utils.common import require, stable_id
from utils.data.sources import Sources
from scripts.evidence_graph.graph import WorkingGraph
from scripts.reasoning_verification.verifier import run_verification
from scripts.reasoning_verification.repair import apply_repair, invalidate_history, repair_feedback
from scripts.decomposition_retrieval.task_dag import TaskDAG
from scripts.decomposition_retrieval.row_templates import RowTemplateStore
from scripts.decomposition_retrieval.router import generate_plan
from scripts.decomposition_retrieval.integrate import answer_subquestions

class Pipeline:
    def __init__(self, config, llm, progress=None, trace_sink=None, selector=None, verifier=None):
        self.config, self.llm = config, llm
        self.progress, self.trace_sink = progress or (lambda s:None), trace_sink or (lambda t:None)
        self.selector = selector
        self.verifier = verifier

    def run(self, sample):
        sources = Sources(sample, self.config.sentences_per_chunk)
        graph = WorkingGraph(sources, preserve_dependencies=True)
        access = AccessExecutor(sources, self.config)
        history, traces, feedback, candidate = [], [], [], ""
        fused_prediction = None
        call_start = len(self.llm.calls)
        dag = TaskDAG()
        templates = RowTemplateStore(sources.headers, self.config.model, self.config.row_template_cache_dir)
        sources.row_template_store = templates
        if sources.rows and (self.config.retrieval_method != "bm25" or
                             self.config.dual_route_selection and self.selector is None):
            templates.require_prepared()
        termination = "max_rounds"
        def fuse_answer(response, eligible_ids, aliases, origin, skipped_alignment_ids=()):
            """Validate candidate support in an isolated graph transaction."""
            trial = copy.copy(graph)
            trial.nodes = dict(graph.nodes)
            trial.events = list(graph.events)
            trial.visible = {nid for nid in eligible_ids if graph.usable(nid)}
            trial.pinned = set(graph.pinned)
            trial.grades = dict(graph.grades)
            view = trial.view()
            prediction = prepare_fused_answer(response, view, aliases, skipped_alignment_ids)
            prediction = normalize_candidate_references(prediction, view)
            prediction["candidate_node_id"] = trial.add_candidate(prediction)
            prediction["answer_source"] = origin
            trial.visible = set(graph.visible) | {prediction["candidate_node_id"]}
            trial.maintain_view()
            graph.nodes, graph.events = trial.nodes, trial.events
            graph.visible, graph.pinned, graph.grades = trial.visible, trial.pinned, trial.grades
            return prediction

        def execute_round(round_id, forced_plan=None, repair_notes=None, *, round_budget_exhausted=False):
            nonlocal candidate, feedback, termination, fused_prediction
            fused_prediction = None
            trace = dict(round=round_id, access_results=[], errors=[],
                         round_budget_exhausted=round_budget_exhausted)
            if repair_notes is not None:
                trace["phase"] = "repair"
            window = set(graph.visible)
            questions = []
            stop = False
            planner = "verifier repair plan" if forced_plan is not None else "router"
            self.progress(f"round {round_id}: {planner}; visible={len(window)}")
            try:
                dag.sync(graph)
                for plan_attempt in range(1 if forced_plan is not None else 2):
                    plan = forced_plan if forced_plan is not None else generate_plan(
                        self.llm, self.config, sources, graph, history, feedback, round_id,
                        access.seen_documents, repair_notes)
                    trace["router"] = plan
                    trace["plan_source"] = "verifier" if forced_plan is not None else "router"
                    try:
                        require(isinstance(plan, dict) and plan.get("action") in {"retrieve", "answer"}, "Invalid router action")
                        transaction = dag.stage(plan.get("subquestions", []), round_id) if plan["action"] == "retrieve" else None
                        break
                    except ValueError as exc:
                        trace.setdefault("plan_validation_failures", []).append(dict(plan=copy.deepcopy(plan), error=str(exc)))
                        if forced_plan is not None or plan_attempt == 1:
                            raise
                        feedback = ["Correct this task plan before execution: " + str(exc)]
                if plan["action"] == "answer":
                    candidate = plan.get("candidate_answer") or candidate
                    if self.config.fuse_final_answer and "final_answer" in plan:
                        try:
                            fused_prediction = fuse_answer(plan["final_answer"], window, {}, "router_fused")
                            trace["fused_answer"] = copy.deepcopy(fused_prediction)
                        except (ValueError, KeyError, TypeError) as exc:
                            trace["fused_answer_error"] = str(exc)
                    termination, stop = "router_answer", True
                else:
                    questions = dag.commit(transaction)
                    summaries, touched, added, receipts = [], set(), set(), []
                    for question in questions:
                        for i,request in enumerate(question["retrievals"], 1):
                            kind = request.get("kind")
                            require(kind in {"table_query", "table_retrieve", "text_retrieve", "text_read"}, "Unknown retrieval kind")
                            task = dict(request, id=f"{question['id']}s{i}", kind=kind, question=question["question"],
                                        query_spec=request.get("query_spec", {}))
                            result = access.execute(task)
                            delta = graph.upsert(result, question["id"])
                            touched.update(delta["touched"])
                            added.update(delta["added"])
                            window.update(delta["touched"])
                            receipts.append(result["receipt"]["id"])
                            trace["access_results"].append(dict(subquestion_id=question["id"], **result))
                            summaries.append(dict(subquestion_id=question["id"], requested_kind=kind,
                                requested_query=task["query_spec"], executed_query=result["receipt"]["executed_query"],
                                returned_count=len(result["records"]),
                                new_node_count=len(delta["added"]), receipt=result["receipt"]))
                    self.progress(f"round {round_id}: integrate; retrieved={len(touched)}, new={len(added)}")
                    prune = any(q["retrievals"] for q in questions)
                    relevance_candidates = sorted(graph.pruning_candidates(window)) if prune else []
                    response = answer_subquestions(self.llm, self.config,
                        question=sources.question, subquestions=questions, working_view=graph.view(window),
                        touched=touched, added=added, pinned_ids=graph.pinned, retrieval_results=summaries,
                        prune=prune, relevance_candidates=relevance_candidates, repair_notes=repair_notes,
                        round_budget_exhausted=round_budget_exhausted)
                    trace["integration_response"] = response
                    graph_response = response if self.config.fuse_final_answer or not isinstance(response, dict) else {
                        key: value for key, value in response.items() if key != "final_answer"}
                    integrated = graph.integrate(graph_response, questions, window, receipts,
                                                 prune=prune, task_outputs=dag.outputs())
                    trace["integration"] = integrated
                    dag.record_outcomes(integrated["subquestions"], graph)
                    if self.config.fuse_final_answer and (integrated["can_answer_original"] or round_budget_exhausted) and "final_answer" in response:
                        try:
                            eligible = window | set(integrated["aliases"].values())
                            fused_prediction = fuse_answer(response["final_answer"], eligible,
                                integrated["aliases"], "integrate_fused",
                                integrated["skipped_alignment_ids"])
                            trace["fused_answer"] = copy.deepcopy(fused_prediction)
                        except (ValueError, KeyError, TypeError) as exc:
                            trace["fused_answer_error"] = str(exc)
                    candidates = set(integrated["pruning_candidate_ids"])
                    protected = graph.protected_ids() | graph.pinned
                    candidates.difference_update(protected)
                    if self.config.dual_route_selection and prune and candidates:
                        if self.selector is None:
                            from scripts.view_maintenance.ensemble_selection import DualRouteSelector
                            self.selector = DualRouteSelector(self.config, row_formatter=templates.render)
                        selection = self.selector.select(sources.question, questions,
                            {nid: graph.nodes[nid] for nid in candidates},
                            {nid: integrated["llm_grades"][nid] for nid in candidates}, graph.pinned)
                        graph.grades.update(selection["grades"])
                        graph.visible = set(selection["selected_ids"]) | protected
                        trace["selection"] = selection
                    trace["dependency_retained_ids"] = graph.maintain_view()
                    candidate = integrated.get("candidate_answer") or candidate
                    history.append(dict(round=round_id, subquestions=integrated["subquestions"],
                        retrievals=[{k:v for k,v in s.items() if k != "receipt"} for s in summaries],
                        can_answer_original=integrated["can_answer_original"]))
                    feedback = integrated["errors"]
                    if integrated["can_answer_original"]:
                        termination, stop = "integrate_answer", True
            except Exception as exc:
                feedback = [str(exc)]
                trace["errors"].append(str(exc))
                graph.retain_on_error(window)
                dag.record_outcomes([dict(id=q["id"], status="unanswered", answer_fact_id=None)
                    for q in questions if dag.tasks[q["id"]]["status"] == "planned"], graph)
                history.append(dict(round=round_id, subquestions=[dict(id=q["id"], question=q["question"],
                    status="unanswered", missing="Round error: "+str(exc)) for q in questions]))
            trace["working_view"] = graph.view()
            trace["task_dag"] = dag.snapshot()
            trace["pinned_ids"] = sorted(graph.pinned)
            trace["feedback"] = copy.deepcopy(feedback)
            if stop:
                trace["termination_reason"] = termination
            traces.append(trace)
            self.trace_sink(trace)
            return stop

        for round_id in range(1, self.config.max_rounds+1):
            if execute_round(round_id, round_budget_exhausted=round_id == self.config.max_rounds):
                break
        normal_termination_reason = termination
        def answer(repair_notes=None):
            nonlocal fused_prediction
            self.progress("reuse grounded in-call answer" if fused_prediction is not None else "answer from current working view")
            graph.maintain_view()
            view = graph.view()
            if fused_prediction is not None:
                prediction, fused_prediction = fused_prediction, None
            else:
                prediction = answer_from_view(self.llm, sources.question, view,
                    min(self.config.answer_tokens, self.config.repair_answer_tokens)
                        if repair_notes is not None else self.config.answer_tokens,
                    candidate, repair_feedback=repair_notes, max_attempts=1 if repair_notes is not None else 2)
                prediction["answer_source"] = "answer"
            if self.config.verification_mode == "llm":
                prediction = normalize_candidate_references(prediction, view)
                try:
                    prediction["candidate_node_id"] = graph.add_candidate(prediction)
                except (ValueError, KeyError, TypeError) as exc:
                    prediction["candidate_provenance_error"] = str(exc)
                view = graph.view()
            return prediction, view

        prediction, final_view = answer()
        self.progress(f"verification: {self.config.verification_mode}")
        verification = run_verification(self.config.verification_mode, question=sources.question,
            prediction=prediction, working_view=final_view, verifier=self.verifier,
            llm=self.llm, max_tokens=self.config.verification_tokens, task_dag=dag.snapshot())
        verification["checked_view_id"] = stable_id("W_", final_view)
        verification["candidate_node_id"] = prediction.get("candidate_node_id")
        verification_history = [dict(candidate=copy.deepcopy(prediction), report=copy.deepcopy(verification))]
        repairs = []
        if self.config.verification_mode == "llm" and verification.get("repair_required") is True and not verification["verified"] and verification["structural_status"] != "error":
            may_repair = self.config.max_repair_rounds > 0
            try:
                repair = apply_repair(graph, verification, prediction, allow_corrections=may_repair,
                                      task_dag=dag, next_round=round_id + 1)
            except (ValueError, KeyError, TypeError) as exc:
                repairs.append(dict(rejected=True, errors=[str(exc)], invalidated_node_ids=[],
                                    invalidated_task_ids=[], subquestions=[]))
                prediction["status"] = "verification_failed"
                termination = "repair_plan_rejected"
            else:
                invalidate_history(history, repair["invalidated_node_ids"], repair.get("invalidated_task_ids", []))
                repairs.append(repair)
                candidate = ""
                fused_prediction = None
                notes = repair_feedback(verification) + repair["errors"]
                feedback = notes
                if may_repair:
                    self.progress("resume affected branch; no second verification call")
                    repair_complete = True
                    if repair["subquestions"] and not repair["errors"]:
                        repair_complete = execute_round(round_id + 1,
                            dict(action="retrieve", subquestions=repair["subquestions"]), notes,
                            round_budget_exhausted=self.config.repair_execution_rounds == 1)
                        for offset in range(2, self.config.repair_execution_rounds + 1):
                            if repair_complete:
                                break
                            repair_complete = execute_round(round_id + offset, repair_notes=notes,
                                round_budget_exhausted=offset == self.config.repair_execution_rounds)
                    prediction, final_view = answer(notes)
                    if prediction["status"] != "answer_failed":
                        prediction["status"] = "repaired_unverified"
                    repair["execution_complete"] = repair_complete
                    termination = "single_repair_complete" if repair_complete else "repair_budget_exhausted"
                    verification = dict(enabled=True, performed=False, backend="llm_path_v2",
                        structural_status="not_checked", semantic_status="not_checked", verified=False,
                        issues=[], reason="Repaired candidate was not reverified: single verification/repair budget",
                        candidate_node_id=prediction.get("candidate_node_id"))
                else:
                    prediction["status"] = "verification_failed"
                    final_view = graph.view()
                    termination = "repair_disabled"
        elif verification["verified"]:
            prediction["status"] = "verified"
            termination = "verification_passed"
        elif self.config.verification_mode == "llm":
            termination = "verification_error" if verification["structural_status"] == "error" else "verification_no_repair"
        prediction["termination_reason"] = termination
        prediction["verification_performed"] = verification["performed"]
        prediction["verified"] = verification["verified"]
        prediction["verification_scope"] = verification["backend"]
        dag.sync(graph)
        result = dict(question_id=sources.question_id, question=sources.question, source_version=sources.version,
            normal_termination_reason=normal_termination_reason,
            prediction=prediction, final_answer=dict(answer=prediction["answer"], status=prediction["status"],
                is_evidence_enough=prediction["evidence_sufficient"], verification_performed=verification["performed"],
                verified=verification["verified"], verification_scope=verification["backend"]),
            verification=verification, verification_history=verification_history, repairs=repairs,
            task_dag=dag.snapshot(), row_template=templates.diagnostics(),
            final_evidence_graph=graph.snapshot(), final_working_view=final_view,
            final_grades=graph.grades, original_relevant_ids=sorted(graph.pinned),
            subquestion_history=history, round_traces=traces, configuration_snapshot=asdict(self.config),
            llm_calls=self.llm.calls[call_start:])
        return result
