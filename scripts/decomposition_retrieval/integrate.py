"""One response supplies subquestion answers, alignments and logical relevance."""
from utils.prompt_loader import load_prompt


def answer_subquestions(llm, config, *, question, subquestions, working_view,
                        touched, added, pinned_ids, retrieval_results, prune,
                        relevance_candidates, repair_notes=None, round_budget_exhausted=False):
    instructions = load_prompt("integrate_fused" if config.fuse_final_answer else "integrate")
    if repair_notes is not None:
        instructions += load_prompt("repair_suffix")
    return llm.ask("integrate", instructions, dict(question=question,
        subquestions=[{k:v for k,v in q.items() if k != "retrievals"} for q in subquestions],
        working_view=working_view, retrieved_node_ids=sorted(touched), new_node_ids=sorted(added),
        original_relevant_ids=sorted(pinned_ids), retrieval_results=retrieval_results,
        pruning_enabled=prune, relevance_candidate_ids=relevance_candidates,
        **({"round_budget_exhausted": round_budget_exhausted} if config.fuse_final_answer else {}),
        **({"repair_feedback": repair_notes} if repair_notes else {})),
        max_tokens=min(config.integrate_tokens,
            config.repair_integrate_tokens * max(1, len(subquestions)))
            if repair_notes is not None else config.integrate_tokens)
