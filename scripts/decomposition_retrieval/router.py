"""Generate one dynamic subquestion plan from the current evidence view."""
from utils.prompt_loader import load_prompt


def generate_plan(llm, config, sources, graph, history, feedback, round_id,
                  seen_documents, repair_notes=None):
    instructions = load_prompt("router_fused" if config.fuse_final_answer else "router")
    return llm.ask("router", instructions, dict(question=sources.question,
        preview=sources.preview(config.preview_text_chars),
        subquestion_history=history, working_view=graph.view(), feedback=feedback,
        round=round_id, max_rounds=config.max_rounds +
            (config.repair_execution_rounds if repair_notes is not None else 0),
        expandable_document_ids=sorted(seen_documents),
        retrieval_defaults=dict(table_top_k=config.table_top_k, text_top_k=config.text_top_k)),
        max_tokens=config.router_tokens)
