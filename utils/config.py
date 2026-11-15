"""Method configuration."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from .paths import ROOT
from .contracts import require
from .llm.config import get_model_name

@dataclass
class Config:
    max_rounds: int = 3
    table_top_k: int = 3
    text_top_k: int = 5
    text_candidate_k: int = 20
    text_rrf_k: int = 60
    table_rrf_k: int = 60
    row_template_cache_dir: str = os.getenv("DOCQA_ROW_TEMPLATE_CACHE_DIR", str(ROOT / ".cache" / "row_templates"))
    sentences_per_chunk: int = 3
    retrieval_method: str = "hybrid"
    embedding_model_path: str = os.getenv("DOCQA_EMBEDDING_MODEL_PATH", str(ROOT / "model" / "embedding"))
    embedding_device: str = "cpu"
    embedding_max_length: int = 512
    embedding_batch_size: int = 16
    model: str = get_model_name()
    api_mode: str = os.getenv("DOCQA_LLM_API_MODE", "chat")
    router_tokens: int = 1800
    integrate_tokens: int = 4500
    answer_tokens: int = 800
    fuse_final_answer: bool = True
    preview_text_chars: int = 2000
    dual_route_selection: bool = True
    reranker_model_path: str = os.getenv("DOCQA_RERANKER_MODEL_PATH", str(ROOT / "model" / "reranker"))
    reranker_device: str = "auto"
    reranker_max_length: int = 512
    reranker_batch_size: int = 8
    reranker_threshold: float | None = 0.7
    verification_mode: str = "llm"
    verification_tokens: int = 700
    max_repair_rounds: int = 1
    repair_execution_rounds: int = 2
    repair_integrate_tokens: int = 1600
    repair_answer_tokens: int = 500

    def __post_init__(self):
        require(self.api_mode in {"chat", "responses"}, "api_mode must be chat or responses")
        require(self.max_rounds > 0, "max_rounds must be positive")
        require(self.table_top_k > 0 and self.text_top_k > 0, "top-k must be positive")
        require(type(self.table_rrf_k) is int and self.table_rrf_k > 0, "Invalid table RRF constant")
        require(type(self.dual_route_selection) is bool, "dual_route_selection must be boolean")
        require(type(self.fuse_final_answer) is bool, "fuse_final_answer must be boolean")
        require(self.verification_mode in {"off", "local", "llm"}, "verification_mode must be off, local or llm")
        require(type(self.verification_tokens) is int and self.verification_tokens > 0, "Invalid verification_tokens")
        require(type(self.repair_integrate_tokens) is int and self.repair_integrate_tokens > 0, "Invalid repair_integrate_tokens")
        require(type(self.repair_answer_tokens) is int and self.repair_answer_tokens > 0, "Invalid repair_answer_tokens")
        require(type(self.max_repair_rounds) is int and self.max_repair_rounds in (0, 1), "max_repair_rounds must be 0 or 1 (single verification)")
        require(type(self.repair_execution_rounds) is int and self.repair_execution_rounds > 0,
                "repair_execution_rounds must be positive")
        if self.reranker_threshold is not None:
            require(type(self.reranker_threshold) in (int, float) and math.isfinite(self.reranker_threshold)
                    and 0 < self.reranker_threshold < 1, "reranker_threshold must be finite and between 0 and 1")
        if self.dual_route_selection:
            require(self.reranker_threshold is not None, "Explicit reranker_threshold required for dual-route selection")
        require(type(self.reranker_max_length) is int and self.reranker_max_length > 0, "Invalid reranker_max_length")
        require(type(self.reranker_batch_size) is int and self.reranker_batch_size > 0, "Invalid reranker_batch_size")
