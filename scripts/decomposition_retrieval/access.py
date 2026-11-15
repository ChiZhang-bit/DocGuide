"""Bounded table/text retrieval and same-round table-query fallback."""
from __future__ import annotations
import copy
from dataclasses import replace
from utils.common import ContractError, canonical, require, stable_id
from utils.model.encoder_pool import get_encoder
from utils.data.records import TextChunk
from .row_representation import table_to_rows, chunk_search_text
from .bm25 import BM25Retriever
from .table_retrieval import TableRetriever
from .text_retrieval import TextRetriever
from .table_query import execute_ast

class AccessExecutor:
    def __init__(self, sources, config):
        self.sources, self.config = sources, config
        self.table_retriever = self.text_retriever = None
        self.counter = 0
        self.seen_documents = set()
        self.last_retrieval = None

    def retrieve(self, kind, query):
        if kind != "table_retrieve":
            return self._retrieve_text(query)
        self.last_retrieval = None
        require(isinstance(query.get("query"), str) and query["query"].strip(), "Missing retrieval query")
        k = query.get("top_k", self.config.table_top_k)
        require(type(k) is int and k > 0, "top_k must be positive")
        if self.table_retriever is None:
            methods = {"hybrid": (1, 1, 0, 0), "embedding": (1, 0, 0, 0), "bm25": (0, 1, 0, 0)}
            require(self.config.retrieval_method in methods, "Unknown retrieval method")
            self.table_retriever = TableRetriever(
                table_to_rows(self.sources.table, row_formatter=(self.sources.row_template_store.render
                    if getattr(self.sources, "row_template_store", None) else None)),
                embedding_model_path=self.config.embedding_model_path,
                embedding_device=self.config.embedding_device,
                embedding_batch_size=self.config.embedding_batch_size,
                embedding_max_length=self.config.embedding_max_length,
                encoder=get_encoder(self.config, "table")
                        if self.config.retrieval_method != "bm25" else None,
                rank_methods=methods[self.config.retrieval_method],
                rank_weights=(0.5, 0.5, 0.0, 0.0), rrf_k=self.config.table_rrf_k)
        hits = self.table_retriever.search(query["query"], top_k=k, original_question=self.sources.question)
        self.last_retrieval = copy.deepcopy(self.table_retriever.last_search)
        for hit in self.last_retrieval["hits"]:
            hit["record_id"] = self.sources.rows[hit["row_index"]]["id"]
        return [self.sources.rows[h.row.row_index] for h in hits]

    def _retrieve_text(self, query):
        self.last_retrieval = None
        require(isinstance(query.get("query"), str) and query["query"].strip(), "Missing retrieval query")
        k = query.get("top_k", self.config.text_top_k)
        require(type(k) is int and k > 0, "top_k must be positive")
        if self.text_retriever is None:
            chunks = [TextChunk(chunk_id=i, paragraph_index=r["paragraph_index"],
                                      paragraph_number=r["paragraph_index"]+1, chunk_index=r["chunk_index"],
                                      chunk_number=r["chunk_index"]+1, sentence_start=0,
                                      sentence_end=len(r["sentences"])-1, text=r["raw_text"])
                      for i,r in enumerate(self.sources.chunks)]
            self.text_retriever = TextRetriever(
                [chunk_search_text(c) for c in chunks],
                method=self.config.retrieval_method,
                encoder=get_encoder(self.config, "text")
                        if self.config.retrieval_method != "bm25" else None,
                batch_size=self.config.embedding_batch_size,
                candidate_k=self.config.text_candidate_k, rrf_k=self.config.text_rrf_k)
        hits = self.text_retriever.search(query["query"], top_k=k)
        self.last_retrieval = copy.deepcopy(self.text_retriever.last_search)
        for hit in self.last_retrieval["hits"]:
            hit["record_id"] = self.sources.chunks[hit["doc_id"]]["id"]
        return [self.sources.chunks[h.doc_id] for h in hits]

    @staticmethod
    def _normalize_retrieval_task(task):
        if task["kind"] not in {"table_retrieve", "text_retrieve"}:
            return task, []
        original = task.get("query_spec")
        spec = copy.deepcopy(original) if isinstance(original, dict) else {}
        repairs = []

        def has_query(value):
            return isinstance(value, str) and bool(value.strip())

        if not has_query(spec.get("query")):
            if has_query(task.get("query")):
                spec["query"] = task["query"]
                repairs.append("query_from_top_level")
            elif has_query(task.get("question")):
                spec["query"] = task["question"]
                repairs.append("query_from_subquestion")
        if "top_k" not in spec and "top_k" in task:
            spec["top_k"] = task["top_k"]
            repairs.append("top_k_from_top_level")
        return dict(task, query_spec=spec), repairs

    def execute(self, task):
        task, repairs = self._normalize_retrieval_task(task)
        result = self._execute_once(task)
        if repairs:
            result["receipt"]["parameter_repairs"] = repairs
        if task["kind"] != "table_query" or result["receipt"]["outcome"] == "success":
            return result
        code_receipt = copy.deepcopy(result["receipt"])
        spec = task.get("query_spec", {})
        spec = spec if isinstance(spec, dict) else {}
        k = spec.get("top_k", self.config.table_top_k)
        k = k if type(k) is int and k > 0 else self.config.table_top_k
        query = str(task.get("question") or self.sources.question)
        if spec.get("filters"):
            query += " Conditions: " + canonical(spec["filters"])
        query_spec = dict(query=query, top_k=k)
        errors = []
        try:
            if self.config.retrieval_method != "hybrid":
                if not hasattr(self, "hybrid_fallback"):
                    self.hybrid_fallback = AccessExecutor(self.sources, replace(self.config, retrieval_method="hybrid"))
                records = self.hybrid_fallback.retrieve("table_retrieve", query_spec)
            else:
                records = self.retrieve("table_retrieve", query_spec)
            backend = "table_hybrid"
        except Exception as exc:
            errors.append(str(exc))
            lexical = BM25Retriever(
                [canonical(r["fields"]) for r in self.sources.rows])
            records = [self.sources.rows[h.doc_id] for h in lexical.search(query, k)]
            backend = "bm25_after_hybrid_error"
        if not records and self.sources.rows:
            records = self.sources.rows[:k]
            backend = "original_order_after_empty_retrieval"
        receipt = dict(id=stable_id("Q_", [code_receipt["id"], "fallback", query_spec]),
                       task_id=task["id"], source_version=self.sources.version,
                       mode="table_retrieve", executed_query=query_spec,
                       coverage="candidate_only", truncated=True,
                       outcome="success" if records else "empty",
                       returned_ids=[r["id"] for r in records], fallback_from="table_query",
                       fallback_backend=backend,
                       code_attempt={k:v for k,v in code_receipt.items() if k not in {"source_version", "id", "task_id"}})
        if errors:
            receipt["fallback_errors"] = errors
        return dict(records=records, receipt=receipt)

    def _execute_once(self, task):
        self.counter += 1
        query = copy.deepcopy(task.get("query_spec", {}))
        receipt = dict(id=stable_id("Q_", [self.sources.version, task["id"], self.counter]),
                       task_id=task["id"], source_version=self.sources.version, executed_query=query,
                       coverage="unknown", truncated=False, outcome="error", mode=task["kind"], returned_ids=[])
        try:
            if task["kind"] == "table_query":
                records, scope = execute_ast(self.sources.rows, self.sources.headers, query, self.config.table_top_k)
                receipt.update(scope)
            elif task["kind"] in {"table_retrieve", "text_retrieve"}:
                records = self.retrieve(task["kind"], query)
                receipt.update(coverage="candidate_only", truncated=True)
            elif task["kind"] == "text_read":
                ids = query.get("document_ids", [])
                require(ids and all(p in {r["document_id"] for r in self.sources.chunks} for p in ids), "Unknown text document")
                require(set(ids) <= self.seen_documents, "text_read expands previously retrieved paragraphs; use text_retrieve first")
                records = [r for r in self.sources.chunks if r["document_id"] in ids]
                receipt.update(coverage="scope_complete", truncated=False)
            else:
                raise ContractError("Not an access task")
            receipt.update(outcome="success" if records else "empty", returned_ids=[r["id"] for r in records])
            self.seen_documents.update(r["document_id"] for r in records if r["type"] == "text_record")
            result = dict(records=records, receipt=receipt)
            if task["kind"] in {"table_retrieve", "text_retrieve"} and self.last_retrieval:
                result["retrieval_diagnostics"] = copy.deepcopy(self.last_retrieval)
            return result
        except (ValueError, KeyError, TypeError, RuntimeError, ImportError, OSError) as exc:
            receipt["error"] = str(exc)
            return dict(records=[], receipt=receipt)

    def replay_receipt(self, receipt):
        require(receipt["source_version"] == self.sources.version, "Source version mismatch")
        if receipt["mode"] == "table_query":
            records, scope = execute_ast(self.sources.rows, self.sources.headers, receipt["executed_query"], self.config.table_top_k)
            require([r["id"] for r in records] == receipt["returned_ids"], "Query replay mismatch")
            require(all(receipt[k] == v for k,v in scope.items()), "Query coverage receipt mismatch")
        return True
