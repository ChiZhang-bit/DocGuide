"""Run DocGuide on the selected local questions and optionally evaluate accuracy."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import uuid

from utils.common import Config, append_jsonl, require, write_json
from utils.paths import ROOT
from utils.data.loader import dataset_sha256, load_dataset, select_indices, inference_record
from utils.llm.client import LoggedLLM
from utils.llm.config import configured_base_url, credential, get_model_name, get_request_timeout
from scripts.pipeline import Pipeline


DATASETS = {
    "hybridqa": ROOT / "dataset" / "HybridQA",
    "ottqa": ROOT / "dataset" / "OTT-QA",
    "mmqa": ROOT / "dataset" / "MultimodalQA",
}


def resolve_device(requested):
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def run_question(index, sample, config, output, timeout):
    llm = LoggedLLM(config.model, timeout=timeout, api_mode=config.api_mode,
                    call_sink=lambda call: append_jsonl(output / "samples" / f"{index}_calls.jsonl", call))
    pipeline = Pipeline(config, llm,
        progress=lambda message: print(f"[{index}] {message}", flush=True),
        trace_sink=lambda trace: append_jsonl(output / "samples" / f"{index}_trace.jsonl", trace))
    try:
        result = pipeline.run(inference_record(sample))
    except Exception as exc:
        result = dict(question_id=sample["question_id"], question=sample["question"],
            prediction=dict(answer="UNKNOWN", status="answer_failed", error=str(exc), verification_performed=False),
            round_traces=[], llm_calls=llm.calls, configuration_snapshot=asdict(config))
    result.update(sample_index=index, gold_answer=sample.get("answer", ""))
    write_json(output / "samples" / f"{index}.json", result)
    print(json.dumps(dict(index=index, answer=result["prediction"]["answer"],
                          status=result["prediction"]["status"]), ensure_ascii=False), flush=True)
    return result


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    data = command.add_mutually_exclusive_group()
    data.add_argument("--dataset", choices=DATASETS, help="Use a dataset supplied in this checkout")
    data.add_argument("--data", type=Path, help="Use a compatible JSON/JSONL file or directory of dev-*.jsonl shards")
    command.add_argument("--model", default=get_model_name())
    command.add_argument("--api-mode", choices=("chat", "responses"), default=Config.api_mode,
                         help="Default: Chat Completions; Responses requires an explicit selection")
    command.add_argument("--indices", type=int, nargs="+", help="Zero-based indices in the selected dataset")
    command.add_argument("--start-index", type=int, default=0)
    command.add_argument("--end-index", type=int, help="Inclusive; default is the last question")
    command.add_argument("--output-dir", type=Path)
    command.add_argument("--phase", choices=("all", "qa", "eval"), default="all")
    command.add_argument("--workers", type=int, default=4)
    command.add_argument("--judge-workers", type=int, default=4)
    command.add_argument("--timeout", type=float, default=get_request_timeout())
    command.add_argument("--judge-timeout", type=float, default=get_request_timeout())
    command.add_argument("--max-rounds", type=int, default=Config.max_rounds)
    command.add_argument("--fuse-final-answer", action=argparse.BooleanOptionalAction, default=Config.fuse_final_answer)
    command.add_argument("--verification-mode", choices=("off", "local", "llm"), default=Config.verification_mode)
    command.add_argument("--verification-tokens", type=int, default=Config.verification_tokens,
                         help="Maximum verification response length")
    command.add_argument("--max-repair-rounds", type=int, choices=(0, 1), default=Config.max_repair_rounds)
    command.add_argument("--repair-execution-rounds", type=int, default=Config.repair_execution_rounds)
    command.add_argument("--retrieval-method", choices=("hybrid", "embedding", "bm25"), default=Config.retrieval_method)
    command.add_argument("--table-top-k", type=int, default=Config.table_top_k)
    command.add_argument("--text-top-k", type=int, default=Config.text_top_k)
    command.add_argument("--text-candidate-k", type=int, default=Config.text_candidate_k)
    command.add_argument("--embedding-model-path", default=Config.embedding_model_path)
    command.add_argument("--embedding-device", default="auto")
    command.add_argument("--dual-route-selection", action=argparse.BooleanOptionalAction, default=Config.dual_route_selection)
    command.add_argument("--reranker-model-path", default=Config.reranker_model_path)
    command.add_argument("--reranker-device", default=Config.reranker_device)
    command.add_argument("--reranker-threshold", type=float, default=Config.reranker_threshold)
    command.add_argument("--reranker-max-length", type=int, default=Config.reranker_max_length)
    command.add_argument("--reranker-batch-size", type=int, default=Config.reranker_batch_size)
    command.add_argument("--row-template-cache-dir", default=Config.row_template_cache_dir)
    command.add_argument("--prepare-templates", action=argparse.BooleanOptionalAction, default=True,
                         help="Before QA, prepare missing row templates for the selected questions")
    command.add_argument("--torch-threads", type=int, default=4)
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    args.timeout = get_request_timeout(args.timeout)
    args.judge_timeout = get_request_timeout(args.judge_timeout)
    supplied_data = args.data or os.getenv("DOCQA_DATA_PATH")
    if not args.dataset and not supplied_data:
        command.error("Choose --dataset or --data (or set DOCQA_DATA_PATH)")
    data = (DATASETS[args.dataset] if args.dataset else Path(supplied_data)).expanduser().resolve()
    require(data.is_file() or data.is_dir(), f"Dataset not found: {data}")
    require(min(args.workers, args.judge_workers, args.timeout, args.judge_timeout, args.torch_threads) > 0,
            "Workers, threads and request timeouts must be positive")
    samples = load_dataset(data)
    indices = select_indices(samples, args.indices, args.start_index, args.end_index)
    label = args.dataset or data.stem
    output = (args.output_dir or ROOT / "outputs" / label / args.model.replace("/", "_") / uuid.uuid4().hex[:8]).resolve()
    judge_model = get_model_name(judge=True) or args.model
    require(output.is_relative_to(ROOT), "Choose an output directory inside this checkout")
    if args.phase == "eval":
        require((output / "results.json").is_file(), "Evaluation requires an existing results.json")
        explicit_selection = args.indices is not None or args.start_index != 0 or args.end_index is not None
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and (not explicit_selection or not judge_model):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            judge_model = judge_model or manifest.get("configuration", {}).get("model", "")
            if not explicit_selection:
                require(manifest.get("dataset_sha256") == dataset_sha256(data),
                        "Dataset differs from the saved QA run")
                indices = select_indices(samples, manifest["indices"])
    else:
        require(not output.exists(), "Choose a new output directory for each QA run")
        require(args.model, "Set model in utils/llm/config.py or pass --model")
        config = Config(model=args.model, api_mode=args.api_mode, max_rounds=args.max_rounds,
            fuse_final_answer=args.fuse_final_answer, verification_mode=args.verification_mode,
            verification_tokens=args.verification_tokens, max_repair_rounds=args.max_repair_rounds,
            repair_execution_rounds=args.repair_execution_rounds, retrieval_method=args.retrieval_method,
            table_top_k=args.table_top_k, text_top_k=args.text_top_k, text_candidate_k=args.text_candidate_k,
            embedding_model_path=args.embedding_model_path,
            embedding_device=resolve_device(args.embedding_device) if args.retrieval_method != "bm25" else "cpu",
            dual_route_selection=args.dual_route_selection, reranker_model_path=args.reranker_model_path,
            reranker_device=resolve_device(args.reranker_device) if args.dual_route_selection else "cpu",
            reranker_threshold=args.reranker_threshold, reranker_max_length=args.reranker_max_length,
            reranker_batch_size=args.reranker_batch_size, row_template_cache_dir=str(Path(args.row_template_cache_dir).expanduser().resolve()))
        for needed, model_path in ((config.retrieval_method != "bm25", config.embedding_model_path),
                                   (config.dual_route_selection, config.reranker_model_path)):
            if needed:
                require(model_path and Path(model_path).expanduser().is_dir(), "Configure the required local model directory")
        LoggedLLM(config.model, timeout=args.timeout, api_mode=config.api_mode)
        if args.phase == "all":
            require(judge_model, "Set model or judge_model in utils/llm/config.py")
            configured_base_url(judge=True)
            credential(judge=True)
        if args.prepare_templates and (config.retrieval_method != "bm25" or config.dual_route_selection):
            from scripts.prepare_row_templates import main as prepare
            prepare(["--data", str(data), "--model", args.model, "--cache-dir", config.row_template_cache_dir,
                     "--timeout", str(args.timeout), "--api-mode", config.api_mode,
                     "--indices", *map(str, indices)])
        elif config.retrieval_method != "bm25" or config.dual_route_selection:
            from utils.data.sources import Sources
            from scripts.decomposition_retrieval.row_templates import RowTemplateStore
            for index in indices:
                source = Sources(inference_record(samples[index]), config.sentences_per_chunk)
                RowTemplateStore(source.headers, config.model, config.row_template_cache_dir).require_prepared()
        if config.retrieval_method != "bm25" or config.dual_route_selection:
            import torch
            torch.set_num_threads(args.torch_threads)
        if config.retrieval_method != "bm25":
            from utils.model.encoder_pool import get_encoder
            get_encoder(config, kind="startup").encode(["Warmup."], batch_size=1)
        output.mkdir(parents=True)
        manifest = dict(dataset=str(data), dataset_sha256=dataset_sha256(data),
                        indices=indices, workers=args.workers, configuration=asdict(config))
        write_json(output / "manifest.json", manifest)
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(run_question, i, samples[i], config, output, args.timeout) for i in indices]
            for job in as_completed(jobs):
                result = job.result()
                results.append(result)
                append_jsonl(output / "results.jsonl", result)
                write_json(output / "summary.json", dict(selected=len(indices), completed=len(results),
                    status_counts=dict(Counter(r["prediction"]["status"] for r in results))))
        write_json(output / "results.json", dict(results=sorted(results, key=lambda r: r["sample_index"])))
    if args.phase in ("all", "eval"):
        from utils.evaluation import main as evaluate
        evaluation_args = ["--input", str(output / "results.json"), "--output", str(output / "accuracy.json"),
                           "--data", str(data), "--indices", *map(str, indices),
                           "--workers", str(args.judge_workers), "--timeout", str(args.judge_timeout)]
        if judge_model:
            evaluation_args.extend(["--model", judge_model])
        status = evaluate(evaluation_args)
        if status:
            return status
    print(f"Results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
