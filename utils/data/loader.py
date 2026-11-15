"""Read datasets while preserving content and record order."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from ..common import require


def normalize_record(record):
    """Unwrap storage fields; preserve gold only for the evaluator."""
    require(isinstance(record, dict), "Each dataset record must be an object")
    nested = record.get("data")
    nested = nested if isinstance(nested, dict) else {}
    qid = record.get("question_id", record.get("id"))
    require(qid is not None and str(qid).strip(), "Missing question ID")
    question = record.get("question")
    require(isinstance(question, str) and question.strip(), "Missing question")
    table = record.get("table", nested.get("table"))
    text = record.get("text", nested.get("text", []))
    require(isinstance(table, dict) and isinstance(table.get("header"), list)
            and isinstance(table.get("data"), list), "Invalid table")
    require(isinstance(text, list) and all(isinstance(p, str) for p in text), "Invalid text list")
    result = {"question_id": str(qid), "question": question, "table": table, "text": text}
    for key in ("answer", "required_modalities", "answer_source"):
        if key in record:
            result[key] = record[key]
    return result


def inference_record(record):
    """Use an explicit allowlist so reference answers never enter inference."""
    normalized = normalize_record(record)
    return {key: normalized[key] for key in ("question_id", "question", "table", "text")}


def _dataset_files(path):
    path = Path(path)
    if path.is_dir():
        shards = sorted(shard for shard in path.glob("dev-*.jsonl") if shard.is_file())
        require(bool(shards), f"No dev-*.jsonl dataset shards found: {path}")
        return shards
    return [path]


def dataset_sha256(path):
    """Hash raw dataset bytes, concatenating ordered shards without separators."""
    digest = hashlib.sha256()
    for source in _dataset_files(path):
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_dataset(path):
    rows = []
    for source in _dataset_files(path):
        if source.suffix.lower() == ".jsonl":
            with source.open(encoding="utf-8-sig") as handle:
                for number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        rows.append(json.loads(line))
                    except ValueError as exc:
                        raise ValueError(f"Invalid JSON at {source}:{number}") from exc
        else:
            rows = json.loads(source.read_text(encoding="utf-8-sig"))
    require(isinstance(rows, list) and bool(rows), "Dataset must be a nonempty list or JSONL")
    result = [normalize_record(row) for row in rows]
    ids = [row["question_id"] for row in result]
    require(len(set(ids)) == len(ids), "Duplicate dataset question IDs")
    return result


def select_indices(rows, indices=None, start_index=0, end_index=None):
    """Select original zero-based row positions; end_index is inclusive."""
    if indices is not None:
        require(start_index == 0 and end_index is None,
                "Use --indices or --start-index/--end-index, not both")
        selected = list(indices)
    else:
        end = len(rows) - 1 if end_index is None else end_index
        require(type(start_index) is int and type(end) is int and 0 <= start_index <= end < len(rows),
                "Invalid dataset range")
        selected = list(range(start_index, end + 1))
    require(bool(selected) and all(type(i) is int and 0 <= i < len(rows) for i in selected)
            and len(set(selected)) == len(selected), "Invalid or duplicate dataset indices")
    return sorted(selected)
