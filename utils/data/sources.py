"""Immutable table/text records with stable IDs and positional duplicate headers."""
from __future__ import annotations
from collections import Counter
import copy
import re
from ..common import require, stable_id
from .references import resolve_reference
from .loader import inference_record

class Sources:
    """Inference input only; gold answers are never retained."""
    def __init__(self, sample, sentences_per_chunk=3):
        sample = inference_record(sample)
        original = [str(name) for name in sample["table"]["header"]]
        counts = Counter(original)
        self.column_mapping = []
        if len(counts) != len(original):
            reserved = set(original)
            names = []
            for index, name in enumerate(original):
                normalized = name
                if counts[name] > 1:
                    normalized = f"{name} [column {index + 1}]"
                    suffix = 1
                    while normalized in reserved:
                        normalized = f"{name} [column {index + 1}, alias {suffix}]"
                        suffix += 1
                reserved.add(normalized)
                names.append(normalized)
                self.column_mapping.append(dict(column_index=index, original_header=name,
                                                column=normalized))
            sample = dict(sample, table=copy.deepcopy(sample["table"]))
            sample["table"]["header"] = names
        self.question = str(sample["question"])
        self.question_id = str(sample["question_id"])
        self.table = copy.deepcopy(sample["table"])
        self.text = list(sample.get("text", []))
        self.version = stable_id("src_", [self.question_id, self.table, self.text])
        self.headers = [str(x) for x in self.table["header"]]
        require(len(set(self.headers)) == len(self.headers), "Duplicate columns require explicit disambiguation")
        self.records = {}
        self.rows, self.chunks = [], []
        for i, values in enumerate(self.table["data"]):
            require(len(values) == len(self.headers), f"Row {i} has incorrect column count")
            record = dict(id=stable_id("T_", [self.version, "table", i]), type="table_record",
                          source_version=self.version, table_id="table", row_id=f"r{i+1}", row_index=i,
                          fields=dict(zip(self.headers, copy.deepcopy(values))))
            self.rows.append(record)
            self.records[record["id"]] = record
        for p, raw in enumerate(self.text):
            spans = []
            start = 0
            for m in re.finditer(r"[.!?。！？]+(?=\s|$)|[。！？]+", raw):
                token = re.search(r"\b([A-Za-z]+)\.$", raw[:m.end()])
                if m.group() == "." and token and (len(token[1]) == 1 or token[1].lower() in
                                                  {"mr", "mrs", "ms", "dr", "prof", "jr", "sr", "st", "vs"}):
                    continue
                if raw[start:m.end()].strip():
                    spans.append((start, m.end()))
                start = m.end()
            if raw[start:].strip():
                spans.append((start, len(raw)))
            n = sentences_per_chunk or max(1, len(spans))
            for c in range(0, len(spans), n):
                selected = spans[c:c+n]
                a, b = selected[0][0], selected[-1][1]
                rid = stable_id("P_", [self.version, p, a, b])
                record = dict(id=rid, type="text_record", source_version=self.version,
                              document_id=f"p{p+1}", paragraph_index=p, chunk_index=c//n,
                              canonical_span=[a, b], raw_text=raw[a:b],
                              sentences=[dict(id=f"p{p+1}.s{c+j}", text=raw[x:y], original_span=[x,y])
                                         for j,(x,y) in enumerate(selected)])
                self.chunks.append(record)
                self.records[rid] = record
        if self.column_mapping:
            for record in self.rows:
                record["column_mapping"] = copy.deepcopy(self.column_mapping)

    def preview(self, text_char_limit=2000):
        text_index, remaining = [], text_char_limit
        for i, text in enumerate(self.text):
            if remaining <= 0:
                break
            preview = text[:min(110, remaining)]
            text_index.append(dict(document_id=f"p{i+1}", preview=preview))
            remaining -= len(preview)
        result = dict(table_id="table", headers=self.headers, row_count=len(self.rows),
                    sample_rows=[dict(row_id=r["row_id"], fields=r["fields"]) for r in self.rows[:5]],
                    text_index=text_index, total_text_documents=len(self.text),
                    omitted_preview_documents=len(self.text)-len(text_index))
        if self.column_mapping:
            result["column_mapping"] = copy.deepcopy(self.column_mapping)
        return result

    def resolve(self, ref: dict, records: dict | None = None) -> dict:
        return resolve_reference(ref, self.records if records is None else records)

