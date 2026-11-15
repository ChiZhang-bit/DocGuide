"""Validated table query AST and exact scope receipts."""
from datetime import date
from decimal import Decimal
import re
from utils.common import ContractError, require

def typed(value, dtype):
    if dtype == "text":
        return " ".join(str(value).split()).casefold()
    if dtype == "number":
        text = str(value).strip().replace(",", "")
        require(bool(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text)), f"Unparseable number {value!r}")
        return Decimal(text)
    if dtype == "date":
        try:
            return date.fromisoformat(str(value))
        except ValueError as exc:
            raise ContractError("Date operations require unambiguous full ISO dates") from exc
    raise ContractError(f"Unsupported datatype {dtype}")

def execute_ast(rows, headers, spec, default_k):
    require(isinstance(spec, dict), "Query AST must be an object")
    require(set(spec) <= {"filters", "order_by", "return_mode", "top_k", "expected_count", "project", "table_id"}, "Unknown query AST field")
    require(spec.get("table_id", "table") == "table", "Unknown table")
    project = spec.get("project", [])
    require(isinstance(project, list), "project must be a list")
    require(all(c in headers for c in project), "Unknown projection column")
    filters = spec.get("filters", [])
    require(isinstance(filters, list), "Filters must be a list")
    for f in filters:
        require(isinstance(f, dict), "Each filter must be an object")
        require(f.get("column") in headers, "Unknown filter column")
        require(isinstance(f.get("operator"), str) and f["operator"] in
                {"eq", "ne", "in", "contains", "gt", "ge", "lt", "le"}, "Unsupported filter operator")
        require(isinstance(f.get("datatype", "text"), str) and
                f.get("datatype", "text") in {"text", "number", "date"}, "Unsupported filter datatype")
    result = []
    for row in rows:
        keep = True
        for f in filters:
            dtype, op = f.get("datatype", "text"), f["operator"]
            left = typed(row["fields"][f["column"]], dtype)
            if op == "in":
                require(isinstance(f.get("values"), list), "in requires values array")
                match = left in [typed(x, dtype) for x in f["values"]]
            else:
                require("value" in f, "Filter missing value")
                right = typed(f["value"], dtype)
                if op == "contains":
                    require(dtype == "text", "contains requires text")
                    match = right in left
                else:
                    match = {"eq": lambda: left == right, "ne": lambda: left != right,
                             "gt": lambda: left > right, "ge": lambda: left >= right,
                             "lt": lambda: left < right, "le": lambda: left <= right}[op]()
            if not match:
                keep = False
                break
        if keep:
            result.append(row)
    order = spec.get("order_by", [])
    require(isinstance(order, list), "order_by must be a list")
    for key in reversed(order):
        require(isinstance(key, dict), "Each ordering must be an object")
        require(key.get("column") in headers, "Unknown sort column")
        require(isinstance(key.get("direction", "asc"), str) and
                key.get("direction", "asc") in {"asc", "desc"}, "Unknown sort direction")
        require(isinstance(key.get("datatype", "text"), str) and
                key.get("datatype", "text") in {"text", "number", "date"}, "Unsupported sort datatype")
        result.sort(key=lambda r: typed(r["fields"][key["column"]], key.get("datatype", "text")),
                    reverse=key.get("direction", "asc") == "desc")
    count = len(result)
    mode = spec.get("return_mode", "all_matches")
    require(isinstance(mode, str) and mode in {"all_matches", "top_k", "exact"}, "Invalid return_mode")
    if mode == "top_k":
        k = spec.get("top_k", default_k)
        require(type(k) is int and k > 0, "top_k must be positive")
        result = result[:k]
    if mode == "exact":
        require(type(spec.get("expected_count")) is int and len(result) == spec["expected_count"],
                f"Exact row count mismatch: matched {len(result)}")
    limited = len(result) < count
    complete = not limited or bool(order)
    return result, dict(matched_count=count, selection_limited=limited,
                        coverage="scope_complete" if complete else "candidate_only",
                        truncated=limited and not order)
