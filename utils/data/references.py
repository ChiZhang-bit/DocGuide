"""Resolve exact evidence citations without retrieving additional records."""
import re
from ..common import require


def resolve_reference(ref: dict, records: dict) -> dict:
    """Resolve a located citation against supplied records, without retrieving more."""
    rid = ref.get("node_id")
    require(rid in records, f"Unknown source node {rid}")
    node = records[rid]
    if node["type"] == "table_record":
        col = ref.get("column")
        require(col in node["fields"], f"Unknown column {col} in {rid}")
        return dict(node_id=rid, column=col, value=node["fields"][col])
    require(node["type"] == "text_record", "Evidence refs must locate raw records")
    sentence = next((s for s in node["sentences"] if s["id"] == ref.get("sentence_id")), None)
    require(sentence is not None, "Unknown sentence ID")
    quote = ref.get("quote", "")
    require(isinstance(quote, str) and quote.strip(), "Missing exact quotation")
    positions = [m.start() for m in re.finditer(re.escape(quote), sentence["text"])]
    require(positions, "Quotation not found verbatim")
    occurrence = ref.get("occurrence")
    require(occurrence is not None or len(positions) == 1, "Ambiguous quotation; occurrence required")
    occurrence = 0 if occurrence is None else occurrence
    require(type(occurrence) is int and 0 <= occurrence < len(positions), "Invalid quotation occurrence")
    a = sentence["original_span"][0] + positions[occurrence]
    return dict(node_id=rid, sentence_id=sentence["id"], quote=quote, occurrence=occurrence,
                original_span=[a, a+len(quote)], value=quote)
