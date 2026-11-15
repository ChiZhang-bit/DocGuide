"""Serialization of full rows and text chunks for retrieval."""
from __future__ import annotations
from typing import Any
from utils.data.records import TableRow, TextChunk

def normalize_cell(value: Any) -> str:
    return " ".join(str(value).split())

def normalize_header(value: Any) -> str:
    text = normalize_cell(value)
    return text if text else "column"

def build_table_row_description(row_number: int, values: dict[str, str]) -> str:
    facts = []
    for header, value in values.items():
        if value:
            facts.append(f"{header} is {value}")
    if not facts:
        return f"Row {row_number} is empty."
    return f"Row {row_number}: " + "; ".join(facts) + "."

def table_to_rows(table: dict[str, Any], row_formatter=None) -> list[TableRow]:
    headers = [normalize_header(item) for item in table.get("header", [])]
    rows: list[TableRow] = []
    for row_index, raw_row in enumerate(table.get("data", [])):
        values: dict[str, str] = {}
        for col_index, header in enumerate(headers):
            cell = raw_row[col_index] if col_index < len(raw_row) else ""
            values[header] = normalize_cell(cell)
        text = " | ".join(f"{key}: {value}" for key, value in values.items())
        rows.append(
            TableRow(
                row_index=row_index,
                row_number=row_index + 1,
                values=values,
                text=text,
                description=(row_formatter(values, row_index + 1) if row_formatter else
                             build_table_row_description(row_index + 1, values)),
            )
        )
    return rows

def chunk_search_text(chunk: TextChunk) -> str:
    return f"{chunk.source_label} {chunk.text}"
