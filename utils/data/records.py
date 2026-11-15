"""Typed records used by retrieval and API configuration."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(slots=True)
class LLMConfig:
    model: str
    base_url: str
    api_key: str
    timeout_seconds: float = 60

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.model and self.base_url)

@dataclass(slots=True)
class TableRow:
    row_index: int
    row_number: int
    values: dict[str, str]
    text: str
    description: str

    @property
    def source_label(self) -> str:
        return f"r{self.row_number}"

    @property
    def subject(self) -> str:
        return f"row_{self.row_number}"

@dataclass(slots=True)
class TextChunk:
    chunk_id: int
    paragraph_index: int
    paragraph_number: int
    chunk_index: int
    chunk_number: int
    sentence_start: int
    sentence_end: int
    text: str

    @property
    def source_label(self) -> str:
        return f"p{self.paragraph_number},c{self.chunk_number}"

    @property
    def subject(self) -> str:
        return f"chunk_p{self.paragraph_number}_c{self.chunk_number}"

@dataclass(slots=True)
class SearchHit:
    doc_id: int
    score: float

@dataclass(slots=True)
class TableHit:
    row: TableRow
    score: float
    rank_score: float = 0.0
    row_embedding_score: float = 0.0
    row_bm25_score: float = 0.0
    cell_embedding_score: float = 0.0
    cell_match_count: int = 0
    cell_match_score: float = 0.0
    row_embedding_rank: int = 0
    row_bm25_rank: int = 0
    cell_embedding_rank: int = 0
    cell_match_rank: int = 0
    anchor_column_name: str = ""
    anchor_value: str = ""
    matched_column_names: list[str] = field(default_factory=list)
    cell_match_query_source: str = "none"
