"""Read complete prompt templates without altering whitespace."""
from functools import lru_cache
from .paths import PROMPT_ROOT

@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    if not name or not all(c.isalnum() or c == "_" for c in name):
        raise ValueError("Invalid prompt name")
    return (PROMPT_ROOT / f"{name}.md").read_text(encoding="utf-8")
