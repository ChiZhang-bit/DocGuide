"""Read the relevance decisions already returned by the batched reasoning call."""
from __future__ import annotations



def logical_grade(value):
    """An omitted or malformed decision is unknown rather than a negative vote."""
    return value if type(value) is int and value in (0, 1) else None
