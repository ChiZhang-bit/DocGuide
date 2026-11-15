"""Parse the compact JSON protocol and the verifier's PASS response."""
from __future__ import annotations

import json


def strip_code_fence(text):
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if len(lines) >= 3 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return cleaned


def extract_json(text):
    cleaned = strip_code_fence(text)
    candidates = [cleaned]
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_char)
        end = cleaned.rfind(close_char)
        if start != -1 and end != -1 and start < end:
            candidates.append(cleaned[start:end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from LLM output: {text[:500]}")


def extract_verification_response(text):
    if strip_code_fence(text) == "PASS":
        return "PASS"
    return extract_json(text)
