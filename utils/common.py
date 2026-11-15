"""Shared serialization, identifiers, contracts and artifact writing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from .paths import ROOT
from .contracts import ContractError, require
from .config import Config

HOME = ROOT

def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def stable_id(prefix: str, value: Any) -> str:
    return prefix + hashlib.sha256(canonical(value).encode()).hexdigest()[:16]

def write_json(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(HOME), "Outputs must stay inside DocGuide")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)

def append_jsonl(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(HOME), "Outputs must stay inside DocGuide")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(value)+"\n")
