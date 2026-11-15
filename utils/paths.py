"""Locations relative to this checkout, independent of the working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "prompt"
