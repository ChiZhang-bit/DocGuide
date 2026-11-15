"""Generate schema-only row templates once, then reuse the local cache."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import string
import tempfile
import threading
from contextlib import contextmanager

from utils.common import canonical, require
from utils.prompt_loader import load_prompt

_CACHE = {}
_MODELS = {}
_LOCK = threading.RLock()
VERSION = "local-schema-template-v1"
_KEY_LOCKS = {}
TEMPLATE_PROMPT = load_prompt("row_template")


@contextmanager
def cache_lock(path):
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as handle:
        try:
            import fcntl
        except ImportError:
            yield
        else:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


class RowTemplateStore:
    def __init__(self, headers, model, cache_dir=None):
        self.headers = list(map(str, headers))
        self.model = model
        self.schema = dict(version=VERSION, headers=self.headers)
        self.key = hashlib.sha256(canonical(self.schema).encode()).hexdigest()
        self.path = Path(cache_dir) / (self.key + ".json") if cache_dir else None
        self.source = "fallback"
        self.error = None
        with _LOCK:
            if self.key not in _CACHE and self.path:
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    if data["schema"] == self.schema and (self.valid(data["template"]) or
                            data.get("fallback") is True and data["template"] is None):
                        _CACHE[self.key] = data["template"]
                        _MODELS[self.key] = data.get("generator_model")
                        self.error = data.get("error")
                        self.source = "disk_cache"
                except (OSError, ValueError, KeyError, TypeError):
                    pass
            if self.key in _CACHE and self.source == "fallback":
                self.source = "memory_cache" if _CACHE[self.key] else "cached_fallback"

    def valid(self, template):
        if not isinstance(template, str) or not template.strip() or len(template) > min(2048, 160 + 100 * len(self.headers)):
            return False
        try:
            fields = []
            for _, field, spec, conversion in string.Formatter().parse(template):
                if field is not None:
                    if spec or conversion:
                        return False
                    fields.append(field)
            return sorted(fields) == sorted(f"c{i}" for i in range(len(self.headers)))
        except ValueError:
            return False

    def accept(self, template):
        valid = self.valid(template)
        if not valid and not self.error:
            self.error = "Missing/invalid row template; using deterministic column-value description"
        with _LOCK:
            if _CACHE.get(self.key):
                self.source = "memory_cache"
                return
            _CACHE[self.key] = template if valid else None
            _MODELS[self.key] = self.model if valid else None
            self.source = "generated" if valid else "missing_or_invalid_fallback"
            if self.path:
                temporary = None
                try:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                     suffix=".tmp", delete=False) as handle:
                        temporary = handle.name
                        json.dump(dict(schema=self.schema, template=template if valid else None,
                                       fallback=not valid, generator_model=self.model if valid else None, error=self.error),
                                  handle, ensure_ascii=False)
                    os.replace(temporary, self.path)
                except OSError:
                    self.source += "_cache_write_failed"
                finally:
                    if temporary and Path(temporary).exists():
                        Path(temporary).unlink()

    def require_prepared(self):
        """QA only reuses preprocessing results, including cached fallbacks."""
        with _LOCK:
            present = self.key in _CACHE
        if not present and self.path:
            RowTemplateStore(self.headers, self.model, self.path.parent)
            with _LOCK:
                present = self.key in _CACHE
        require(present, "Missing row-template cache for this table schema. Run "
                "python -m scripts.prepare_row_templates for the selected dataset first.")
        return self.diagnostics()

    def prepare(self, llm):
        """One schema call on a cache miss; thread/process workers share its result."""
        with _LOCK:
            if self.key in _CACHE:
                return self.diagnostics()
            lock = _KEY_LOCKS.setdefault(self.key, threading.Lock())
        with lock, cache_lock(self.path):
            with _LOCK:
                if self.key in _CACHE:
                    return self.diagnostics()
            if self.path:
                RowTemplateStore(self.headers, self.model, self.path.parent)
                with _LOCK:
                    if self.key in _CACHE:
                        self.source = "disk_cache"
                        return self.diagnostics()
            if not self.headers:
                self.accept(None)
                return self.diagnostics()
            try:
                response = llm.ask("row_template", TEMPLATE_PROMPT,
                    dict(columns={f"c{i}": h for i, h in enumerate(self.headers)}),
                    max_tokens=min(1024, 96 + 24 * len(self.headers)))
                self.accept(response.get("template") if isinstance(response, dict) else None)
            except Exception as exc:
                self.error = str(exc)[:240]
                self.accept(None)
            return self.diagnostics()

    def retry_fallback(self):
        """Explicit preprocessing opt-in; ordinary QA never invalidates a cache."""
        with _LOCK:
            lock = _KEY_LOCKS.setdefault(self.key, threading.Lock())
        with lock, cache_lock(self.path), _LOCK:
            if _CACHE.get(self.key):
                return False
            if self.path and self.path.exists():
                try:
                    data = json.loads(self.path.read_text(encoding="utf-8"))
                    if data.get("schema") == self.schema and self.valid(data.get("template")):
                        _CACHE[self.key] = data["template"]
                        _MODELS[self.key] = data.get("generator_model")
                        return False
                except (OSError, ValueError, TypeError):
                    pass
                self.path.unlink()
            _CACHE.pop(self.key, None)
            _MODELS.pop(self.key, None)
            self.error, self.source = None, "explicit_retry"
            return True

    def render(self, values, row_number=None):
        with _LOCK:
            template = _CACHE.get(self.key)
        if template:
            description = template.format_map({f"c{i}": str(values.get(h, "")) for i, h in enumerate(self.headers)})
        else:
            description = "; ".join(f"{h} is {values.get(h, '')}" for h in self.headers if values.get(h, "")) + "."
        return (f"Row {row_number}: " if row_number is not None else "") + description

    def diagnostics(self):
        with _LOCK:
            template = _CACHE.get(self.key)
        return dict(cache_key=self.key, source=self.source, template=template, fallback=not bool(template),
                    generator_model=_MODELS.get(self.key), error=self.error)
