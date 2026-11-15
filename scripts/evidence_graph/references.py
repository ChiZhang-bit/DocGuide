"""Resolve references against the one working graph and original sources."""
from __future__ import annotations

from utils.common import ContractError

class GraphReferences:
    def resolve(self, ref):
        return self.sources.resolve(ref, self.nodes)


    def resolve_labeled(self, ref, field):
        try:
            return self.resolve(ref)
        except (ValueError, KeyError, TypeError) as exc:
            raise ContractError(f"{field}: {exc}") from exc
