from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class StateWriteOrigin(Enum):
    """
    Canonical write origin for state updates.

    Use this to drive access control and diagnostics instead of RTTI or
    ambiguous string sources.
    """

    external = "external"
    runtime = "runtime"
    rungraph = "rungraph"
    system = "system"


@dataclass(frozen=True)
class StateWriteContext:
    """
    Explicit write context for state updates.

    - origin: access policy selection
    - source: optional string for diagnostics ("state_edge_init", "endpoint", etc.)
    """

    origin: StateWriteOrigin
    source: str | None = None

    @property
    def resolved_source(self) -> str:
        return str(self.source or self.origin.value)


class StateWriteError(ValueError):
    """
    Structured error for rejecting state writes.
    """

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = str(code or "INVALID_VALUE")
        self.message = str(message)
        self.details = dict(details or {})
