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

class StateWriteSource(Enum):
    """
    Canonical write source for diagnostics and loop-guards.

    Unlike `StateWriteOrigin`, this does not drive access policy; it describes
    *how* a state update was produced (e.g. propagated along edges vs. direct
    runtime write).
    """

    runtime = "runtime"
    rungraph = "rungraph"
    system = "system"
    cmd = "cmd"
    endpoint = "endpoint"

    # State-edge propagation (intra-service).
    state_edge_intra = "state_edge_intra"
    state_edge_intra_init = "state_edge_intra_init"

    # State-edge propagation (cross-service via remote KV watch/binding).
    state_edge_cross = "state_edge_cross"


@dataclass(frozen=True)
class StateWriteContext:
    """
    Explicit write context for state updates.

    - origin: access policy selection
    - source: optional source tag for diagnostics (prefer `StateWriteSource`)
    """

    origin: StateWriteOrigin
    source: StateWriteSource | str | None = None

    @property
    def resolved_source(self) -> str:
        s = self.source
        if isinstance(s, StateWriteSource):
            return s.value
        return str(s or self.origin.value)


class StateWriteError(ValueError):
    """
    Structured error for rejecting state writes.
    """

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = str(code or "INVALID_VALUE")
        self.message = str(message)
        self.details = dict(details or {})
