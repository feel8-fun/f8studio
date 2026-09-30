"""Workflow-side metadata builders owned by `service_bus`."""
from __future__ import annotations

from typing import Any

from ...state import StateWriteSource


def _resolved_source_text(source: StateWriteSource | str | None, *, default: str) -> str:
    if isinstance(source, StateWriteSource):
        return source.value
    text = str(source or "").strip()
    return text or str(default)


def build_rungraph_reconcile_meta() -> dict[str, Any]:
    """
    Metadata for rungraph-driven `stateValues` reconciliation writes.
    """
    return {
        "via": "rungraph",
        "rungraphReconcile": True,
    }


def build_builtin_identity_state_meta() -> dict[str, Any]:
    """
    Metadata for SDK-owned builtin identity state seeding.
    """
    return {"builtin": True}


def build_lifecycle_event_meta(
    *, source: StateWriteSource | str | None, meta: dict[str, Any] | None
) -> dict[str, Any]:
    """
    Metadata delivered to lifecycle callbacks and service hooks.

    Caller-provided metadata may override the default `source` field.
    """
    return {
        "source": _resolved_source_text(source, default=StateWriteSource.runtime.value),
        **(dict(meta or {})),
    }


def build_lifecycle_state_meta(*, meta: dict[str, Any] | None) -> dict[str, Any]:
    """
    Metadata persisted when the SDK writes builtin `active` lifecycle state.
    """
    return {
        "lifecycle": True,
        **(dict(meta or {})),
    }


__all__ = [
    "build_builtin_identity_state_meta",
    "build_lifecycle_event_meta",
    "build_lifecycle_state_meta",
    "build_rungraph_reconcile_meta",
]
