from __future__ import annotations

from f8pysdk.codec import dump_json, validate_as
from typing import Any

from .generated import F8EditorAssistSpec


def validate_editor_assist_spec(payload: F8EditorAssistSpec | dict[str, Any]) -> F8EditorAssistSpec:
    """Validate and normalize one editor-assist payload."""
    if isinstance(payload, F8EditorAssistSpec):
        return payload
    return validate_as(F8EditorAssistSpec, payload)


def dump_editor_assist_spec(spec: F8EditorAssistSpec) -> dict[str, Any]:
    """Dump editor-assist payload in JSON-compatible form."""
    return dump_json(spec)

