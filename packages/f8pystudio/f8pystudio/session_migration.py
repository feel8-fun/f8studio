from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SESSION_SCHEMA_VERSION = "f8studio-session/1"


@dataclass(frozen=True)
class SessionEnvelope:
    schema_version: str
    layout: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": str(self.schema_version),
            "layout": dict(self.layout),
        }


def wrap_layout_for_save(layout: dict[str, Any]) -> dict[str, Any]:
    return SessionEnvelope(schema_version=SESSION_SCHEMA_VERSION, layout=layout).to_dict()


def extract_layout(payload: Any) -> dict[str, Any]:
    """
    Accept only the v2 session envelope payload.
    """
    if not isinstance(payload, dict):
        raise ValueError("session payload must be a JSON object")

    schema_version = str(payload.get("schemaVersion") or "").strip()
    if schema_version != SESSION_SCHEMA_VERSION:
        raise ValueError(f"unsupported session schemaVersion: {schema_version!r}")

    layout = payload.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("v2 session payload missing `layout` object")
    return layout
