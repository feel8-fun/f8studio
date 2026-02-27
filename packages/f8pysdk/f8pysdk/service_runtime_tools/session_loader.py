from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SESSION_SCHEMA_VERSION = "f8studio-session/1"


def extract_layout(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("session payload must be a JSON object")

    schema_version = str(payload.get("schemaVersion") or "").strip()
    if schema_version == SESSION_SCHEMA_VERSION:
        layout = payload.get("layout")
        if not isinstance(layout, dict):
            raise ValueError("v2 session payload missing `layout` object")
        return layout

    if "nodes" in payload and isinstance(payload.get("nodes"), dict):
        return payload

    raise ValueError("unsupported session payload format")


def load_session_layout(path: str | Path) -> dict[str, Any]:
    session_path = Path(str(path or "").strip()).expanduser().resolve()
    if not session_path.is_file():
        raise FileNotFoundError(f"session file not found: {session_path}")
    raw = json.loads(session_path.read_text(encoding="utf-8"))
    return extract_layout(raw)
