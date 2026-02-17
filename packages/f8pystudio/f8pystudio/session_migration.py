from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SESSION_SCHEMA_VERSION_V2 = "f8studio-session/2"


@dataclass(frozen=True)
class SessionEnvelopeV2:
    schema_version: str
    layout: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": str(self.schema_version),
            "layout": dict(self.layout),
        }


def wrap_layout_for_save(layout: dict[str, Any]) -> dict[str, Any]:
    return SessionEnvelopeV2(schema_version=SESSION_SCHEMA_VERSION_V2, layout=layout).to_dict()


def extract_layout(payload: Any) -> dict[str, Any]:
    """
    Accept v2 envelope and legacy NodeGraphQt layout payload.
    """
    if not isinstance(payload, dict):
        raise ValueError("session payload must be a JSON object")

    schema_version = str(payload.get("schemaVersion") or "").strip()
    if schema_version == SESSION_SCHEMA_VERSION_V2:
        layout = payload.get("layout")
        if not isinstance(layout, dict):
            raise ValueError("v2 session payload missing `layout` object")
        return layout

    if "nodes" in payload and isinstance(payload.get("nodes"), dict):
        return payload

    raise ValueError("unsupported session payload format")


def migrate_legacy_layout(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Upgrade legacy layout payload to v2 envelope.
    """
    layout = extract_layout(payload)
    return wrap_layout_for_save(layout)


def migrate_session_file(path: str) -> Path:
    """
    One-time migration helper for session files.
    """
    session_path = Path(str(path or "").strip()).expanduser().resolve()
    if not session_path.is_file():
        raise FileNotFoundError(f"session file not found: {session_path}")
    raw = json.loads(session_path.read_text(encoding="utf-8"))
    upgraded = migrate_legacy_layout(raw)
    session_path.write_text(json.dumps(upgraded, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return session_path


def _main() -> int:
    parser = argparse.ArgumentParser(description="Migrate f8studio session files to f8studio-session/2")
    parser.add_argument("session_file", help="Path to session json file")
    args = parser.parse_args()

    migrated_path = migrate_session_file(args.session_file)
    print(f"migrated: {migrated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
