from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from _bootstrap import ensure_package_importable

ensure_package_importable()

from f8pystudio.session_migration import (
    SESSION_SCHEMA_VERSION_V2,
    extract_layout,
    migrate_legacy_layout,
    migrate_session_file,
    wrap_layout_for_save,
)


class SessionMigrationTests(unittest.TestCase):
    def test_wrap_layout_for_save(self) -> None:
        layout = {"nodes": {}, "connections": []}
        payload = wrap_layout_for_save(layout)
        self.assertEqual(payload["schemaVersion"], SESSION_SCHEMA_VERSION_V2)
        self.assertEqual(payload["layout"], layout)

    def test_extract_layout_v2(self) -> None:
        payload = {"schemaVersion": SESSION_SCHEMA_VERSION_V2, "layout": {"nodes": {"a": {}}, "connections": []}}
        layout = extract_layout(payload)
        self.assertIn("nodes", layout)
        self.assertIn("connections", layout)

    def test_extract_layout_legacy(self) -> None:
        payload = {"nodes": {"a": {}}, "connections": []}
        layout = extract_layout(payload)
        self.assertEqual(layout, payload)

    def test_extract_layout_v2_without_layout_raises(self) -> None:
        payload = {"schemaVersion": SESSION_SCHEMA_VERSION_V2}
        with self.assertRaisesRegex(ValueError, "missing `layout`"):
            _ = extract_layout(payload)

    def test_extract_layout_with_invalid_payload_type_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            _ = extract_layout(["bad", "payload"])

    def test_migrate_legacy_layout_wraps_session(self) -> None:
        legacy = {"nodes": {"a": {}}, "connections": []}
        migrated = migrate_legacy_layout(legacy)
        self.assertEqual(migrated["schemaVersion"], SESSION_SCHEMA_VERSION_V2)
        self.assertEqual(migrated["layout"], legacy)

    def test_migrate_session_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "session.json"
            path.write_text(json.dumps({"nodes": {"a": {}}, "connections": []}), encoding="utf-8")
            migrated = migrate_session_file(str(path))
            self.assertEqual(migrated, path.resolve())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("schemaVersion"), SESSION_SCHEMA_VERSION_V2)
            self.assertIn("layout", payload)


if __name__ == "__main__":
    unittest.main()
