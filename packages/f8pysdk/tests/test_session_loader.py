import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.service_runtime_tools.session_loader import (  # noqa: E402
    SESSION_SCHEMA_VERSION,
    extract_layout,
    load_session_layout,
)


class SessionLoaderTests(unittest.TestCase):
    def test_extract_layout_v2(self) -> None:
        payload = {"schemaVersion": SESSION_SCHEMA_VERSION, "layout": {"nodes": {}, "connections": []}}
        layout = extract_layout(payload)
        self.assertIn("nodes", layout)

    def test_extract_layout_legacy(self) -> None:
        payload = {"nodes": {}, "connections": []}
        layout = extract_layout(payload)
        self.assertIn("nodes", layout)

    def test_load_session_layout_from_file(self) -> None:
        payload = {"schemaVersion": SESSION_SCHEMA_VERSION, "layout": {"nodes": {"n1": {}}, "connections": []}}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "session.json")
            with open(path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp)
            layout = load_session_layout(path)
        self.assertIn("n1", layout["nodes"])


if __name__ == "__main__":
    unittest.main()
