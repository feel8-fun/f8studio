from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path


FORBIDDEN_RUNTIME_TELEMETRY_STATE_FIELDS = {
    "observedFrames",
    "processedFrames",
    "droppedFrames",
    "failedFrames",
    "lastProcessMs",
    "avgProcessMs",
    "lastLatencyMs",
    "avgLatencyMs",
    "lastWaitMs",
    "avgWaitMs",
    "processFps",
    "lastVectorsPerFrame",
    "lastPointsPerFrame",
    "processMsAvg",
    "processMsP95",
    "waitMsAvg",
    "waitMsP95",
    "latencyMsAvg",
    "latencyMsP95",
}


SOURCE_ROOTS = (
    Path("packages"),
    Path("services"),
)


SOURCE_SUFFIXES = {
    ".cpp",
    ".h",
    ".hpp",
    ".py",
}


class ServiceTelemetryStateContractTest(unittest.TestCase):
    def test_service_describes_do_not_expose_runtime_telemetry_as_state_fields(self) -> None:
        root = os.environ.get("F8_CPP_DESCRIBE_ROOT")
        if not root:
            self.skipTest("built C++ contract test requires F8_CPP_DESCRIBE_ROOT")
        describe_paths = sorted(Path(root).rglob("describe.json"))
        self.assertTrue(describe_paths)

        leaks: dict[str, list[str]] = {}
        for describe_path in describe_paths:
            payload = json.loads(describe_path.read_text(encoding="utf-8"))
            leaked_names = self._leaked_state_field_names(payload)
            if leaked_names:
                leaks[str(describe_path)] = leaked_names

        self.assertEqual(leaks, {})

    def test_source_does_not_publish_runtime_telemetry_as_state(self) -> None:
        source_paths = self._source_paths()
        self.assertTrue(source_paths)

        leaks: dict[str, list[str]] = {}
        for source_path in source_paths:
            text = source_path.read_text(encoding="utf-8")
            leaked_names = []
            for field_name in sorted(FORBIDDEN_RUNTIME_TELEMETRY_STATE_FIELDS):
                if self._source_publishes_telemetry_state(text, field_name=field_name):
                    leaked_names.append(field_name)
            if leaked_names:
                leaks[str(source_path)] = leaked_names

        self.assertEqual(leaks, {})

    def _leaked_state_field_names(self, payload: object) -> list[str]:
        specs: list[dict[str, object]] = []
        if isinstance(payload, dict):
            service = payload.get("service")
            if isinstance(service, dict):
                specs.append(service)
            operators = payload.get("operators")
            if isinstance(operators, list):
                specs.extend(item for item in operators if isinstance(item, dict))

        leaked_names: set[str] = set()
        for spec in specs:
            state_fields = spec.get("stateFields")
            if not isinstance(state_fields, list):
                continue
            for field in state_fields:
                if not isinstance(field, dict):
                    continue
                name = field.get("name")
                if isinstance(name, str) and name in FORBIDDEN_RUNTIME_TELEMETRY_STATE_FIELDS:
                    leaked_names.add(name)

        return sorted(leaked_names)

    def _source_paths(self) -> list[Path]:
        paths: list[Path] = []
        for root in SOURCE_ROOTS:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                    paths.append(path)
        return sorted(paths)

    def _source_publishes_telemetry_state(self, text: str, *, field_name: str) -> bool:
        quoted = re.escape(field_name)
        patterns = (
            rf"state_field\(\s*[\"']{quoted}[\"']",
            rf"F8StateSpec\(\s*name\s*=\s*[\"']{quoted}[\"']",
            rf"StateField\(\s*name\s*=\s*[\"']{quoted}[\"']",
            rf"publish_state_if_changed\(\s*[\"']{quoted}[\"']",
            rf"publish_state_runtime\([^)]*[\"']{quoted}[\"']",
            rf"publish_state\([^)]*[\"']{quoted}[\"']",
        )
        return any(re.search(pattern, text) is not None for pattern in patterns)


if __name__ == "__main__":
    unittest.main()
