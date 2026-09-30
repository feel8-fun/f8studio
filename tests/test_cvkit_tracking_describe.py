from __future__ import annotations

import json
import os
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
    "processFps",
    "lastVectorsPerFrame",
    "lastPointsPerFrame",
}


def built_services_root() -> Path:
    root = os.environ.get("F8_CPP_DESCRIBE_ROOT")
    if not root:
        raise unittest.SkipTest("built C++ contract test requires F8_CPP_DESCRIBE_ROOT")
    return Path(root)


class CvkitTrackingDescribeTest(unittest.TestCase):
    def test_tracking_describe_exposes_current_configurable_state_fields(self) -> None:
        describe_path = (built_services_root() / "f8/cvkit/tracking/describe.json")
        payload = json.loads(describe_path.read_text(encoding="utf-8"))

        service = payload["service"]
        state_fields = service["stateFields"]
        init_select_field = next(field for field in state_fields if field["name"] == "initSelect")
        stop_cooldown_field = next(field for field in state_fields if field["name"] == "stopTrackingCooldownMs")
        active_field = next(field for field in state_fields if field["name"] == "active")

        self.assertEqual(init_select_field["access"], "rw")
        self.assertEqual(init_select_field["valueSchema"]["default"], "closest_center")
        self.assertEqual(
            init_select_field["valueSchema"]["enum"],
            ["first_box", "closest_center", "largest_area", "highest_score"],
        )
        self.assertEqual(stop_cooldown_field["access"], "rw")
        self.assertEqual(stop_cooldown_field["valueSchema"]["default"], 1000)
        self.assertEqual(stop_cooldown_field["valueSchema"]["minimum"], 0.0)
        self.assertEqual(stop_cooldown_field["valueSchema"]["maximum"], 60000.0)
        self.assertEqual(active_field["access"], "rw")
        self.assertIs(active_field["valueSchema"]["default"], True)

    def test_tracking_describe_keeps_process_metrics_off_state_fields(self) -> None:
        describe_path = (built_services_root() / "f8/cvkit/tracking/describe.json")
        payload = json.loads(describe_path.read_text(encoding="utf-8"))

        service = payload["service"]
        state_fields = service["stateFields"]
        state_field_names = {field["name"] for field in state_fields}
        data_out_port_names = {port["name"] for port in service["dataOutPorts"]}

        self.assertFalse(FORBIDDEN_RUNTIME_TELEMETRY_STATE_FIELDS & state_field_names)
        self.assertIn("monitor", data_out_port_names)

    def test_cvkit_describes_do_not_expose_runtime_telemetry_as_state_fields(self) -> None:
        describe_paths = sorted((built_services_root() / "f8/cvkit").rglob("describe.json"))
        self.assertTrue(describe_paths)

        leaks: dict[str, list[str]] = {}
        for describe_path in describe_paths:
            payload = json.loads(describe_path.read_text(encoding="utf-8"))
            service = payload["service"]
            state_field_names = {field["name"] for field in service["stateFields"]}
            leaked_names = sorted(FORBIDDEN_RUNTIME_TELEMETRY_STATE_FIELDS & state_field_names)
            if leaked_names:
                leaks[str(describe_path)] = leaked_names

        self.assertEqual(leaks, {})

    def test_cvkit_cpp_sources_do_not_publish_runtime_telemetry_as_state(self) -> None:
        source_paths = sorted(Path("packages/f8cvkit/src/services").rglob("*.cpp"))
        source_paths += sorted(Path("packages/f8cvkit/src/services").rglob("*.h"))
        self.assertTrue(source_paths)

        leaks: dict[str, list[str]] = {}
        for source_path in source_paths:
            text = source_path.read_text(encoding="utf-8")
            leaked_names = []
            for field_name in sorted(FORBIDDEN_RUNTIME_TELEMETRY_STATE_FIELDS):
                state_field_marker = f'state_field("{field_name}"'
                state_publish_marker = f'publish_state_if_changed("{field_name}"'
                if state_field_marker in text or state_publish_marker in text:
                    leaked_names.append(field_name)
            if leaked_names:
                leaks[str(source_path)] = leaked_names

        self.assertEqual(leaks, {})


if __name__ == "__main__":
    unittest.main()
