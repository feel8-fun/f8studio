from __future__ import annotations

import os
import sys
import unittest
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.monitoring import (  # noqa: E402
    MonitorContractError,
    monitor_snapshot_schema_dict,
    validate_monitor_snapshot_payload,
    validate_describe_monitor_contract,
)


def _describe_payload_with_ports(ports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schemaVersion": "f8describe/1",
        "service": {
            "schemaVersion": "f8service/1",
            "serviceClass": "f8.tests.svc",
            "version": "0.0.1",
            "label": "svc",
            "dataOutPorts": ports,
        },
        "operators": [],
    }


class MonitorSchemaTests(unittest.TestCase):
    def test_validate_describe_monitor_contract_passes_with_monitor_port(self) -> None:
        payload = _describe_payload_with_ports(
            [
                {
                    "name": "monitor",
                    "valueSchema": monitor_snapshot_schema_dict(),
                    "definitionProtected": True,
                    "showOnNode": True,
                    "description": "Unified runtime monitor snapshots (health/resource/perf/error).",
                }
            ]
        )
        validate_describe_monitor_contract(payload)

    def test_validate_describe_monitor_contract_rejects_missing_monitor(self) -> None:
        payload = _describe_payload_with_ports(
            [
                {
                    "name": "out",
                    "valueSchema": {"type": "string"},
                    "definitionProtected": False,
                    "showOnNode": True,
                    "description": "output",
                }
            ]
        )
        with self.assertRaises(MonitorContractError):
            validate_describe_monitor_contract(payload)

    def test_validate_describe_monitor_contract_rejects_telemetry_port(self) -> None:
        payload = _describe_payload_with_ports(
            [
                {
                    "name": "monitor",
                    "valueSchema": monitor_snapshot_schema_dict(),
                    "definitionProtected": True,
                    "showOnNode": True,
                    "description": "Unified runtime monitor snapshots (health/resource/perf/error).",
                },
                {
                    "name": "telemetry",
                    "valueSchema": {"type": "object"},
                    "definitionProtected": False,
                    "showOnNode": True,
                    "description": "legacy stream",
                },
            ]
        )
        with self.assertRaises(MonitorContractError):
            validate_describe_monitor_contract(payload)

    def test_validate_describe_monitor_contract_rejects_unprotected_monitor(self) -> None:
        payload = _describe_payload_with_ports(
            [
                {
                    "name": "monitor",
                    "valueSchema": monitor_snapshot_schema_dict(),
                    "definitionProtected": False,
                    "showOnNode": True,
                    "description": "Unified runtime monitor snapshots (health/resource/perf/error).",
                }
            ]
        )
        with self.assertRaises(MonitorContractError):
            validate_describe_monitor_contract(payload)

    def test_validate_monitor_snapshot_payload_accepts_legacy_error_block(self) -> None:
        payload: dict[str, Any] = {
            "schemaVersion": "f8monitor/1",
            "serviceId": "svcA",
            "serviceClass": "f8.tests",
            "nodeId": "svcA",
            "tsMs": 1000,
            "alive": True,
            "ready": True,
            "active": True,
            "uptimeMs": 100,
            "cpu": {"processPercent": 0.0, "systemPercent": 0.0},
            "memory": {"rssBytes": 0, "vmsBytes": 0},
            "gpu": {
                "vendor": "",
                "deviceIndex": None,
                "utilPercent": None,
                "memoryUsedBytes": None,
                "memoryTotalBytes": None,
                "available": False,
            },
            "frame": {
                "observed": 0,
                "processed": 0,
                "dropped": 0,
                "localOnlyEmits": 0,
                "routedCrossEmits": 0,
                "suppressedCrossPublishes": 0,
                "callbackDeliveries": 0,
                "bufferPullDeliveries": 0,
            },
            "timing": {
                "processMsAvg": None,
                "processMsP95": None,
                "waitMsAvg": None,
                "waitMsP95": None,
                "latencyMsAvg": None,
                "latencyMsP95": None,
            },
            "queue": {"depth": 0},
            "error": {
                "countWindow": 1,
                "lastCode": "E_OLD",
                "lastMessage": "legacy payload",
                "lastTsMs": 990,
            },
        }

        snapshot = validate_monitor_snapshot_payload(payload)

        self.assertEqual(str(snapshot.error.lastCode), "E_OLD")
        self.assertEqual(str(snapshot.error.lastMessage), "legacy payload")
        self.assertEqual(str(snapshot.error.lastNodeId), "")
        self.assertEqual(int(snapshot.error.lastRepeatCount), 0)


if __name__ == "__main__":
    unittest.main()
