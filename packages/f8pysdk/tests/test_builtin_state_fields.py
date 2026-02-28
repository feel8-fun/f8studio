import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.builtin_state_fields import (  # noqa: E402
    normalize_describe_payload_dict,
    operator_state_fields_with_builtins,
    service_state_fields_with_builtins,
)
from f8pysdk.generated import F8StateAccess, F8StateSpec  # noqa: E402
from f8pysdk.nats_naming import kv_key_node_state  # noqa: E402
from f8pysdk.service_bus.codec import decode_obj  # noqa: E402
from f8pysdk.schema_helpers import boolean_schema, string_schema  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402


class BuiltinStateFieldTests(unittest.TestCase):
    def test_service_state_fields_force_override(self) -> None:
        fields = [
            F8StateSpec(
                name="active",
                label="Old Active",
                description="legacy",
                valueSchema=boolean_schema(default=False),
                access=F8StateAccess.ro,
                showOnNode=False,
            ),
            F8StateSpec(
                name="svcId",
                label="Legacy Service",
                description="legacy",
                valueSchema=string_schema(),
                access=F8StateAccess.rw,
                showOnNode=True,
            ),
            F8StateSpec(
                name="custom",
                valueSchema=string_schema(),
                access=F8StateAccess.rw,
            ),
        ]
        out = service_state_fields_with_builtins(fields)
        self.assertEqual([str(x.name) for x in out], ["custom", "active", "svcId"])
        self.assertEqual(out[-2].access, F8StateAccess.rw)
        self.assertTrue(bool(out[-2].showOnNode))
        self.assertEqual(out[-1].access, F8StateAccess.ro)
        self.assertFalse(bool(out[-1].showOnNode))

    def test_operator_state_fields_force_override(self) -> None:
        fields = [
            F8StateSpec(name="svcId", valueSchema=string_schema(), access=F8StateAccess.rw),
            F8StateSpec(name="operatorId", valueSchema=string_schema(), access=F8StateAccess.rw),
            F8StateSpec(name="mode", valueSchema=string_schema(), access=F8StateAccess.rw),
        ]
        out = operator_state_fields_with_builtins(fields)
        self.assertEqual([str(x.name) for x in out], ["mode", "svcId", "operatorId"])
        self.assertEqual(out[-2].access, F8StateAccess.ro)
        self.assertEqual(out[-1].access, F8StateAccess.ro)

    def test_normalize_describe_payload_dict_force_override(self) -> None:
        payload = {
            "schemaVersion": "f8describe/1",
            "service": {
                "schemaVersion": "f8service/1",
                "serviceClass": "f8.tests.svc",
                "version": "0.0.1",
                "label": "svc",
                "stateFields": [
                    {"name": "active", "valueSchema": {"type": "boolean"}, "access": "ro", "showOnNode": False},
                    {"name": "svcId", "valueSchema": {"type": "string"}, "access": "rw", "showOnNode": True},
                    {"name": "custom", "valueSchema": {"type": "string"}, "access": "rw"},
                ],
            },
            "operators": [
                {
                    "schemaVersion": "f8operator/1",
                    "serviceClass": "f8.tests.svc",
                    "operatorClass": "f8.tests.op",
                    "version": "0.0.1",
                    "label": "op",
                    "stateFields": [
                        {"name": "svcId", "valueSchema": {"type": "string"}, "access": "rw"},
                        {"name": "operatorId", "valueSchema": {"type": "string"}, "access": "rw"},
                        {"name": "threshold", "valueSchema": {"type": "number"}, "access": "rw"},
                    ],
                }
            ],
        }
        out = normalize_describe_payload_dict(payload)
        service_fields = out["service"]["stateFields"]
        operator_fields = out["operators"][0]["stateFields"]
        self.assertEqual([x["name"] for x in service_fields], ["custom", "active", "svcId"])
        self.assertEqual([x["name"] for x in operator_fields], ["threshold", "svcId", "operatorId"])


class LifecycleBootstrapTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_seeds_active_state(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        with patch("f8pysdk.service_bus.lifecycle._ensure_micro_endpoints_started") as ensure_micro:
            async def _noop(_bus: object) -> None:
                return None
            ensure_micro.side_effect = _noop
            await bus.start()
        state = await bus.get_state("svcA", "active")
        await bus.stop()
        self.assertTrue(state.found)
        self.assertTrue(bool(state.value))

    async def test_seeded_active_state_origin_runtime(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        with patch("f8pysdk.service_bus.lifecycle._ensure_micro_endpoints_started") as ensure_micro:
            async def _noop(_bus: object) -> None:
                return None
            ensure_micro.side_effect = _noop
            await bus.start()
        key = kv_key_node_state(node_id="svcA", field="active")
        raw = await bus._transport.kv_get(key)
        await bus.stop()
        self.assertIsNotNone(raw)
        payload = decode_obj(raw) if raw is not None else {}
        self.assertEqual(payload.get("origin"), "runtime")


if __name__ == "__main__":
    unittest.main()
