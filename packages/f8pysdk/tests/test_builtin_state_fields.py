import os
import sys
import unittest
from typing import Any
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk._specs.builtin_fields import (  # noqa: E402
    MONITOR_PORT_NAME,
    normalize_describe_payload_dict,
    operator_state_fields_with_builtins,
    service_data_out_ports_with_builtins,
    service_state_fields_with_builtins,
)
from f8pysdk.specs import F8DataPortSpec, F8StateAccess, F8StateFieldEditPolicy, F8StateSpec  # noqa: E402
from f8pysdk.codec import decode_obj  # noqa: E402
from f8pysdk.zenoh_naming import zenoh_state_key  # noqa: E402
from f8pysdk.specs import (  # noqa: E402
    boolean_schema,
    can_delete_state_field,
    can_edit_state_field_access,
    can_edit_state_field_value_required,
    can_edit_state_field_value_schema,
    can_rename_state_field,
    F8RuntimeGraph,
    F8RuntimeNode,
    string_schema,
)
from f8pysdk.testing import ServiceBusHarness  # noqa: E402


LOCKED_BUILTIN_STATE_POLICY = {
    "canRename": False,
    "canEditAccess": False,
    "canEditValueRequired": False,
    "canEditValueSchema": False,
}


class _LifecycleRecordingNode:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self.lifecycle_calls: list[bool] = []

    def attach(self, bus: object) -> None:
        self._bus = bus

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del field, ts_ms, meta
        return value

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del field, value, ts_ms

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self.lifecycle_calls.append(bool(active))


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
        self.assertTrue(bool(out[-2].valueRequired))
        self.assertFalse(bool(out[-2].editPolicy.canRename))
        self.assertFalse(bool(out[-2].editPolicy.canEditAccess))
        self.assertFalse(bool(out[-2].editPolicy.canEditValueRequired))
        self.assertFalse(bool(out[-2].editPolicy.canEditValueSchema))
        self.assertFalse(bool(out[-2].showOnNode))
        self.assertEqual(out[-1].access, F8StateAccess.ro)
        self.assertTrue(bool(out[-1].valueRequired))
        self.assertFalse(bool(out[-1].editPolicy.canRename))
        self.assertFalse(bool(out[-1].editPolicy.canEditAccess))
        self.assertFalse(bool(out[-1].editPolicy.canEditValueRequired))
        self.assertFalse(bool(out[-1].editPolicy.canEditValueSchema))
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
        self.assertTrue(bool(out[-2].valueRequired))
        self.assertFalse(bool(out[-2].editPolicy.canRename))
        self.assertFalse(bool(out[-2].editPolicy.canEditAccess))
        self.assertFalse(bool(out[-2].editPolicy.canEditValueRequired))
        self.assertFalse(bool(out[-2].editPolicy.canEditValueSchema))
        self.assertEqual(out[-1].access, F8StateAccess.ro)
        self.assertTrue(bool(out[-1].valueRequired))
        self.assertFalse(bool(out[-1].editPolicy.canRename))
        self.assertFalse(bool(out[-1].editPolicy.canEditAccess))
        self.assertFalse(bool(out[-1].editPolicy.canEditValueRequired))
        self.assertFalse(bool(out[-1].editPolicy.canEditValueSchema))

    def test_required_state_value_schema_policy_defaults_to_editable_and_honors_lock(self) -> None:
        editable_value = F8StateSpec(
            name="value",
            valueSchema=string_schema(),
            access=F8StateAccess.rw,
            valueRequired=True,
        )
        readonly_preview = F8StateSpec(
            name="preview",
            valueSchema=string_schema(),
            access=F8StateAccess.ro,
            valueRequired=True,
        )
        locked_preview = F8StateSpec(
            name="lockedPreview",
            valueSchema=string_schema(),
            access=F8StateAccess.rw,
            valueRequired=True,
            editPolicy=F8StateFieldEditPolicy(canEditValueSchema=False),
        )

        self.assertTrue(can_edit_state_field_value_schema(editable_value))
        self.assertTrue(can_edit_state_field_value_schema(readonly_preview))
        self.assertFalse(can_edit_state_field_value_schema(locked_preview))

    def test_required_state_identity_policy_defaults_to_editable(self) -> None:
        field = F8StateSpec(
            name="value",
            valueSchema=string_schema(),
            access=F8StateAccess.rw,
            valueRequired=True,
        )

        self.assertTrue(can_rename_state_field(field))
        self.assertTrue(can_edit_state_field_access(field))
        self.assertTrue(can_edit_state_field_value_required(field))
        self.assertTrue(can_delete_state_field(field))

    def test_optional_state_field_policy_can_disable_identity_edits(self) -> None:
        field = F8StateSpec(
            name="value",
            valueSchema=string_schema(),
            access=F8StateAccess.rw,
            valueRequired=False,
            editPolicy=F8StateFieldEditPolicy(
                canRename=False,
                canEditAccess=False,
                canEditValueRequired=False,
            ),
        )

        self.assertFalse(can_rename_state_field(field))
        self.assertFalse(can_edit_state_field_access(field))
        self.assertFalse(can_edit_state_field_value_required(field))
        self.assertFalse(can_delete_state_field(field))

    def test_service_data_out_ports_force_monitor(self) -> None:
        ports = [
            F8DataPortSpec(name="telemetry", valueSchema=string_schema()),
            F8DataPortSpec(name="out", valueSchema=string_schema()),
            F8DataPortSpec(name="monitor", valueSchema=string_schema()),
        ]
        out = service_data_out_ports_with_builtins(ports)
        names = [str(port.name) for port in out]
        self.assertEqual(names.count(MONITOR_PORT_NAME), 1)
        self.assertIn("out", names)
        self.assertIn("telemetry", names)
        monitor_ports = [port for port in out if str(port.name) == MONITOR_PORT_NAME]
        self.assertEqual(len(monitor_ports), 1)
        self.assertTrue(bool(monitor_ports[0].definitionProtected))
        self.assertFalse(bool(monitor_ports[0].showOnNode))

    def test_normalize_describe_payload_dict_force_override(self) -> None:
        payload = {
            "schemaVersion": "f8describe/1",
            "service": {
                "schemaVersion": "f8service/1",
                "serviceClass": "f8.tests.svc",
                "version": "0.0.1",
                "label": "svc",
                "dataOutPorts": [
                    {"name": "telemetry", "valueSchema": {"type": "string"}},
                ],
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
        active_fields = [x for x in service_fields if str(x.get("name")) == "active"]
        self.assertEqual(len(active_fields), 1)
        self.assertTrue(bool(active_fields[0].get("valueRequired")))
        self.assertFalse(bool(active_fields[0].get("showOnNode")))
        self.assertEqual(active_fields[0].get("editPolicy"), LOCKED_BUILTIN_STATE_POLICY)
        self.assertEqual([x["name"] for x in operator_fields], ["threshold", "svcId", "operatorId"])
        svc_id_fields = [x for x in service_fields if str(x.get("name")) == "svcId"]
        self.assertEqual(len(svc_id_fields), 1)
        self.assertTrue(bool(svc_id_fields[0].get("valueRequired")))
        self.assertEqual(svc_id_fields[0].get("editPolicy"), LOCKED_BUILTIN_STATE_POLICY)
        operator_svc_id_fields = [x for x in operator_fields if str(x.get("name")) == "svcId"]
        self.assertEqual(len(operator_svc_id_fields), 1)
        self.assertTrue(bool(operator_svc_id_fields[0].get("valueRequired")))
        self.assertEqual(operator_svc_id_fields[0].get("editPolicy"), LOCKED_BUILTIN_STATE_POLICY)
        operator_id_fields = [x for x in operator_fields if str(x.get("name")) == "operatorId"]
        self.assertEqual(len(operator_id_fields), 1)
        self.assertTrue(bool(operator_id_fields[0].get("valueRequired")))
        self.assertEqual(operator_id_fields[0].get("editPolicy"), LOCKED_BUILTIN_STATE_POLICY)
        service_data_ports = out["service"]["dataOutPorts"]
        self.assertTrue(any(str(x.get("name")) == MONITOR_PORT_NAME for x in service_data_ports))
        self.assertTrue(any(str(x.get("name")) == "telemetry" for x in service_data_ports))
        monitor_ports = [x for x in service_data_ports if str(x.get("name")) == MONITOR_PORT_NAME]
        self.assertEqual(len(monitor_ports), 1)
        self.assertTrue(bool(monitor_ports[0].get("definitionProtected")))
        self.assertFalse(bool(monitor_ports[0].get("showOnNode")))

    def test_normalize_local_describe_snapshot_to_authoring_schema(self) -> None:
        payload = {
            "service": {
                "serviceClass": "f8.tests.svc",
                "label": "Service",
                "launch": {"command": "legacy"},
                "stateFields": [{"name": "device", "valueSchema": {"type": "string"},
                                 "required": True, "uiControl": "select[devices]",
                                 "editPolicy": {"canEditRequired": False}}],
                "dataInPorts": [{"name": "input", "valueSchema": {"type": "string"}, "required": False}],
                "commands": [{"name": "open", "required": True,
                              "params": [{"name": "path", "valueSchema": {"type": "string"},
                                          "required": True, "uiControl": "wrapline"}]}],
            },
            "operators": [{"operatorClass": "f8.tests.op", "label": "Operator",
                           "execInPorts": ["run"], "execOutPorts": ["done"]}],
        }

        normalized = normalize_describe_payload_dict(payload)
        service = normalized["service"]
        self.assertNotIn("launch", service)
        device = service["stateFields"][0]
        self.assertTrue(device["valueRequired"])
        self.assertEqual(device["control"], {"kind": "select", "optionsFromState": "devices"})
        self.assertFalse(device["editPolicy"]["canEditValueRequired"])
        self.assertFalse(service["dataInPorts"][0]["definitionProtected"])
        command = service["commands"][0]
        self.assertTrue(command["definitionProtected"])
        self.assertTrue(command["params"][0]["valueRequired"])
        self.assertEqual(command["params"][0]["control"], {"kind": "textarea"})
        self.assertEqual(normalized["operators"][0]["execInPorts"], [{"name": "run"}])
        self.assertEqual(normalized["operators"][0]["execOutPorts"], [{"name": "done"}])


class LifecycleBootstrapTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_seeds_active_state(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        with patch("f8pysdk.service_bus.workflow.lifecycle._ensure_control_endpoints_started") as ensure_control:

            async def _noop(_bus: object) -> None:
                return None

            ensure_control.side_effect = _noop
            await bus.start()
        state = await bus.get_state("svcA", "active")
        await bus.stop()
        self.assertTrue(state.found)
        self.assertTrue(bool(state.value))

    async def test_seeded_active_state_origin_runtime(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        with patch("f8pysdk.service_bus.workflow.lifecycle._ensure_control_endpoints_started") as ensure_control:

            async def _noop(_bus: object) -> None:
                return None

            ensure_control.side_effect = _noop
            await bus.start()
        key = zenoh_state_key("svcA", node_id="svcA", field="active")
        raw = await bus._transport.retained_get(key)
        await bus.stop()
        self.assertIsNotNone(raw)
        payload = decode_obj(raw) if raw is not None else {}
        self.assertEqual(payload.get("origin"), "runtime")

    async def test_external_active_state_write_applies_lifecycle(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        node = _LifecycleRecordingNode("svcA")
        bus.register_node(node)

        await bus.publish_state_external("svcA", "active", False)

        self.assertFalse(bus.active)
        self.assertEqual(node.lifecycle_calls, [False])
        state = await bus.get_state("svcA", "active")
        self.assertTrue(state.found)
        self.assertFalse(bool(state.value))

    async def test_rungraph_active_state_value_applies_lifecycle(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        node = _LifecycleRecordingNode("svcA")
        bus.register_node(node)

        service_node = F8RuntimeNode(
            nodeId="svcA",
            serviceId="svcA",
            serviceClass="svc.test",
            operatorClass=None,
            stateFields=service_state_fields_with_builtins([]),
            stateValues={"active": False},
        )
        graph = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[service_node], edges=[])

        await bus.set_rungraph(graph)

        self.assertFalse(bus.active)
        self.assertEqual(node.lifecycle_calls, [False])
        state = await bus.get_state("svcA", "active")
        self.assertTrue(state.found)
        self.assertFalse(bool(state.value))

    async def test_external_active_state_rejects_invalid_value(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")

        with self.assertRaisesRegex(ValueError, "active must be a boolean"):
            await bus.publish_state_external("svcA", "active", "maybe")
        self.assertTrue(bus.active)


if __name__ == "__main__":
    unittest.main()
