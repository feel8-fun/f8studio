from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode, F8StateAccess, F8StateSpec  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.specs import any_schema, number_schema, string_schema  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402
from f8pysdk.time_utils import now_ms  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators import state_expr as state_expr_mod  # noqa: E402
from f8pyengine.operators.state_expr import StateExprRuntimeNode  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402


def _build_state_expr_runtime_node(
    *,
    node_id: str,
    service_id: str,
    code: str,
    extra_state_fields: list[F8StateSpec] | None = None,
    extra_state_values: dict[str, object] | None = None,
) -> F8RuntimeNode:
    state_fields = list(StateExprRuntimeNode.SPEC.stateFields or [])
    if extra_state_fields:
        state_fields.extend(list(extra_state_fields))

    state_values: dict[str, object] = {"code": code}
    if extra_state_values:
        state_values.update(dict(extra_state_values))

    return F8RuntimeNode(
        nodeId=node_id,
        serviceId=service_id,
        serviceClass=SERVICE_CLASS,
        operatorClass=StateExprRuntimeNode.SPEC.operatorClass,
        stateFields=state_fields,
        stateValues=state_values,
        dataInPorts=[],
        dataOutPorts=[],
    )


def _monitor_error_message(bus: object) -> str:
    snapshot = bus.monitor_collector._build_snapshot(ts_ms=int(now_ms()))
    return str(snapshot.error.currentMessage or "")


class StateExprNodeTests(unittest.IsolatedAsyncioTestCase):
    def _setup_bus(self, *, service_id: str) -> object:
        harness = ServiceBusHarness()
        bus = harness.create_bus(service_id)
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
        return bus

    def test_registered_in_pyengine_specs(self) -> None:
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        desc = reg.describe(SERVICE_CLASS)
        operator_classes = {str(spec.operatorClass or "") for spec in list(desc.operators or [])}
        self.assertIn("f8.state_expr", operator_classes)

    def test_expr_state_field_is_required(self) -> None:
        code_fields = [field for field in list(StateExprRuntimeNode.SPEC.stateFields or []) if field.name == "code"]
        out_fields = [field for field in list(StateExprRuntimeNode.SPEC.stateFields or []) if field.name == "out"]
        self.assertEqual(len(code_fields), 1)
        self.assertEqual(len(out_fields), 1)
        self.assertTrue(code_fields[0].valueRequired)
        self.assertTrue(out_fields[0].valueRequired)
        self.assertEqual(out_fields[0].access, F8StateAccess.ro)

    async def test_maps_rw_state_fields_into_expression_symbols(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                _build_state_expr_runtime_node(
                    node_id="s1",
                    service_id="svcA",
                    code="a + b",
                    extra_state_fields=[
                        F8StateSpec(
                            name="a",
                            label="a",
                            description="",
                            valueSchema=number_schema(default=1.5),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        ),
                        F8StateSpec(
                            name="b",
                            label="b",
                            description="",
                            valueSchema=number_schema(default=2.5),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        ),
                    ],
                    extra_state_values={"a": 1.5, "b": 2.5},
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("s1")
        self.assertIsInstance(runtime, StateExprRuntimeNode)
        assert isinstance(runtime, StateExprRuntimeNode)

        await runtime.on_state("a", 4.0, ts_ms=1)
        self.assertEqual(await runtime.get_state_value("out"), 6.5)
        self.assertEqual(_monitor_error_message(bus), "")

    async def test_non_identifier_state_name_available_via_states_mapping(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g2",
            revision="r1",
            nodes=[
                _build_state_expr_runtime_node(
                    node_id="s1",
                    service_id="svcA",
                    code="states['left-value'] * gain",
                    extra_state_fields=[
                        F8StateSpec(
                            name="left-value",
                            label="left-value",
                            description="",
                            valueSchema=number_schema(default=2.0),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        ),
                        F8StateSpec(
                            name="gain",
                            label="gain",
                            description="",
                            valueSchema=number_schema(default=0.5),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        ),
                    ],
                    extra_state_values={"left-value": 2.0, "gain": 0.5},
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("s1")
        assert isinstance(runtime, StateExprRuntimeNode)

        await runtime.on_state("left-value", 8.0, ts_ms=1)
        self.assertEqual(await runtime.get_state_value("out"), 4.0)

    async def test_unwraps_json_like_output_into_plain_state_value(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g3",
            revision="r1",
            nodes=[
                _build_state_expr_runtime_node(
                    node_id="s1",
                    service_id="svcA",
                    code="config.center",
                    extra_state_fields=[
                        F8StateSpec(
                            name="config",
                            label="config",
                            description="",
                            valueSchema=any_schema(),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        )
                    ],
                    extra_state_values={"config": {"center": {"x": 3, "y": 9}}},
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("s1")
        assert isinstance(runtime, StateExprRuntimeNode)

        await runtime.on_state("config", {"center": {"x": 7, "y": 11}}, ts_ms=1)
        self.assertEqual(await runtime.get_state_value("out"), {"x": 7, "y": 11})

    async def test_compile_error_publishes_last_error_and_clears_out(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g4",
            revision="r1",
            nodes=[
                _build_state_expr_runtime_node(
                    node_id="s1",
                    service_id="svcA",
                    code="a +",
                    extra_state_fields=[
                        F8StateSpec(
                            name="a",
                            label="a",
                            description="",
                            valueSchema=number_schema(default=1.0),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        )
                    ],
                    extra_state_values={"a": 1.0},
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("s1")
        assert isinstance(runtime, StateExprRuntimeNode)

        await runtime.on_state("code", "a +", ts_ms=1)
        self.assertIsNone(await runtime.get_state_value("out"))
        self.assertIn("syntax error", _monitor_error_message(bus))

        await runtime.on_state("code", "a + 2", ts_ms=2)
        self.assertEqual(await runtime.get_state_value("out"), 3.0)
        self.assertEqual(_monitor_error_message(bus), "")

    @unittest.skipIf(state_expr_mod.np is None, "numpy not available in test environment")
    async def test_allow_numpy_enables_numpy_calls(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g5",
            revision="r1",
            nodes=[
                _build_state_expr_runtime_node(
                    node_id="s1",
                    service_id="svcA",
                    code="np.clip(value, 0, 1)",
                    extra_state_fields=[
                        F8StateSpec(
                            name="value",
                            label="value",
                            description="",
                            valueSchema=number_schema(default=1.5),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        )
                    ],
                    extra_state_values={"allowNumpy": True, "value": 1.5},
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("s1")
        assert isinstance(runtime, StateExprRuntimeNode)

        await runtime.on_state("value", 1.5, ts_ms=1)
        self.assertEqual(float(await runtime.get_state_value("out")), 1.0)

    async def test_ro_fields_do_not_become_expression_symbols(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g6",
            revision="r1",
            nodes=[
                _build_state_expr_runtime_node(
                    node_id="s1",
                    service_id="svcA",
                    code="public_value + hidden",
                    extra_state_fields=[
                        F8StateSpec(
                            name="public_value",
                            label="public_value",
                            description="",
                            valueSchema=number_schema(default=1.0),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        ),
                        F8StateSpec(
                            name="hidden",
                            label="hidden",
                            description="",
                            valueSchema=number_schema(default=9.0),
                            access=F8StateAccess.ro,
                            valueRequired=False,
                            showOnNode=True,
                        ),
                    ],
                    extra_state_values={"public_value": 1.0},
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("s1")
        assert isinstance(runtime, StateExprRuntimeNode)

        await runtime.on_state("public_value", 2.0, ts_ms=1)
        self.assertIsNone(await runtime.get_state_value("out"))
        self.assertIn("NameError", _monitor_error_message(bus))


if __name__ == "__main__":
    unittest.main()
