from __future__ import annotations

import os
import sys
import unittest
import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode, F8StateAccess, F8StateSpec  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.specs import any_schema, number_schema, string_schema  # noqa: E402
from f8pysdk.testing import buffer_input  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402
from f8pysdk.time_utils import now_ms  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.wave_expr import WaveExprRuntimeNode  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402


def _build_wave_expr_runtime_node(
    *,
    node_id: str,
    service_id: str,
    template: str,
    max_t: float,
    extra_state_fields: list[F8StateSpec] | None = None,
    extra_state_values: dict[str, object] | None = None,
) -> F8RuntimeNode:
    state_fields = list(WaveExprRuntimeNode.SPEC.stateFields or [])
    if extra_state_fields:
        state_fields.extend(list(extra_state_fields))

    state_values: dict[str, object] = {
        "template": template,
        "maxT": max_t,
    }
    if extra_state_values:
        state_values.update(dict(extra_state_values))

    return F8RuntimeNode(
        nodeId=node_id,
        serviceId=service_id,
        serviceClass=SERVICE_CLASS,
        operatorClass=WaveExprRuntimeNode.SPEC.operatorClass,
        stateFields=state_fields,
        stateValues=state_values,
        dataInPorts=list(WaveExprRuntimeNode.SPEC.dataInPorts or []),
        dataOutPorts=list(WaveExprRuntimeNode.SPEC.dataOutPorts or []),
    )


def _monitor_error_message(bus: object) -> str:
    snapshot = bus.monitor_collector._build_snapshot(ts_ms=int(now_ms()))
    return str(snapshot.error.currentMessage or "")


class WaveExprNodeTests(unittest.IsolatedAsyncioTestCase):
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
        self.assertIn("f8.wave_expr", operator_classes)

    def test_template_is_read_write_configuration_state(self) -> None:
        template_fields = [field for field in list(WaveExprRuntimeNode.SPEC.stateFields or []) if field.name == "template"]
        self.assertEqual(len(template_fields), 1)
        self.assertEqual(template_fields[0].access, F8StateAccess.rw)
        self.assertTrue(template_fields[0].valueRequired)

    async def test_generates_express_with_state_variables_and_max_t(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="a * cos(t) + b + maxT * 0",
                    max_t=5.0,
                    extra_state_fields=[
                        F8StateSpec(
                            name="a",
                            label="a",
                            description="",
                            valueSchema=number_schema(default=0.25),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        ),
                        F8StateSpec(
                            name="b",
                            label="b",
                            description="",
                            valueSchema=number_schema(default=0.75),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        ),
                    ],
                    extra_state_values={"a": 0.25, "b": 0.75},
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        self.assertIsInstance(runtime, WaveExprRuntimeNode)
        assert isinstance(runtime, WaveExprRuntimeNode)

        await runtime.on_state("template", "a * cos(t) + b + maxT * 0", ts_ms=1)
        express = str(await runtime.get_state_value("express") or "")

        self.assertIn("cos(t)", express)
        self.assertIn("0.25", express)
        self.assertIn("0.75", express)
        self.assertIn("5.0", express)

        preview = await runtime.get_state_value("preview")
        self.assertIsInstance(preview, list)
        self.assertGreaterEqual(len(preview or []), 32)
        first_point = list(preview or [])[0]
        self.assertIsInstance(first_point, list)
        self.assertEqual(len(first_point), 2)

    async def test_compute_output_from_scalar_t(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g2",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="cond(t > 5, 1, 2)",
                    max_t=10.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 4.0, ts_ms=0, edge=None, ctx_id=None)
        out1 = await runtime.compute_output("value", ctx_id=1)
        self.assertEqual(out1, 2.0)

        buffer_input(bus, "w1", "t", 6.0, ts_ms=0, edge=None, ctx_id=None)
        out2 = await runtime.compute_output("value", ctx_id=2)
        self.assertEqual(out2, 1.0)

    async def test_compute_output_wraps_t_by_max_t(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g2b",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="t",
                    max_t=10.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 12.5, ts_ms=0, edge=None, ctx_id=None)
        out = await runtime.compute_output("value", ctx_id=22)
        self.assertEqual(out, 2.5)

    async def test_sequence_function_uses_integer_t_index(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g2c",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="sequence([10, 20, 30, 20, 10])",
                    max_t=10.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 0.2, ts_ms=0, edge=None, ctx_id=None)
        out0 = await runtime.compute_output("value", ctx_id=30)
        self.assertEqual(out0, 10.0)

        buffer_input(bus, "w1", "t", 1.8, ts_ms=0, edge=None, ctx_id=None)
        out1 = await runtime.compute_output("value", ctx_id=31)
        self.assertEqual(out1, 20.0)

        buffer_input(bus, "w1", "t", 4.9, ts_ms=0, edge=None, ctx_id=None)
        out4 = await runtime.compute_output("value", ctx_id=32)
        self.assertEqual(out4, 10.0)

        buffer_input(bus, "w1", "t", 5.1, ts_ms=0, edge=None, ctx_id=None)
        out5 = await runtime.compute_output("value", ctx_id=33)
        self.assertEqual(out5, 10.0)

    async def test_sequence_function_populates_preview(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g2d",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="sequence([10, 20, 30])",
                    max_t=6.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 0.0, ts_ms=0, edge=None, ctx_id=None)
        _ = await runtime.compute_output("value", ctx_id=34)

        preview = list(await runtime.get_state_value("preview") or [])
        self.assertGreaterEqual(len(preview), 64)
        self.assertEqual(float(preview[0][1]), 10.0)

        preview_values = {float(point[1]) for point in preview[:128]}
        self.assertIn(10.0, preview_values)
        self.assertIn(20.0, preview_values)
        self.assertIn(30.0, preview_values)

    async def test_tempest_function_uses_cycle_t_not_radians(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g2e",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="tempest(t, 0.5, 0.25)",
                    max_t=10.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 0.25, ts_ms=0, edge=None, ctx_id=None)
        out = await runtime.compute_output("value", ctx_id=35)

        theta = 2.0 * math.pi * 0.25
        phase = theta + ((math.pi * 0.5) / 2.0)
        expected = -math.cos(phase + (0.25 * math.sin(phase)))
        self.assertAlmostEqual(out, expected, places=9)

    async def test_tempest_function_populates_preview(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g2f",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="tempest(t, 0.25, 0.5)",
                    max_t=2.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 0.0, ts_ms=0, edge=None, ctx_id=None)
        _ = await runtime.compute_output("value", ctx_id=36)

        preview = list(await runtime.get_state_value("preview") or [])
        self.assertGreaterEqual(len(preview), 64)
        first_value = float(preview[0][1])
        theta = 0.0
        phase = theta + ((math.pi * 0.25) / 2.0)
        expected = -math.cos(phase + (0.5 * math.sin(phase)))
        self.assertAlmostEqual(first_value, expected, places=9)

    async def test_fadein_function_uses_cycle_phase(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g2h",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="fadein(0.4)",
                    max_t=10.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 0.1, ts_ms=0, edge=None, ctx_id=None)
        out_a = await runtime.compute_output("value", ctx_id=40)
        self.assertAlmostEqual(out_a, 0.25, places=9)

        buffer_input(bus, "w1", "t", 0.6, ts_ms=0, edge=None, ctx_id=None)
        out_b = await runtime.compute_output("value", ctx_id=41)
        self.assertAlmostEqual(out_b, 1.0, places=9)

        buffer_input(bus, "w1", "t", 1.1, ts_ms=0, edge=None, ctx_id=None)
        out_c = await runtime.compute_output("value", ctx_id=42)
        self.assertAlmostEqual(out_c, 1.0, places=9)

    async def test_fadeout_function_uses_cycle_phase(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g2i",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="fadeout(0.4)",
                    max_t=10.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 9.9, ts_ms=0, edge=None, ctx_id=None)
        out_a = await runtime.compute_output("value", ctx_id=43)
        self.assertAlmostEqual(out_a, 0.25, places=9)

        buffer_input(bus, "w1", "t", 9.5, ts_ms=0, edge=None, ctx_id=None)
        out_b = await runtime.compute_output("value", ctx_id=44)
        self.assertAlmostEqual(out_b, 1.0, places=9)

        buffer_input(bus, "w1", "t", 8.5, ts_ms=0, edge=None, ctx_id=None)
        out_c = await runtime.compute_output("value", ctx_id=45)
        self.assertAlmostEqual(out_c, 1.0, places=9)

    async def test_numeric_state_defaults_feed_express_and_preview(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g2g",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="tempest(t, 0, c)",
                    max_t=4.0,
                    extra_state_fields=[
                        F8StateSpec(
                            name="c",
                            label="c",
                            description="",
                            valueSchema=number_schema(default=0.25),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        )
                    ],
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 0.0, ts_ms=0, edge=None, ctx_id=None)
        _ = await runtime.compute_output("value", ctx_id=37)

        express = str(await runtime.get_state_value("express") or "")
        preview = list(await runtime.get_state_value("preview") or [])
        last_error = _monitor_error_message(bus)

        self.assertIn("0.25", express)
        self.assertGreaterEqual(len(preview), 64)
        self.assertEqual(last_error, "")

    async def test_invalid_template_keeps_previous_model_and_value(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g3",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="a + t",
                    max_t=10.0,
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

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 2.0, ts_ms=0, edge=None, ctx_id=None)
        value_before = await runtime.compute_output("value", ctx_id=1)
        self.assertEqual(value_before, 3.0)

        express_before = str(await runtime.get_state_value("express") or "")
        preview_before = list(await runtime.get_state_value("preview") or [])

        await runtime.on_state("template", "cos((t)", ts_ms=2)

        express_after = str(await runtime.get_state_value("express") or "")
        preview_after = list(await runtime.get_state_value("preview") or [])
        last_error = _monitor_error_message(bus)

        self.assertEqual(express_after, express_before)
        self.assertEqual(preview_after, preview_before)
        self.assertNotEqual(last_error, "")

        buffer_input(bus, "w1", "t", 3.0, ts_ms=0, edge=None, ctx_id=None)
        value_after = await runtime.compute_output("value", ctx_id=2)
        self.assertEqual(value_after, 4.0)

    async def test_only_numeric_dynamic_state_fields_participate_as_variables(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g4",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="a + t",
                    max_t=10.0,
                    extra_state_fields=[
                        F8StateSpec(
                            name="a",
                            label="a",
                            description="",
                            valueSchema=number_schema(default=2.0),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        ),
                        F8StateSpec(
                            name="noise",
                            label="noise",
                            description="",
                            valueSchema=string_schema(default="x"),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        ),
                        F8StateSpec(
                            name="payload",
                            label="payload",
                            description="",
                            valueSchema=any_schema(),
                            access=F8StateAccess.rw,
                            valueRequired=False,
                            showOnNode=True,
                        ),
                    ],
                    extra_state_values={"a": 2.0, "noise": "x", "payload": {"k": 1}},
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 1.0, ts_ms=0, edge=None, ctx_id=None)
        output_before = await runtime.compute_output("value", ctx_id=4)
        self.assertEqual(output_before, 3.0)

        express_valid = str(await runtime.get_state_value("express") or "")
        await runtime.on_state("template", "a + noise", ts_ms=3)

        express_after_invalid = str(await runtime.get_state_value("express") or "")
        last_error = _monitor_error_message(bus)

        self.assertEqual(express_after_invalid, express_valid)
        self.assertNotEqual(last_error, "")

        output = await runtime.compute_output("value", ctx_id=5)
        self.assertEqual(output, 3.0)

    async def test_preview_uses_zero_to_max_t_domain(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g5",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="cond(t > 5.0, 1, 2)",
                    max_t=10.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        buffer_input(bus, "w1", "t", 0.2, ts_ms=0, edge=None, ctx_id=None)
        _ = await runtime.compute_output("value", ctx_id=6)

        preview = list(await runtime.get_state_value("preview") or [])
        self.assertGreaterEqual(len(preview), 64)

        low_count = sum(1 for point in preview if float(point[1]) == 2.0)
        high_count = sum(1 for point in preview if float(point[1]) == 1.0)
        total = len(preview)

        self.assertAlmostEqual(float(preview[0][0]), 0.0)
        self.assertLess(float(preview[-1][0]), 10.0)

        self.assertGreaterEqual(low_count, int(total * 0.45))
        self.assertGreaterEqual(high_count, int(total * 0.45))

    async def test_min_and_max_value_allow_auto_zoom_sentinel(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g6",
            revision="r1",
            nodes=[
                _build_wave_expr_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    template="t",
                    max_t=10.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WaveExprRuntimeNode)

        min_value = await runtime.validate_state("minValue", 0.0, ts_ms=7, meta={})
        max_value = await runtime.validate_state("maxValue", 0.0, ts_ms=8, meta={})

        self.assertEqual(min_value, 0.0)
        self.assertEqual(max_value, 0.0)


if __name__ == "__main__":
    unittest.main()
