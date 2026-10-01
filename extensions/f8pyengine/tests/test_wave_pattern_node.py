from __future__ import annotations

import math
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.testing import buffer_input  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.wave_pattern import WavePatternRuntimeNode  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402


def _build_wave_pattern_runtime_node(
    *,
    node_id: str,
    service_id: str,
    points: list[list[float]] | None = None,
    max_t: float = 10.0,
    interp: str = "pchip",
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> F8RuntimeNode:
    state_values: dict[str, object] = {
        "maxT": max_t,
        "interp": interp,
        "minValue": min_value,
        "maxValue": max_value,
    }
    if points is not None:
        state_values["points"] = points

    return F8RuntimeNode(
        nodeId=node_id,
        serviceId=service_id,
        serviceClass=SERVICE_CLASS,
        operatorClass=WavePatternRuntimeNode.SPEC.operatorClass,
        stateFields=list(WavePatternRuntimeNode.SPEC.stateFields or []),
        stateValues=state_values,
        dataInPorts=list(WavePatternRuntimeNode.SPEC.dataInPorts or []),
        dataOutPorts=list(WavePatternRuntimeNode.SPEC.dataOutPorts or []),
    )


class WavePatternNodeTests(unittest.IsolatedAsyncioTestCase):
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
        self.assertIn("f8.wave_pattern", operator_classes)

    async def test_default_points_publish_preview(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g1",
            revision="r1",
            nodes=[_build_wave_pattern_runtime_node(node_id="w1", service_id="svcA")],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WavePatternRuntimeNode)

        self.assertEqual(runtime._state_values["points"], [[0.0, 0.0], [10.0, 0.0]])
        self.assertGreaterEqual(len(runtime._preview_cycle), 32)

    async def test_wraps_t_by_max_t(self) -> None:
        bus = self._setup_bus(service_id="svcA")
        graph = F8RuntimeGraph(
            graphId="g2",
            revision="r1",
            nodes=[
                _build_wave_pattern_runtime_node(
                    node_id="w1",
                    service_id="svcA",
                    points=[[0.0, 0.0], [5.0, 1.0], [10.0, 0.0]],
                    max_t=10.0,
                    interp="linear",
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WavePatternRuntimeNode)

        buffer_input(bus, "w1", "t", 12.5, ts_ms=0, edge=None, ctx_id=None)
        out = await runtime.compute_output("value", ctx_id=22)
        self.assertAlmostEqual(float(out), 0.5, places=6)

    async def test_linear_output_uses_cached_sampler(self) -> None:
        bus = self._setup_bus(service_id='svcLinearCache')
        graph = F8RuntimeGraph(
            graphId='g_linear_cache',
            revision='r1',
            nodes=[
                _build_wave_pattern_runtime_node(
                    node_id='w1',
                    service_id='svcLinearCache',
                    points=[[0.0, 0.0], [5.0, 1.0], [10.0, 0.0]],
                    max_t=10.0,
                    interp='linear',
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node('w1')
        assert isinstance(runtime, WavePatternRuntimeNode)

        self.assertTrue(runtime._use_linear_sampler_for_output)
        buffer_input(bus, 'w1', 't', 1.0, ts_ms=0, edge=None, ctx_id=None)
        _ = await runtime.compute_output('value', ctx_id=100)
        first_index = runtime._linear_sampler.last_segment_index
        buffer_input(bus, 'w1', 't', 1.2, ts_ms=0, edge=None, ctx_id=None)
        _ = await runtime.compute_output('value', ctx_id=101)
        self.assertIsNotNone(first_index)
        self.assertEqual(runtime._linear_sampler.last_segment_index, first_index)

    async def test_all_interp_methods_compute(self) -> None:
        methods = ["linear", "pchip", "akima", "cubic_spline"]
        for index, method in enumerate(methods):
            bus = self._setup_bus(service_id=f"svc{index}")
            graph = F8RuntimeGraph(
                graphId=f"g_interp_{index}",
                revision="r1",
                nodes=[
                    _build_wave_pattern_runtime_node(
                        node_id="w1",
                        service_id=f"svc{index}",
                        points=[[1.0, 0.0], [4.0, 1.0], [7.0, 0.25]],
                        max_t=8.0,
                        interp=method,
                    )
                ],
                edges=[],
            )
            await bus.set_rungraph(graph)
            runtime = bus.get_node("w1")
            assert isinstance(runtime, WavePatternRuntimeNode)

            buffer_input(bus, "w1", "t", 2.0, ts_ms=0, edge=None, ctx_id=None)
            out = await runtime.compute_output("value", ctx_id=index)
            self.assertIsInstance(out, float)
            preview = await runtime.get_state_value("preview")
            self.assertGreater(len(preview or []), 10)

    async def test_does_not_clamp_runtime_output_to_preview_range(self) -> None:
        bus = self._setup_bus(service_id="svcB")
        graph = F8RuntimeGraph(
            graphId="g3",
            revision="r1",
            nodes=[
                _build_wave_pattern_runtime_node(
                    node_id="w1",
                    service_id="svcB",
                    points=[[0.0, 0.0], [5.0, 2.0], [10.0, 0.0]],
                    max_t=10.0,
                    interp="linear",
                    min_value=0.0,
                    max_value=1.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WavePatternRuntimeNode)

        buffer_input(bus, "w1", "t", 5.0, ts_ms=0, edge=None, ctx_id=None)
        out = await runtime.compute_output("value", ctx_id=1)
        self.assertEqual(float(out), 2.0)

    async def test_fallbacks_for_zero_one_and_two_points(self) -> None:
        cases = [
            ([], 2.0, 0.0),
            ([[2.0, 0.75]], 2.0, 0.75),
            ([[0.0, 0.0], [5.0, 1.0]], 2.5, 0.5),
        ]
        for index, (points, t_value, expected) in enumerate(cases):
            bus = self._setup_bus(service_id=f"fallback{index}")
            graph = F8RuntimeGraph(
                graphId=f"fallback_{index}",
                revision="r1",
                nodes=[
                    _build_wave_pattern_runtime_node(
                        node_id="w1",
                        service_id=f"fallback{index}",
                        points=points,
                        max_t=10.0,
                        interp="cubic_spline",
                    )
                ],
                edges=[],
            )
            await bus.set_rungraph(graph)
            runtime = bus.get_node("w1")
            assert isinstance(runtime, WavePatternRuntimeNode)

            buffer_input(bus, "w1", "t", t_value, ts_ms=0, edge=None, ctx_id=None)
            out = await runtime.compute_output("value", ctx_id=index)
            self.assertAlmostEqual(float(out), expected, places=6)

    async def test_preserves_raw_points_while_deduping_and_sorting(self) -> None:
        bus = self._setup_bus(service_id="svcC")
        graph = F8RuntimeGraph(
            graphId="g4",
            revision="r1",
            nodes=[
                _build_wave_pattern_runtime_node(
                    node_id="w1",
                    service_id="svcC",
                    points=[[9.0, 0.9], [-2.0, 0.1], [9.0, 0.5], [12.0, 0.3], [3.0, 0.4]],
                    max_t=10.0,
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)

        runtime = bus.get_node("w1")
        assert isinstance(runtime, WavePatternRuntimeNode)
        self.assertEqual(runtime._state_values["points"], [[-2.0, 0.1], [3.0, 0.4], [9.0, 0.5], [12.0, 0.3]])

    async def test_shrinking_max_t_preserves_hidden_points(self) -> None:
        bus = self._setup_bus(service_id="svcHidden")
        graph = F8RuntimeGraph(
            graphId="g_hidden",
            revision="r1",
            nodes=[
                _build_wave_pattern_runtime_node(
                    node_id="w1",
                    service_id="svcHidden",
                    points=[[1.0, 0.1], [6.0, 0.6], [12.0, 1.2]],
                    max_t=12.0,
                    interp="linear",
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node("w1")
        assert isinstance(runtime, WavePatternRuntimeNode)

        await runtime.on_state("maxT", 5.0, ts_ms=1)
        self.assertEqual(runtime._state_values["points"], [[1.0, 0.1], [6.0, 0.6], [12.0, 1.2]])

        buffer_input(bus, "w1", "t", 1.0, ts_ms=0, edge=None, ctx_id=None)
        out_small = await runtime.compute_output("value", ctx_id=1)
        self.assertAlmostEqual(float(out_small), 0.1, places=6)

        await runtime.on_state("maxT", 12.0, ts_ms=2)
        buffer_input(bus, "w1", "t", 6.0, ts_ms=0, edge=None, ctx_id=None)
        out_restored = await runtime.compute_output("value", ctx_id=2)
        self.assertAlmostEqual(float(out_restored), 0.6, places=6)

    async def test_periodic_extension_keeps_boundary_values_close(self) -> None:
        bus = self._setup_bus(service_id="svcD")
        graph = F8RuntimeGraph(
            graphId="g5",
            revision="r1",
            nodes=[
                _build_wave_pattern_runtime_node(
                    node_id="w1",
                    service_id="svcD",
                    points=[[2.0, 0.2], [5.0, 1.0], [8.0, 0.1]],
                    max_t=10.0,
                    interp="pchip",
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node("w1")
        assert isinstance(runtime, WavePatternRuntimeNode)

        buffer_input(bus, "w1", "t", 0.1, ts_ms=0, edge=None, ctx_id=None)
        left = await runtime.compute_output("value", ctx_id=1)
        buffer_input(bus, "w1", "t", 9.9, ts_ms=0, edge=None, ctx_id=None)
        right = await runtime.compute_output("value", ctx_id=2)
        self.assertTrue(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=0.06))
