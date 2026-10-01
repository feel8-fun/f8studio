from __future__ import annotations

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
from f8pyengine.operators.wave_funscript import WaveFunscriptRuntimeNode  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402

_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_wave.funscript')


def _build_wave_funscript_runtime_node(
    *,
    node_id: str,
    service_id: str,
    funscript_path: str,
    selected_axis: str | None = None,
    max_t: float | None = None,
    interp: str | None = None,
) -> F8RuntimeNode:
    state_values: dict[str, object] = {"funscriptPath": funscript_path}
    if selected_axis is not None:
        state_values["selectedAxis"] = selected_axis
    if max_t is not None:
        state_values["maxT"] = max_t
    if interp is not None:
        state_values["interp"] = interp
    return F8RuntimeNode(
        nodeId=node_id,
        serviceId=service_id,
        serviceClass=SERVICE_CLASS,
        operatorClass=WaveFunscriptRuntimeNode.SPEC.operatorClass,
        stateFields=list(WaveFunscriptRuntimeNode.SPEC.stateFields or []),
        stateValues=state_values,
        dataInPorts=list(WaveFunscriptRuntimeNode.SPEC.dataInPorts or []),
        dataOutPorts=list(WaveFunscriptRuntimeNode.SPEC.dataOutPorts or []),
    )


class WaveFunscriptNodeTests(unittest.IsolatedAsyncioTestCase):
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
        self.assertIn("f8.wave_funscript", operator_classes)

    async def test_loads_fixture_and_defaults_to_toplevel(self) -> None:
        bus = self._setup_bus(service_id='svcA')
        graph = F8RuntimeGraph(
            graphId='g1',
            revision='r1',
            nodes=[_build_wave_funscript_runtime_node(node_id='w1', service_id='svcA', funscript_path=_FIXTURE_PATH)],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node('w1')
        assert isinstance(runtime, WaveFunscriptRuntimeNode)

        self.assertEqual(runtime._all_axes, ['TopLevel', 'L1', 'R1'])
        self.assertEqual(runtime._selected_axis, 'TopLevel')
        self.assertEqual(runtime._state_values['maxT'], 12.0)
        self.assertEqual(runtime._points[0], (0.0, 0.1))
        self.assertEqual(runtime._points[1], (4.0, 0.9))

    async def test_defaults_to_linear_interp(self) -> None:
        bus = self._setup_bus(service_id='svcInterpDefault')
        graph = F8RuntimeGraph(
            graphId='g_interp_default',
            revision='r1',
            nodes=[_build_wave_funscript_runtime_node(node_id='w1', service_id='svcInterpDefault', funscript_path=_FIXTURE_PATH)],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node('w1')
        assert isinstance(runtime, WaveFunscriptRuntimeNode)

        self.assertEqual(runtime._interp, 'linear')
        self.assertTrue(runtime._use_linear_sampler_for_output)

    async def test_selected_axis_switches_points(self) -> None:
        bus = self._setup_bus(service_id='svcB')
        graph = F8RuntimeGraph(
            graphId='g2',
            revision='r1',
            nodes=[_build_wave_funscript_runtime_node(node_id='w1', service_id='svcB', funscript_path=_FIXTURE_PATH)],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node('w1')
        assert isinstance(runtime, WaveFunscriptRuntimeNode)

        await runtime.on_state('selectedAxis', 'L1', ts_ms=1)
        self.assertEqual(runtime._selected_axis, 'L1')
        self.assertEqual(runtime._points, [(0.0, 0.0), (3.0, 1.0), (6.0, 0.5), (12.0, 0.0)])

    async def test_manual_max_t_overrides_file_duration(self) -> None:
        bus = self._setup_bus(service_id='svcC')
        graph = F8RuntimeGraph(
            graphId='g3',
            revision='r1',
            nodes=[_build_wave_funscript_runtime_node(node_id='w1', service_id='svcC', funscript_path=_FIXTURE_PATH)],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node('w1')
        assert isinstance(runtime, WaveFunscriptRuntimeNode)

        await runtime.on_state('maxT', 6.0, ts_ms=2)
        self.assertEqual(runtime._max_t, 6.0)
        buffer_input(bus, 'w1', 't', 7.0, ts_ms=0, edge=None, ctx_id=None)
        out = await runtime.compute_output('value', ctx_id=1)
        self.assertAlmostEqual(float(out), 0.3, places=6)

    async def test_t_uses_seconds_and_loops_by_max_t(self) -> None:
        bus = self._setup_bus(service_id='svcD')
        graph = F8RuntimeGraph(
            graphId='g4',
            revision='r1',
            nodes=[_build_wave_funscript_runtime_node(node_id='w1', service_id='svcD', funscript_path=_FIXTURE_PATH)],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node('w1')
        assert isinstance(runtime, WaveFunscriptRuntimeNode)

        buffer_input(bus, 'w1', 't', 13.0, ts_ms=0, edge=None, ctx_id=None)
        out = await runtime.compute_output('value', ctx_id=2)
        self.assertAlmostEqual(float(out), 0.3, places=6)

    async def test_linear_sampler_caches_segment_index(self) -> None:
        bus = self._setup_bus(service_id='svcCache')
        graph = F8RuntimeGraph(
            graphId='g_cache',
            revision='r1',
            nodes=[_build_wave_funscript_runtime_node(node_id='w1', service_id='svcCache', funscript_path=_FIXTURE_PATH)],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node('w1')
        assert isinstance(runtime, WaveFunscriptRuntimeNode)

        buffer_input(bus, 'w1', 't', 1.0, ts_ms=0, edge=None, ctx_id=None)
        _ = await runtime.compute_output('value', ctx_id=10)
        first_index = runtime._linear_sampler.last_segment_index
        buffer_input(bus, 'w1', 't', 1.1, ts_ms=0, edge=None, ctx_id=None)
        _ = await runtime.compute_output('value', ctx_id=11)
        self.assertIsNotNone(first_index)
        self.assertEqual(runtime._linear_sampler.last_segment_index, first_index)

    async def test_smooth_interp_changes_output_but_not_heatmap_mode(self) -> None:
        bus = self._setup_bus(service_id='svcSmooth')
        graph = F8RuntimeGraph(
            graphId='g_smooth',
            revision='r1',
            nodes=[_build_wave_funscript_runtime_node(node_id='w1', service_id='svcSmooth', funscript_path=_FIXTURE_PATH, interp='pchip')],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node('w1')
        assert isinstance(runtime, WaveFunscriptRuntimeNode)

        self.assertEqual(runtime._interp, 'pchip')
        self.assertFalse(runtime._use_linear_sampler_for_output)
        heatmap_before = list(runtime._heatmap)

        buffer_input(bus, 'w1', 't', 1.0, ts_ms=0, edge=None, ctx_id=None)
        smooth_out = await runtime.compute_output('value', ctx_id=20)
        linear_out = runtime._linear_sampler.sample(1.0)

        self.assertNotAlmostEqual(float(smooth_out), float(linear_out), places=6)
        self.assertEqual(runtime._heatmap, heatmap_before)

    async def test_heatmap_is_normalized(self) -> None:
        bus = self._setup_bus(service_id='svcE')
        graph = F8RuntimeGraph(
            graphId='g5',
            revision='r1',
            nodes=[_build_wave_funscript_runtime_node(node_id='w1', service_id='svcE', funscript_path=_FIXTURE_PATH)],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node('w1')
        assert isinstance(runtime, WaveFunscriptRuntimeNode)

        self.assertEqual(len(runtime._heatmap), 128)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in runtime._heatmap))

    async def test_invalid_path_sets_last_error(self) -> None:
        bus = self._setup_bus(service_id='svcF')
        graph = F8RuntimeGraph(
            graphId='g6',
            revision='r1',
            nodes=[
                _build_wave_funscript_runtime_node(
                    node_id='w1',
                    service_id='svcF',
                    funscript_path=os.path.join(os.path.dirname(__file__), 'fixtures', 'missing.funscript'),
                )
            ],
            edges=[],
        )
        await bus.set_rungraph(graph)
        runtime = bus.get_node('w1')
        assert isinstance(runtime, WaveFunscriptRuntimeNode)

        self.assertIn('failed to load funscript', runtime._last_error)
        self.assertEqual(runtime._points, [])
