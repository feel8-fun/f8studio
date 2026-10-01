import math
import os
import sys
import unittest
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.testing import buffer_input  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.bandpass_filter import BandpassFilterRuntimeNode, register_operator as register_bandpass  # noqa: E402
from f8pyengine.operators.highpass_filter import HighpassFilterRuntimeNode, register_operator as register_highpass  # noqa: E402
from f8pyengine.operators.lowpass_filter import LowpassFilterRuntimeNode, register_operator as register_lowpass  # noqa: E402


def _mean_abs(values: list[float]) -> float:
    return sum(abs(value) for value in values) / max(1, len(values))


class SignalFilterNodeTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, runtime_cls: type[Any], register_fn: Any, *, state_values: dict[str, Any] | None = None) -> tuple[Any, Any]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_fn(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="filter1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=runtime_cls.SPEC.operatorClass,
            stateFields=list(runtime_cls.SPEC.stateFields or []),
            stateValues=dict(state_values or {}),
            dataInPorts=list(runtime_cls.SPEC.dataInPorts or []),
            dataOutPorts=list(runtime_cls.SPEC.dataOutPorts or []),
        )
        graph = F8RuntimeGraph(graphId="g_filter", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("filter1")
        self.assertIsInstance(node, runtime_cls)
        return bus, node

    async def _step(self, bus: Any, node: Any, *, value: Any, idx: int) -> Any:
        buffer_input(bus, "filter1", "value", value, ts_ms=idx, edge=None, ctx_id=None)
        return await node.compute_output("value", ctx_id=idx)

    async def test_lowpass_preserves_low_frequency_and_reacts_to_state_change(self) -> None:
        state_values = {"sampleIntervalMs": 1000.0 / 120.0, "cutoff": 4.0, "order": 2}
        bus_low, node_low = await self._build_node(LowpassFilterRuntimeNode, register_lowpass, state_values=state_values)
        bus_high, node_high = await self._build_node(LowpassFilterRuntimeNode, register_lowpass, state_values=state_values)

        low_outputs: list[float] = []
        high_outputs: list[float] = []
        for idx in range(1, 361):
            low_value = math.sin(2.0 * math.pi * 1.0 * idx / 120.0)
            high_value = math.sin(2.0 * math.pi * 18.0 * idx / 120.0)
            low_outputs.append(float(await self._step(bus_low, node_low, value=low_value, idx=idx)))
            high_outputs.append(float(await self._step(bus_high, node_high, value=high_value, idx=idx)))

        self.assertGreater(_mean_abs(low_outputs[-120:]), 0.45)
        self.assertLess(_mean_abs(high_outputs[-120:]), 0.2)

        await node_low.on_state("cutoff", 12.0)
        self.assertAlmostEqual(node_low._cutoff, 12.0)
        shifted_outputs: list[float] = []
        for idx in range(361, 481):
            high_value = math.sin(2.0 * math.pi * 10.0 * idx / 120.0)
            shifted_outputs.append(float(await self._step(bus_low, node_low, value=high_value, idx=idx)))
        self.assertGreater(_mean_abs(shifted_outputs[-60:]), 0.2)

    async def test_highpass_removes_dc_and_keeps_high_frequency(self) -> None:
        state_values = {"sampleIntervalMs": 1000.0 / 120.0, "cutoff": 4.0, "order": 2}
        bus_constant, node_constant = await self._build_node(
            HighpassFilterRuntimeNode, register_highpass, state_values=state_values
        )
        bus_high, node_high = await self._build_node(HighpassFilterRuntimeNode, register_highpass, state_values=state_values)

        constant_outputs: list[float] = []
        high_outputs: list[float] = []
        for idx in range(1, 361):
            constant_outputs.append(float(await self._step(bus_constant, node_constant, value=3.0, idx=idx)))
            high_value = math.sin(2.0 * math.pi * 18.0 * idx / 120.0)
            high_outputs.append(float(await self._step(bus_high, node_high, value=high_value, idx=idx)))

        self.assertLess(_mean_abs(constant_outputs[-120:]), 0.02)
        self.assertGreater(_mean_abs(high_outputs[-120:]), 0.35)

    async def test_bandpass_keeps_passband_and_handles_invalid_range(self) -> None:
        state_values = {"sampleIntervalMs": 1000.0 / 120.0, "low_cutoff": 4.0, "high_cutoff": 10.0, "order": 2}
        bus_pass, node_pass = await self._build_node(BandpassFilterRuntimeNode, register_bandpass, state_values=state_values)
        bus_stop, node_stop = await self._build_node(BandpassFilterRuntimeNode, register_bandpass, state_values=state_values)

        pass_outputs: list[float] = []
        stop_outputs: list[float] = []
        for idx in range(1, 361):
            pass_value = math.sin(2.0 * math.pi * 6.0 * idx / 120.0)
            stop_value = math.sin(2.0 * math.pi * 1.0 * idx / 120.0)
            pass_outputs.append(float(await self._step(bus_pass, node_pass, value=pass_value, idx=idx)))
            stop_outputs.append(float(await self._step(bus_stop, node_stop, value=stop_value, idx=idx)))

        self.assertGreater(_mean_abs(pass_outputs[-120:]), 0.35)
        self.assertLess(_mean_abs(stop_outputs[-120:]), 0.2)

        previous = await self._step(bus_pass, node_pass, value=0.5, idx=500)
        await node_pass.on_state("low_cutoff", 20.0)
        await node_pass.on_state("high_cutoff", 10.0)
        invalid = await self._step(bus_pass, node_pass, value=0.75, idx=501)
        self.assertEqual(invalid, previous)

    async def test_filters_rebuild_on_dimension_change(self) -> None:
        bus, node = await self._build_node(
            LowpassFilterRuntimeNode,
            register_lowpass,
            state_values={"sampleIntervalMs": 1000.0 / 120.0, "cutoff": 4.0, "order": 2},
        )
        scalar = await self._step(bus, node, value=1.0, idx=1)
        vector = await self._step(bus, node, value=[1.0, -1.0], idx=2)
        self.assertIsInstance(scalar, float)
        self.assertIsInstance(vector, list)
        assert isinstance(vector, list)
        self.assertEqual(len(vector), 2)


if __name__ == "__main__":
    unittest.main()
