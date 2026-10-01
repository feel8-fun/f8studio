import math
import os
import random
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
from f8pyengine.operators.periodicity_detector import PeriodicityDetectorRuntimeNode, register_operator  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402


class PeriodicityDetectorNodeTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, Any] | None = None) -> tuple[Any, PeriodicityDetectorRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="period1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=PeriodicityDetectorRuntimeNode.SPEC.operatorClass,
            stateFields=list(PeriodicityDetectorRuntimeNode.SPEC.stateFields or []),
            stateValues=dict(state_values or {}),
            dataInPorts=list(PeriodicityDetectorRuntimeNode.SPEC.dataInPorts or []),
            dataOutPorts=list(PeriodicityDetectorRuntimeNode.SPEC.dataOutPorts or []),
        )
        graph = F8RuntimeGraph(graphId="g_period", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("period1")
        self.assertIsInstance(node, PeriodicityDetectorRuntimeNode)
        assert isinstance(node, PeriodicityDetectorRuntimeNode)
        return bus, node

    async def _step(self, bus: Any, node: PeriodicityDetectorRuntimeNode, *, value: Any, idx: int, port: str) -> Any:
        buffer_input(bus, "period1", "value", value, ts_ms=idx, edge=None, ctx_id=None)
        return await node.compute_output(port, ctx_id=idx)

    async def test_periodic_signal_gets_higher_confidence_than_noise(self) -> None:
        state_values = {
            "window": 150,
            "min_lag": 10,
            "max_lag": 150,
            "smoothing_alpha": 0.25,
            "threshold": 0.6,
            "sampleIntervalMs": 1000.0 / 30.0,
        }
        bus_p, node_p = await self._build_node(state_values=state_values)
        bus_n, node_n = await self._build_node(state_values=state_values)

        periodic_conf = 0.0
        noise_conf = 0.0
        rng = random.Random(1234)
        for idx in range(1, 301):
            periodic_value = math.sin(2.0 * math.pi * 1.0 * idx / 30.0)
            periodic_conf = float(await self._step(bus_p, node_p, value=periodic_value, idx=idx, port="confidence"))
            noise_value = rng.uniform(-1.0, 1.0)
            noise_conf = float(await self._step(bus_n, node_n, value=noise_value, idx=idx, port="confidence"))

        self.assertGreater(periodic_conf, 0.6)
        self.assertLess(noise_conf, 0.4)

    async def test_normalized_autocorrelation_reports_expected_hz(self) -> None:
        bus, node = await self._build_node(
            state_values={
                "window": 180,
                "min_lag": 8,
                "max_lag": 120,
                "smoothing_alpha": 1.0,
                "sampleIntervalMs": 1000.0 / 30.0,
                "threshold": 0.5,
            }
        )

        confidence = 0.0
        period_hz = 0.0
        for idx in range(1, 361):
            value = math.sin(2.0 * math.pi * 1.5 * idx / 30.0)
            confidence = float(await self._step(bus, node, value=value, idx=idx, port="confidence"))
            period_hz = float(await self._step(bus, node, value=value, idx=idx, port="period_hz"))

        self.assertGreater(confidence, 0.7)
        self.assertGreater(period_hz, 1.3)
        self.assertLess(period_hz, 1.7)

    async def test_periodic_energy_matches_rms_times_confidence(self) -> None:
        bus, node = await self._build_node(state_values={"window": 120, "min_lag": 8, "max_lag": 80, "rms_window": 32})
        periodic_energy = 0.0
        rms = 0.0
        confidence = 0.0
        for idx in range(1, 240):
            value = math.sin(2.0 * math.pi * 1.5 * idx / 30.0)
            rms = float(await self._step(bus, node, value=value, idx=idx, port="rms"))
            confidence = float(await self._step(bus, node, value=value, idx=idx, port="confidence"))
            periodic_energy = float(await self._step(bus, node, value=value, idx=idx, port="periodicEnergy"))

        self.assertAlmostEqual(periodic_energy, rms * confidence, places=6)

    async def test_period_ms_and_boolean_output(self) -> None:
        bus, node = await self._build_node(
            state_values={
                "window": 150,
                "min_lag": 10,
                "max_lag": 80,
                "threshold": 0.5,
                "sampleIntervalMs": 1000.0 / 30.0,
            }
        )
        period_ms = 0.0
        period_hz = 0.0
        is_periodic = False
        for idx in range(1, 260):
            value = math.sin(2.0 * math.pi * 1.0 * idx / 30.0)
            period_ms = float(await self._step(bus, node, value=value, idx=idx, port="periodMs"))
            period_hz = float(await self._step(bus, node, value=value, idx=idx, port="period_hz"))
            is_periodic = bool(await self._step(bus, node, value=value, idx=idx, port="is_periodic"))

        self.assertGreater(period_ms, 900.0)
        self.assertLess(period_ms, 1100.0)
        self.assertGreater(period_hz, 0.8)
        self.assertLess(period_hz, 1.2)
        self.assertTrue(is_periodic)

    def test_registered_in_pyengine_specs(self) -> None:
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        desc = reg.describe(SERVICE_CLASS)
        operator_classes = {str(spec.operatorClass or "") for spec in list(desc.operators or [])}
        self.assertIn("f8.periodicity_detector", operator_classes)


if __name__ == "__main__":
    unittest.main()
