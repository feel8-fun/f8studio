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
from f8pyengine.operators.detrend import DetrendRuntimeNode, register_operator  # noqa: E402


class DetrendNodeTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, Any] | None = None) -> tuple[Any, DetrendRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="detrend1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=DetrendRuntimeNode.SPEC.operatorClass,
            stateFields=list(DetrendRuntimeNode.SPEC.stateFields or []),
            stateValues=dict(state_values or {}),
            dataInPorts=list(DetrendRuntimeNode.SPEC.dataInPorts or []),
            dataOutPorts=list(DetrendRuntimeNode.SPEC.dataOutPorts or []),
        )
        graph = F8RuntimeGraph(graphId="g_detrend", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("detrend1")
        self.assertIsInstance(node, DetrendRuntimeNode)
        assert isinstance(node, DetrendRuntimeNode)
        return bus, node

    async def _step(self, bus: Any, node: DetrendRuntimeNode, *, value: Any, idx: int) -> Any:
        buffer_input(bus, "detrend1", "value", value, ts_ms=idx, edge=None, ctx_id=None)
        return await node.compute_output("value", ctx_id=idx)

    async def test_constant_mode_converges_to_zero_for_constant_signal(self) -> None:
        bus, node = await self._build_node(state_values={"mode": "CONSTANT", "alpha": 0.1})
        output = 0.0
        for idx in range(1, 81):
            output = float(await self._step(bus, node, value=10.0, idx=idx))
        self.assertLess(abs(output), 0.01)

    async def test_linear_mode_reduces_ramp_more_than_constant_mode(self) -> None:
        bus_constant, constant_node = await self._build_node(state_values={"mode": "CONSTANT", "alpha": 0.08})
        bus_linear, linear_node = await self._build_node(state_values={"mode": "LINEAR", "alpha": 0.08})

        constant_outputs: list[float] = []
        linear_outputs: list[float] = []
        for idx in range(1, 201):
            value = 0.2 * idx
            constant_outputs.append(float(await self._step(bus_constant, constant_node, value=value, idx=idx)))
            linear_outputs.append(float(await self._step(bus_linear, linear_node, value=value, idx=idx)))

        constant_tail = sum(abs(value) for value in constant_outputs[-40:]) / 40.0
        linear_tail = sum(abs(value) for value in linear_outputs[-40:]) / 40.0
        self.assertLess(linear_tail, constant_tail * 0.35)

    async def test_vector_input_and_state_update_work(self) -> None:
        bus, node = await self._build_node(state_values={"mode": "CONSTANT", "alpha": 0.2})
        out1 = await self._step(bus, node, value=[4.0, -4.0], idx=1)
        self.assertIsInstance(out1, list)
        await node.on_state("mode", "LINEAR")
        await node.on_state("alpha", 0.4)
        self.assertEqual(node._mode, "LINEAR")
        self.assertAlmostEqual(node._alpha, 0.4)
        out2 = await self._step(bus, node, value=[4.0, -4.0], idx=2)
        self.assertIsInstance(out2, list)
        assert isinstance(out2, list)
        self.assertEqual(len(out2), 2)

    async def test_invalid_input_keeps_last_output(self) -> None:
        bus, node = await self._build_node()
        valid = await self._step(bus, node, value=3.0, idx=1)
        invalid = await self._step(bus, node, value={"bad": True}, idx=2)
        self.assertEqual(invalid, valid)


if __name__ == "__main__":
    unittest.main()
