import os
import sys
import unittest
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import buffer_input  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.envelope import EnvelopeRuntimeNode, register_operator  # noqa: E402


class EnvelopeNodeTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, Any] | None = None) -> tuple[ServiceBusHarness, Any, EnvelopeRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="env1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=EnvelopeRuntimeNode.SPEC.operatorClass,
            stateFields=list(EnvelopeRuntimeNode.SPEC.stateFields or []),
            stateValues=dict(state_values or {}),
            dataInPorts=list(EnvelopeRuntimeNode.SPEC.dataInPorts or []),
            dataOutPorts=list(EnvelopeRuntimeNode.SPEC.dataOutPorts or []),
        )
        graph = F8RuntimeGraph(graphId="g_env", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)

        node = bus.get_node("env1")
        self.assertIsInstance(node, EnvelopeRuntimeNode)
        assert isinstance(node, EnvelopeRuntimeNode)
        return harness, bus, node

    async def _step(self, bus: Any, node: EnvelopeRuntimeNode, *, value: float, idx: int, port: str = "normalized") -> Any:
        buffer_input(bus, "env1", "value", value, ts_ms=idx, edge=None, ctx_id=None)
        return await node.compute_output(port, ctx_id=idx)

    async def test_outputs_exist_and_normalized_is_bounded(self) -> None:
        _harness, bus, node = await self._build_node(
            state_values={
                "method": "EMA",
                "rise_alpha": 0.4,
                "fall_alpha": 0.05,
                "min_span": 0.25,
                "sma_window": 10,
                "margin": 0.0,
            }
        )

        normalized = await self._step(bus, node, value=0.2, idx=1, port="normalized")
        lower = await node.compute_output("lower", ctx_id=1)
        upper = await node.compute_output("upper", ctx_id=1)
        self.assertIsNotNone(normalized)
        self.assertIsNotNone(lower)
        self.assertIsNotNone(upper)
        self.assertGreaterEqual(float(normalized), 0.0)
        self.assertLessEqual(float(normalized), 1.0)

    async def test_single_outlier_does_not_trigger_jump(self) -> None:
        _harness, bus, node = await self._build_node(
            state_values={
                "jumpEnabled": True,
                "jumpSpanMult": 2.0,
                "jumpConsecutiveFrames": 3,
                "jumpReseedFrames": 6,
            }
        )

        for idx in range(1, 50):
            value = 0.0 if idx % 2 == 0 else 1.0
            await self._step(bus, node, value=value, idx=idx)

        await self._step(bus, node, value=10.0, idx=100)
        await self._step(bus, node, value=0.0, idx=101)

        self.assertEqual(node._jump_count, 0)
        self.assertEqual(node._far_count, 0)

    async def test_consecutive_outliers_trigger_jump(self) -> None:
        _harness, bus, node = await self._build_node(
            state_values={
                "jumpEnabled": True,
                "jumpSpanMult": 2.0,
                "jumpConsecutiveFrames": 3,
                "jumpReseedFrames": 6,
            }
        )

        for idx in range(1, 50):
            value = 0.0 if idx % 2 == 0 else 1.0
            await self._step(bus, node, value=value, idx=idx)

        await self._step(bus, node, value=10.0, idx=200)
        await self._step(bus, node, value=10.0, idx=201)
        await self._step(bus, node, value=10.0, idx=202)

        self.assertEqual(node._jump_count, 1)
        self.assertIsNotNone(node._last_jump_ts_ms)

    async def test_reseed_blend_moves_toward_new_mode(self) -> None:
        _harness, bus, node = await self._build_node(
            state_values={
                "jumpEnabled": True,
                "jumpSpanMult": 1.5,
                "jumpConsecutiveFrames": 2,
                "jumpReseedFrames": 4,
            }
        )

        for idx in range(1, 80):
            value = 0.0 if idx % 2 == 0 else 1.0
            await self._step(bus, node, value=value, idx=idx)

        pre_norm = float(await self._step(bus, node, value=0.0, idx=90))
        self.assertLess(pre_norm, 0.45)

        reseed_outputs: list[float] = []
        for idx in range(100, 105):
            reseed_outputs.append(float(await self._step(bus, node, value=10.0, idx=idx)))

        distances = [abs(v - 0.5) for v in reseed_outputs]
        self.assertGreater(distances[0], distances[-1])
        for i in range(len(distances) - 1):
            self.assertGreaterEqual(distances[i], distances[i + 1] - 1e-6)
        self.assertFalse(node._in_reseed)

    async def test_runtime_state_updates_take_effect(self) -> None:
        _harness, bus, node = await self._build_node()

        await node.on_state("jumpSpanMult", 2.5)
        await node.on_state("jumpConsecutiveFrames", 5)
        await node.on_state("jumpReseedFrames", 9)

        self.assertEqual(node._jump_span_mult, 2.5)
        self.assertEqual(node._jump_consecutive_frames, 5)
        self.assertEqual(node._jump_reseed_frames, 9)

        normalized = await self._step(bus, node, value=0.25, idx=10, port="normalized")
        self.assertIsNotNone(normalized)


if __name__ == "__main__":
    unittest.main()
