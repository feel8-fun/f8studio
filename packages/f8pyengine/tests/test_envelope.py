import math
import os
import random
import sys
import unittest
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.generated import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry  # noqa: E402
from f8pysdk.service_host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.service_bus.routing_data import buffer_input  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.envelope import EnvelopeRuntimeNode, register_operator  # noqa: E402


class EnvelopeNodeTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, Any] | None = None) -> tuple[ServiceBusHarness, Any, EnvelopeRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
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

    async def test_compatibility_and_confidence_port(self) -> None:
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
        confidence = await node.compute_output("confidence", ctx_id=1)

        self.assertIsNotNone(normalized)
        self.assertIsNotNone(lower)
        self.assertIsNotNone(upper)
        self.assertIsNotNone(confidence)
        self.assertGreaterEqual(float(normalized), 0.0)
        self.assertLessEqual(float(normalized), 1.0)
        self.assertGreaterEqual(float(confidence), 0.0)
        self.assertLessEqual(float(confidence), 1.0)

    async def test_single_outlier_does_not_trigger_jump(self) -> None:
        _harness, bus, node = await self._build_node(
            state_values={
                "jumpEnabled": True,
                "jumpSpanMult": 2.0,
                "jumpConsecutiveFrames": 3,
                "jumpReseedFrames": 6,
                "confidenceEnabled": False,
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
                "confidenceEnabled": False,
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
                "confidenceEnabled": False,
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

    async def test_confidence_high_for_periodic_low_for_noise(self) -> None:
        _h1, bus1, node1 = await self._build_node(
            state_values={
                "jumpEnabled": False,
                "confidenceEnabled": True,
                "confidenceWindow": 128,
                "confidenceMinLag": 2,
                "confidenceMaxLag": 32,
                "confidencePeakProminence": 0.05,
                "confidenceSmoothingAlpha": 0.25,
            }
        )

        sine_conf = 0.0
        for idx in range(1, 360):
            value = math.sin(2.0 * math.pi * (idx / 12.0))
            sine_conf = float(await self._step(bus1, node1, value=value, idx=idx, port="confidence"))

        _h2, bus2, node2 = await self._build_node(
            state_values={
                "jumpEnabled": False,
                "confidenceEnabled": True,
                "confidenceWindow": 128,
                "confidenceMinLag": 2,
                "confidenceMaxLag": 32,
                "confidencePeakProminence": 0.05,
                "confidenceSmoothingAlpha": 0.25,
            }
        )

        rng = random.Random(1234)
        noise_conf = 0.0
        for idx in range(1, 360):
            value = rng.uniform(-1.0, 1.0)
            noise_conf = float(await self._step(bus2, node2, value=value, idx=idx, port="confidence"))

        self.assertGreater(sine_conf, 0.6)
        self.assertLess(noise_conf, 0.3)

    async def test_jump_reset_confidence_true_vs_false(self) -> None:
        async def _prepare_node(reset_conf: bool) -> tuple[Any, EnvelopeRuntimeNode]:
            _h, bus, node = await self._build_node(
                state_values={
                    "jumpEnabled": True,
                    "jumpSpanMult": 4.0,
                    "jumpConsecutiveFrames": 2,
                    "jumpReseedFrames": 4,
                    "jumpResetConfidence": reset_conf,
                    "confidenceEnabled": True,
                    "confidenceWindow": 96,
                    "confidenceMinLag": 2,
                    "confidenceMaxLag": 24,
                    "confidencePeakProminence": 0.05,
                }
            )

            for idx in range(1, 260):
                value = math.sin(2.0 * math.pi * (idx / 10.0))
                await self._step(bus, node, value=value, idx=idx, port="confidence")
            return bus, node

        bus_t, node_t = await _prepare_node(True)
        pre_t = float(await node_t.compute_output("confidence", ctx_id=261))
        await self._step(bus_t, node_t, value=12.0, idx=300, port="confidence")
        post_t = float(await self._step(bus_t, node_t, value=12.0, idx=301, port="confidence"))

        bus_f, node_f = await _prepare_node(False)
        pre_f = float(await node_f.compute_output("confidence", ctx_id=261))
        await self._step(bus_f, node_f, value=12.0, idx=300, port="confidence")
        post_f = float(await self._step(bus_f, node_f, value=12.0, idx=301, port="confidence"))

        self.assertGreater(pre_t, 0.5)
        self.assertGreater(pre_f, 0.5)
        self.assertLess(post_t, pre_t * 0.5)
        self.assertGreater(post_f, post_t + 0.1)

    async def test_runtime_state_updates_take_effect(self) -> None:
        _harness, bus, node = await self._build_node()

        await node.on_state("jumpSpanMult", 2.5)
        await node.on_state("jumpConsecutiveFrames", 5)
        await node.on_state("jumpReseedFrames", 9)
        await node.on_state("confidenceWindow", 140)
        await node.on_state("confidenceSmoothingAlpha", 0.3)
        await node.on_state("confidenceEnabled", True)

        self.assertEqual(node._jump_span_mult, 2.5)
        self.assertEqual(node._jump_consecutive_frames, 5)
        self.assertEqual(node._jump_reseed_frames, 9)
        self.assertEqual(node._confidence_window, 140)
        self.assertEqual(node._confidence_smoothing_alpha, 0.3)
        self.assertTrue(node._confidence_enabled)

        normalized = await self._step(bus, node, value=0.25, idx=10, port="normalized")
        confidence = await node.compute_output("confidence", ctx_id=10)
        self.assertIsNotNone(normalized)
        self.assertIsNotNone(confidence)
        self.assertGreaterEqual(float(confidence), 0.0)
        self.assertLessEqual(float(confidence), 1.0)


if __name__ == "__main__":
    unittest.main()
