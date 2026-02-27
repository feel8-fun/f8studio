import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk import F8DataPortSpec, F8RuntimeGraph, F8RuntimeNode, any_schema  # noqa: E402
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry  # noqa: E402
from f8pysdk.service_host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.service_bus.routing_data import buffer_input  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.playback_sync import PlaybackSyncRuntimeNode, register_operator  # noqa: E402


class PlaybackSyncTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, object] | None = None) -> tuple[ServiceBusHarness, object, PlaybackSyncRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="pb1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=PlaybackSyncRuntimeNode.SPEC.operatorClass,
            stateFields=list(PlaybackSyncRuntimeNode.SPEC.stateFields or []),
            stateValues=dict(state_values or {}),
            execInPorts=["exec"],
            execOutPorts=["exec"],
            dataInPorts=[F8DataPortSpec(name="playback", description="", valueSchema=any_schema(), required=False)],
            dataOutPorts=list(PlaybackSyncRuntimeNode.SPEC.dataOutPorts or []),
        )
        await bus.set_rungraph(F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[]))
        node = bus.get_node("pb1")
        self.assertIsInstance(node, PlaybackSyncRuntimeNode)
        assert isinstance(node, PlaybackSyncRuntimeNode)
        return harness, bus, node

    async def test_extrapolates_position_when_playing(self) -> None:
        _, bus, node = await self._build_node(state_values={"maxExtrapolateMs": 3000})
        now_s = 100.0

        with patch("f8pyengine.operators.playback_sync.time.monotonic", side_effect=lambda: now_s):
            buffer_input(
                bus,
                "pb1",
                "playback",
                {"videoId": "v1", "position": 10.0, "duration": 99.0, "playing": True},
                ts_ms=0,
                edge=None,
                ctx_id=None,
            )
            out0 = await node.compute_output("position", ctx_id=1)
            self.assertAlmostEqual(float(out0), 10.0, places=6)

            now_s = 100.25
            out1 = await node.compute_output("position", ctx_id=2)
            self.assertAlmostEqual(float(out1), 10.25, places=6)

            age_ms = await node.compute_output("ageMs", ctx_id=2)
            self.assertEqual(int(age_ms), 250)

    async def test_paused_payload_does_not_extrapolate(self) -> None:
        _, bus, node = await self._build_node()
        now_s = 12.0

        with patch("f8pyengine.operators.playback_sync.time.monotonic", side_effect=lambda: now_s):
            buffer_input(
                bus,
                "pb1",
                "playback",
                {"videoId": "v1", "position": 42.5, "duration": 99.0, "playing": False},
                ts_ms=0,
                edge=None,
                ctx_id=None,
            )
            out0 = await node.compute_output("position", ctx_id=11)
            self.assertAlmostEqual(float(out0), 42.5, places=6)

            now_s = 20.0
            out1 = await node.compute_output("position", ctx_id=12)
            self.assertAlmostEqual(float(out1), 42.5, places=6)

    async def test_marks_stale_and_caps_extrapolation(self) -> None:
        _, bus, node = await self._build_node(state_values={"maxExtrapolateMs": 500})
        now_s = 7.0

        with patch("f8pyengine.operators.playback_sync.time.monotonic", side_effect=lambda: now_s):
            buffer_input(
                bus,
                "pb1",
                "playback",
                {"videoId": "v1", "position": 1.0, "duration": 100.0, "playing": True},
                ts_ms=0,
                edge=None,
                ctx_id=None,
            )
            _ = await node.compute_output("position", ctx_id=21)

            now_s = 9.0
            out = await node.compute_output("position", ctx_id=22)
            stale = await node.compute_output("stale", ctx_id=22)
            self.assertAlmostEqual(float(out), 1.5, places=6)
            self.assertTrue(bool(stale))

    async def test_exec_passthrough(self) -> None:
        _, _, node = await self._build_node()
        ports = await node.on_exec(100, "exec")
        self.assertEqual(ports, ["exec"])


if __name__ == "__main__":
    unittest.main()
