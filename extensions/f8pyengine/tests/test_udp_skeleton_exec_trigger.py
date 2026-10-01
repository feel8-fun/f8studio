import asyncio
import json
import os
import socket
import sys
import unittest
from dataclasses import dataclass
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.skeleton_decoder import SkeletonDecoderRuntimeNode  # noqa: E402
from f8pyengine.operators.udp_in import UdpInRuntimeNode  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _skeleton_payload(*, frame_id: int, chunk_index: int = 0, chunk_count: int = 1, model_name: str = "Model_A") -> dict[str, Any]:
    suffix = f"{frame_id}_{chunk_index}"
    return {
        "type": "skeleton_binary",
        "modelName": model_name,
        "timestampMs": 1000 + int(frame_id),
        "schema": "f8.skeleton.v1",
        "boneCount": 1,
        "bones": [{"name": f"bone_{suffix}", "pos": [0, 0, 0], "rot": [1, 0, 0, 0]}],
        "trailer": {
            "magic": "LMEX",
            "extVersion": 1,
            "frameId": int(frame_id),
            "chunkIndex": int(chunk_index),
            "chunkCount": int(chunk_count),
            "totalBoneCount": int(chunk_count),
            "characterId": 1,
        },
    }


def _send_udp_json(*, port: int, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(data, ("127.0.0.1", int(port)))


@dataclass
class _FakeEntrypointContext:
    calls: list[tuple[str, str]]

    def __init__(self) -> None:
        self.calls = []

    async def emit_exec(self, out_port: str, *, exec_id: str | int) -> None:
        self.calls.append((str(out_port), str(exec_id)))


class SkeletonDecoderPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def _setup_udp_runtime(self, *, port: int) -> tuple[object, UdpInRuntimeNode, _FakeEntrypointContext]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        udp = F8RuntimeNode(
            nodeId="udp_in",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=UdpInRuntimeNode.SPEC.operatorClass,
            stateFields=list(UdpInRuntimeNode.SPEC.stateFields or []),
            stateValues={
                "bindAddress": "127.0.0.1",
                "port": int(port),
                "maxQueue": 1024,
                "reuseAddress": False,
            },
            execInPorts=list(UdpInRuntimeNode.SPEC.execInPorts or []),
            execOutPorts=list(UdpInRuntimeNode.SPEC.execOutPorts or []),
            dataOutPorts=list(UdpInRuntimeNode.SPEC.dataOutPorts or []),
        )
        graph = F8RuntimeGraph(graphId="g_udp_in", revision="r1", nodes=[udp], edges=[])
        await bus.set_rungraph(graph)  # type: ignore[attr-defined]

        node = bus.get_node("udp_in")
        self.assertIsInstance(node, UdpInRuntimeNode)
        assert isinstance(node, UdpInRuntimeNode)
        ctx = _FakeEntrypointContext()
        await node.start_entrypoint(ctx)  # type: ignore[arg-type]
        await node.on_lifecycle(True, {"case": "test"})

        runtime_ready = False
        for _ in range(100):
            if node._entrypoint_ctx is not None and node._drain_task is not None:
                runtime_ready = True
                break
            await asyncio.sleep(0.01)
        self.assertTrue(runtime_ready)
        return bus, node, ctx

    async def _new_decoder(self) -> SkeletonDecoderRuntimeNode:
        decoder_node = F8RuntimeNode(
            nodeId="decoder",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=SkeletonDecoderRuntimeNode.SPEC.operatorClass,
            stateFields=list(SkeletonDecoderRuntimeNode.SPEC.stateFields or []),
            stateValues={"cleanupAfterMs": 10000, "selectedKey": ""},
            execInPorts=list(SkeletonDecoderRuntimeNode.SPEC.execInPorts or []),
            execOutPorts=list(SkeletonDecoderRuntimeNode.SPEC.execOutPorts or []),
            dataInPorts=list(SkeletonDecoderRuntimeNode.SPEC.dataInPorts or []),
            dataOutPorts=list(SkeletonDecoderRuntimeNode.SPEC.dataOutPorts or []),
        )
        return SkeletonDecoderRuntimeNode(node_id="decoder", node=decoder_node, initial_state={"cleanupAfterMs": 10000})

    async def _wait_exec_calls(self, ctx: _FakeEntrypointContext, *, at_least: int, timeout_s: float = 1.2) -> None:
        end = asyncio.get_running_loop().time() + timeout_s
        while True:
            if len(ctx.calls) >= at_least:
                return
            if asyncio.get_running_loop().time() >= end:
                self.fail(f"timed out waiting for exec calls >= {at_least}")
            await asyncio.sleep(0.01)

    async def test_packet_commit_triggers_decoder_exec(self) -> None:
        port = _free_udp_port()
        bus, udp_node, ctx = await self._setup_udp_runtime(port=port)
        decoder = await self._new_decoder()
        try:
            async def _pull(_port: str, *, ctx_id: str | int | None = None) -> Any:
                return await udp_node.compute_output("packet", ctx_id=ctx_id)

            decoder.pull = _pull  # type: ignore[method-assign]
            _send_udp_json(port=port, payload=_skeleton_payload(frame_id=1))
            await self._wait_exec_calls(ctx, at_least=1, timeout_s=1.5)

            outputs = await decoder.on_exec(ctx.calls[0][1], "packet")
            self.assertEqual(outputs, ["packet"])

            self.assertEqual(sorted(decoder._skeletons_by_key.keys()), ["Model_A"])
            self.assertEqual(decoder._selected_key, "Model_A")

            payload = await decoder.compute_output("selectedSkeleton", ctx_id=None)
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            self.assertEqual(payload["modelName"], "Model_A")
            self.assertEqual(payload["boneCount"], 1)
        finally:
            await udp_node.close()

    async def test_chunked_payload_emits_only_after_complete_frame(self) -> None:
        port = _free_udp_port()
        bus, udp_node, ctx = await self._setup_udp_runtime(port=port)
        decoder = await self._new_decoder()
        try:
            async def _pull(_port: str, *, ctx_id: str | int | None = None) -> Any:
                return await udp_node.compute_output("packet", ctx_id=ctx_id)

            decoder.pull = _pull  # type: ignore[method-assign]
            _send_udp_json(port=port, payload=_skeleton_payload(frame_id=10, chunk_index=0, chunk_count=2))
            await self._wait_exec_calls(ctx, at_least=1, timeout_s=1.5)
            first_outputs = await decoder.on_exec(ctx.calls[0][1], "packet")
            self.assertEqual(first_outputs, [])

            _send_udp_json(port=port, payload=_skeleton_payload(frame_id=10, chunk_index=1, chunk_count=2))
            await self._wait_exec_calls(ctx, at_least=2, timeout_s=1.5)
            second_outputs = await decoder.on_exec(ctx.calls[1][1], "packet")
            self.assertEqual(second_outputs, ["packet"])

            payload = await decoder.compute_output("selectedSkeleton", ctx_id=None)
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            self.assertEqual([bone["name"] for bone in payload["bones"]], ["bone_10_0", "bone_10_1"])
            self.assertEqual(int(payload["trailer"]["assembledChunkCount"]), 2)
        finally:
            await udp_node.close()

    async def test_deactivate_udp_in_stops_decoder_exec(self) -> None:
        port = _free_udp_port()
        bus, udp_node, ctx = await self._setup_udp_runtime(port=port)
        try:
            _send_udp_json(port=port, payload=_skeleton_payload(frame_id=1))
            await self._wait_exec_calls(ctx, at_least=1, timeout_s=1.5)
            self.assertEqual(len(ctx.calls), 1)

            await udp_node.on_lifecycle(False, {"case": "deactivate"})
            _send_udp_json(port=port, payload=_skeleton_payload(frame_id=2))
            await asyncio.sleep(0.4)
            self.assertEqual(len(ctx.calls), 1)
        finally:
            await udp_node.close()


if __name__ == "__main__":
    unittest.main()
