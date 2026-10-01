import asyncio
import os
import socket
import struct
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
from f8pyengine.operators.udp_in import UdpInRuntimeNode  # noqa: E402
from f8pyengine.operators.vmc_decoder import VmcDecoderRuntimeNode  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _osc_pad(raw: bytes) -> bytes:
    pad = (4 - (len(raw) & 0x03)) & 0x03
    return raw + (b"\x00" * pad)


def _osc_string(text: str) -> bytes:
    return _osc_pad(text.encode("utf-8") + b"\x00")


def _osc_message(address: str, args: list[Any]) -> bytes:
    tags = [","]
    payload = bytearray()
    for arg in args:
        if isinstance(arg, str):
            tags.append("s")
            payload.extend(_osc_string(arg))
        elif isinstance(arg, int):
            tags.append("i")
            payload.extend(struct.pack(">i", int(arg)))
        else:
            tags.append("f")
            payload.extend(struct.pack(">f", float(arg)))
    out = bytearray()
    out.extend(_osc_string(address))
    out.extend(_osc_string("".join(tags)))
    out.extend(payload)
    return bytes(out)


def _osc_bundle(messages: list[bytes]) -> bytes:
    out = bytearray()
    out.extend(b"#bundle\x00")
    out.extend(struct.pack(">Q", 0))
    for msg in messages:
        out.extend(struct.pack(">I", len(msg)))
        out.extend(msg)
    return bytes(out)


def _send_udp(*, port: int, payload: bytes) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(payload, ("127.0.0.1", int(port)))


@dataclass
class _FakeEntrypointContext:
    calls: list[tuple[str, str]]

    def __init__(self) -> None:
        self.calls = []

    async def emit_exec(self, out_port: str, *, exec_id: str | int) -> None:
        self.calls.append((str(out_port), str(exec_id)))


class VmcDecoderPipelineTests(unittest.IsolatedAsyncioTestCase):
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

    async def _new_decoder(self) -> VmcDecoderRuntimeNode:
        decoder_node = F8RuntimeNode(
            nodeId="vmc_decoder",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=VmcDecoderRuntimeNode.SPEC.operatorClass,
            stateFields=list(VmcDecoderRuntimeNode.SPEC.stateFields or []),
            stateValues={"cleanupAfterMs": 10000, "selectedKey": ""},
            execInPorts=list(VmcDecoderRuntimeNode.SPEC.execInPorts or []),
            execOutPorts=list(VmcDecoderRuntimeNode.SPEC.execOutPorts or []),
            dataInPorts=list(VmcDecoderRuntimeNode.SPEC.dataInPorts or []),
            dataOutPorts=list(VmcDecoderRuntimeNode.SPEC.dataOutPorts or []),
        )
        return VmcDecoderRuntimeNode(node_id="vmc_decoder", node=decoder_node, initial_state={"cleanupAfterMs": 10000})

    async def _wait_exec_calls(self, ctx: _FakeEntrypointContext, *, at_least: int, timeout_s: float = 1.2) -> None:
        end = asyncio.get_running_loop().time() + timeout_s
        while True:
            if len(ctx.calls) >= at_least:
                return
            if asyncio.get_running_loop().time() >= end:
                self.fail(f"timed out waiting for exec calls >= {at_least}")
            await asyncio.sleep(0.01)

    async def test_decode_single_bone_message_to_skeleton_payload(self) -> None:
        port = _free_udp_port()
        bus, udp_node, ctx = await self._setup_udp_runtime(port=port)
        decoder = await self._new_decoder()
        try:
            async def _pull(_port: str, *, ctx_id: str | int | None = None) -> Any:
                return await udp_node.compute_output("packet", ctx_id=ctx_id)

            decoder.pull = _pull  # type: ignore[method-assign]
            msg = _osc_message("/VMC/Ext/Bone/Pos", ["Hips", 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9])
            _send_udp(port=port, payload=msg)
            await self._wait_exec_calls(ctx, at_least=1, timeout_s=1.5)

            outputs = await decoder.on_exec(ctx.calls[0][1], "packet")
            self.assertEqual(outputs, ["packet"])
            payload = await decoder.compute_output("selectedSkeleton", ctx_id=None)
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            self.assertEqual(payload["modelName"], "VMC")
            self.assertEqual(payload["skeletonProtocol"], "unity_humanoid")
            self.assertEqual(payload["boneCount"], 1)
            self.assertEqual(payload["bones"][0]["name"], "Hips")
        finally:
            await udp_node.close()

    async def test_bundle_updates_multiple_bones(self) -> None:
        port = _free_udp_port()
        bus, udp_node, ctx = await self._setup_udp_runtime(port=port)
        decoder = await self._new_decoder()
        try:
            async def _pull(_port: str, *, ctx_id: str | int | None = None) -> Any:
                return await udp_node.compute_output("packet", ctx_id=ctx_id)

            decoder.pull = _pull  # type: ignore[method-assign]
            msg_a = _osc_message("/VMC/Ext/Bone/Pos", ["A", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            msg_b = _osc_message("/VMC/Ext/Bone/Pos", ["B", 1.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.9])
            _send_udp(port=port, payload=_osc_bundle([msg_a, msg_b]))
            await self._wait_exec_calls(ctx, at_least=1, timeout_s=1.5)

            outputs = await decoder.on_exec(ctx.calls[0][1], "packet")
            self.assertEqual(outputs, ["packet"])
            payload = await decoder.compute_output("selectedSkeleton", ctx_id=None)
            self.assertIsInstance(payload, dict)
            assert isinstance(payload, dict)
            self.assertEqual(payload["boneCount"], 2)
            self.assertEqual([bone["name"] for bone in payload["bones"]], ["A", "B"])
        finally:
            await udp_node.close()

    async def test_non_vmc_message_does_not_commit(self) -> None:
        port = _free_udp_port()
        bus, udp_node, ctx = await self._setup_udp_runtime(port=port)
        decoder = await self._new_decoder()
        try:
            async def _pull(_port: str, *, ctx_id: str | int | None = None) -> Any:
                return await udp_node.compute_output("packet", ctx_id=ctx_id)

            decoder.pull = _pull  # type: ignore[method-assign]
            msg = _osc_message("/not/vmc", [1, 2, 3])
            _send_udp(port=port, payload=msg)
            await self._wait_exec_calls(ctx, at_least=1, timeout_s=1.5)

            outputs = await decoder.on_exec(ctx.calls[0][1], "packet")
            self.assertEqual(outputs, [])
            payload = await decoder.compute_output("selectedSkeleton", ctx_id=None)
            self.assertIsNone(payload)
        finally:
            await udp_node.close()

    async def test_available_keys_selects_vmc_model(self) -> None:
        port = _free_udp_port()
        bus, udp_node, ctx = await self._setup_udp_runtime(port=port)
        decoder = await self._new_decoder()
        try:
            async def _pull(_port: str, *, ctx_id: str | int | None = None) -> Any:
                return await udp_node.compute_output("packet", ctx_id=ctx_id)

            decoder.pull = _pull  # type: ignore[method-assign]
            msg = _osc_message("/VMC/Ext/Bone/Pos", ["Hips", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            _send_udp(port=port, payload=msg)
            await self._wait_exec_calls(ctx, at_least=1, timeout_s=1.5)

            outputs = await decoder.on_exec(ctx.calls[0][1], "packet")
            self.assertEqual(outputs, ["packet"])
            self.assertEqual(sorted(decoder._skeletons_by_key.keys()), ["VMC"])
            self.assertEqual(decoder._selected_key, "VMC")
        finally:
            await udp_node.close()


if __name__ == "__main__":
    unittest.main()
