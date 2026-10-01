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

from f8pysdk.specs import (  # noqa: E402
    F8RuntimeGraph,
    F8RuntimeNode,
)
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.udp_in import UdpInRuntimeNode  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _send_udp(*, port: int, payload: bytes) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(payload, ("127.0.0.1", int(port)))


@dataclass
class _FakeEntrypointContext:
    calls: list[tuple[str, str]]
    emit_delay_s: float

    def __init__(self, *, emit_delay_s: float = 0.0) -> None:
        self.calls = []
        self.emit_delay_s = float(emit_delay_s)

    async def emit_exec(self, out_port: str, *, exec_id: str | int) -> None:
        if self.emit_delay_s > 0:
            await asyncio.sleep(self.emit_delay_s)
        self.calls.append((str(out_port), str(exec_id)))


class UdpInTests(unittest.IsolatedAsyncioTestCase):
    async def _setup_runtime(self, *, port: int) -> tuple[object, UdpInRuntimeNode, _FakeEntrypointContext]:
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
        graph = F8RuntimeGraph(
            graphId="g_udp_in",
            revision="r1",
            nodes=[udp],
            edges=[],
        )
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

    async def _teardown_runtime(self, node: UdpInRuntimeNode) -> None:
        await node.close()

    async def _wait_exec_calls(self, ctx: _FakeEntrypointContext, *, at_least: int, timeout_s: float = 1.2) -> None:
        end = asyncio.get_running_loop().time() + timeout_s
        while True:
            if len(ctx.calls) >= at_least:
                return
            if asyncio.get_running_loop().time() >= end:
                self.fail(f"timed out waiting for exec calls >= {at_least}")
            await asyncio.sleep(0.01)

    async def _compute_output(self, node: UdpInRuntimeNode, port: str) -> Any:
        return await node.compute_output(port, ctx_id=f"ctx:{port}")

    def test_spec_has_no_value_alias_output(self) -> None:
        state_names = {str(field.name or "") for field in list(UdpInRuntimeNode.SPEC.stateFields or [])}
        port_names = {str(port.name or "") for port in list(UdpInRuntimeNode.SPEC.dataOutPorts or [])}
        self.assertNotIn("outputMode", state_names)
        self.assertNotIn("value", port_names)

    def test_spec_excludes_packet_rate_state_fields(self) -> None:
        state_names = {str(field.name or "") for field in list(UdpInRuntimeNode.SPEC.stateFields or [])}
        self.assertNotIn("packetCount", state_names)
        self.assertNotIn("droppedPackets", state_names)
        self.assertNotIn("lastRemoteAddress", state_names)
        self.assertNotIn("lastRemotePort", state_names)
        self.assertNotIn("lastByteLength", state_names)
        self.assertNotIn("lastParseError", state_names)

    async def test_json_packet_exposes_text_json_and_packet_views(self) -> None:
        port = _free_udp_port()
        bus, node, ctx = await self._setup_runtime(port=port)
        try:
            payload = {"kind": "ping", "count": 3}
            _send_udp(port=port, payload=json.dumps(payload).encode("utf-8"))
            await self._wait_exec_calls(ctx, at_least=1, timeout_s=1.5)

            value_alias = await self._compute_output(node, "value")
            self.assertIsNone(value_alias)

            text_value = await self._compute_output(node, "text")
            self.assertEqual(text_value, json.dumps(payload))

            json_value = await self._compute_output(node, "json")
            self.assertEqual(json_value, payload)

            raw_value = await self._compute_output(node, "raw")
            self.assertIsInstance(raw_value, bytearray)
            self.assertEqual(bytes(raw_value), json.dumps(payload).encode("utf-8"))

            packet = await self._compute_output(node, "packet")
            self.assertIsInstance(packet, dict)
            assert isinstance(packet, dict)
            self.assertEqual(packet["json"], payload)
            self.assertEqual(packet["text"], json.dumps(payload))
            self.assertNotIn("value", packet)
            self.assertEqual(packet["remoteAddress"], "127.0.0.1")
            self.assertGreater(int(packet["remotePort"]), 0)
        finally:
            await self._teardown_runtime(node)

    async def test_raw_output_preserves_non_ascii_bytes(self) -> None:
        port = _free_udp_port()
        bus, node, ctx = await self._setup_runtime(port=port)
        try:
            raw = bytes([0xFF, 0x00, 0x41, 0x80])
            _send_udp(port=port, payload=raw)
            await self._wait_exec_calls(ctx, at_least=1, timeout_s=1.5)

            raw_out = await self._compute_output(node, "raw")
            self.assertIsInstance(raw_out, bytearray)
            self.assertEqual(bytes(raw_out), raw)

            text = await self._compute_output(node, "text")
            self.assertIsInstance(text, str)
            self.assertIn("A", text)
            self.assertIn("\ufffd", text)

            json_value = await self._compute_output(node, "json")
            self.assertIsNone(json_value)

            packet = await self._compute_output(node, "packet")
            self.assertIsInstance(packet, dict)
            assert isinstance(packet, dict)
            self.assertEqual(bytes(packet["raw"]), raw)
            self.assertEqual(int(packet["byteLength"]), len(raw))
            self.assertNotIn("value", packet)
        finally:
            await self._teardown_runtime(node)

    async def test_invalid_json_keeps_text_without_packet_rate_state(self) -> None:
        port = _free_udp_port()
        bus, node, ctx = await self._setup_runtime(port=port)
        try:
            payload = b"{bad json"
            _send_udp(port=port, payload=payload)
            await self._wait_exec_calls(ctx, at_least=1, timeout_s=1.5)

            text_value = await self._compute_output(node, "text")
            self.assertEqual(text_value, payload.decode("utf-8", errors="replace"))

            json_value = await self._compute_output(node, "json")
            self.assertIsNone(json_value)

            packet = await self._compute_output(node, "packet")
            self.assertIsInstance(packet, dict)
            assert isinstance(packet, dict)
            self.assertFalse(bool(packet["jsonValid"]))
            self.assertNotIn("value", packet)

            parse_error = await bus.get_state("udp_in", "lastParseError")
            self.assertFalse(parse_error.found)
        finally:
            await self._teardown_runtime(node)

    async def test_bind_security_rejects_non_loopback_by_default(self) -> None:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="udp_in1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=UdpInRuntimeNode.SPEC.operatorClass,
            stateFields=list(UdpInRuntimeNode.SPEC.stateFields or []),
            stateValues={"bindAddress": "127.0.0.1", "port": _free_udp_port()},
        )
        await bus.set_rungraph(F8RuntimeGraph(graphId="g_udp_in_bind", revision="r1", nodes=[op], edges=[]))
        try:
            with self.assertRaises(ValueError):
                await bus.publish_state_external("udp_in1", "bindAddress", "0.0.0.0", source="test")

            await bus.publish_state_external("udp_in1", "allowNonLoopbackBind", True, source="test")
            await bus.publish_state_external("udp_in1", "bindAddress", "0.0.0.0", source="test")
        finally:
            node = bus.get_node("udp_in1")
            if isinstance(node, UdpInRuntimeNode):
                await node.close()

    async def test_packet_output_tracks_exec_context_snapshot(self) -> None:
        port = _free_udp_port()
        bus, node, ctx = await self._setup_runtime(port=port)
        try:
            _send_udp(port=port, payload=b"first-packet")
            await self._wait_exec_calls(ctx, at_least=1, timeout_s=1.5)
            first_exec_id = ctx.calls[0][1]

            _send_udp(port=port, payload=b"second-packet")
            await self._wait_exec_calls(ctx, at_least=2, timeout_s=1.5)
            second_exec_id = ctx.calls[1][1]

            first_packet = await node.compute_output("packet", ctx_id=first_exec_id)
            second_packet = await node.compute_output("packet", ctx_id=second_exec_id)

            self.assertIsInstance(first_packet, dict)
            self.assertIsInstance(second_packet, dict)
            assert isinstance(first_packet, dict)
            assert isinstance(second_packet, dict)
            self.assertEqual(first_packet["text"], "first-packet")
            self.assertEqual(second_packet["text"], "second-packet")
            self.assertNotIn("value", first_packet)
            self.assertNotIn("value", second_packet)

            latest_packet = await self._compute_output(node, "packet")
            self.assertIsInstance(latest_packet, dict)
            assert isinstance(latest_packet, dict)
            self.assertEqual(latest_packet["text"], "second-packet")
        finally:
            await self._teardown_runtime(node)

    async def test_exec_triggers_are_fifo_when_emit_context_is_slow(self) -> None:
        port = _free_udp_port()
        _bus, node, _ctx = await self._setup_runtime(port=port)
        slow_ctx = _FakeEntrypointContext(emit_delay_s=0.01)
        await node.start_entrypoint(slow_ctx)  # type: ignore[arg-type]
        try:
            packet_count = 20
            for index in range(packet_count):
                _send_udp(port=port, payload=f"packet-{index:02d}".encode("utf-8"))

            await self._wait_exec_calls(slow_ctx, at_least=packet_count, timeout_s=2.5)

            exec_ids = [call[1] for call in slow_ctx.calls]
            self.assertEqual(len(exec_ids), packet_count)
            self.assertEqual(len(set(exec_ids)), packet_count)
            self.assertEqual([call[0] for call in slow_ctx.calls], ["packet"] * packet_count)

            first_packet = await node.compute_output("packet", ctx_id=exec_ids[0])
            last_packet = await node.compute_output("packet", ctx_id=exec_ids[-1])

            self.assertIsInstance(first_packet, dict)
            self.assertIsInstance(last_packet, dict)
            assert isinstance(first_packet, dict)
            assert isinstance(last_packet, dict)
            self.assertEqual(first_packet["text"], "packet-00")
            self.assertEqual(last_packet["text"], "packet-19")
        finally:
            await self._teardown_runtime(node)


if __name__ == "__main__":
    unittest.main()
