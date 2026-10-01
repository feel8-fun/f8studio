import asyncio
import os
import socket
import sys
import unittest
from typing import Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from f8pysdk.specs import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.operators.udp_out import UdpOutRuntimeNode, register_operator  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402


def _free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class UdpOutTests(unittest.IsolatedAsyncioTestCase):
    async def _build_node(self, *, state_values: dict[str, Any]) -> tuple[Any, UdpOutRuntimeNode]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
        op = F8RuntimeNode(
            nodeId="udp_out1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=UdpOutRuntimeNode.SPEC.operatorClass,
            stateFields=list(UdpOutRuntimeNode.SPEC.stateFields or []),
            stateValues=dict(state_values),
        )
        graph = F8RuntimeGraph(graphId="g_udp_out", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph)
        node = bus.get_node("udp_out1")
        self.assertIsInstance(node, UdpOutRuntimeNode)
        assert isinstance(node, UdpOutRuntimeNode)
        return bus, node

    def test_registered_in_pyengine_specs(self) -> None:
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        desc = reg.describe(SERVICE_CLASS)
        operator_classes = {str(spec.operatorClass or "") for spec in list(desc.operators or [])}
        self.assertIn("f8.udp_out", operator_classes)

    async def test_sends_text_datagram_to_target_port(self) -> None:
        port = _free_udp_port()
        _bus, node = await self._build_node(
            state_values={"enabled": True, "host": "127.0.0.1", "port": port, "appendNewline": False, "forceText": True}
        )

        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", port))
        recv_sock.settimeout(1.0)

        emitted: list[tuple[str, Any]] = []

        async def _emit(port_name: str, value: Any, *, ts_ms: int | None = None) -> None:
            del ts_ms
            emitted.append((port_name, value))

        async def _pull(port_name: str, *, ctx_id: str | int | None = None) -> Any:
            del ctx_id
            if port_name == "value":
                return "L09999I020"
            return None

        node.emit = _emit  # type: ignore[method-assign]
        node.pull = _pull  # type: ignore[method-assign]

        try:
            await node.on_exec("e1")
            packet, addr = await asyncio.to_thread(recv_sock.recvfrom, 4096)
            self.assertEqual(packet.decode("utf-8"), "L09999I020")
            self.assertEqual(str(addr[0]), "127.0.0.1")
            self.assertIsNone(node._last_error)
            self.assertIn(("isOpen", True), emitted)
            self.assertIn(("sentBytes", len(packet)), emitted)
            self.assertIn(("error", ""), emitted)
        finally:
            recv_sock.close()
            await node.close()

    async def test_disabled_node_does_not_send_packets(self) -> None:
        port = _free_udp_port()
        _bus, node = await self._build_node(
            state_values={"enabled": False, "host": "127.0.0.1", "port": port, "appendNewline": False, "forceText": True}
        )

        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", port))
        recv_sock.settimeout(0.2)

        emitted: list[tuple[str, Any]] = []

        async def _emit(port_name: str, value: Any, *, ts_ms: int | None = None) -> None:
            del ts_ms
            emitted.append((port_name, value))

        async def _pull(_port_name: str, *, ctx_id: str | int | None = None) -> Any:
            del ctx_id
            return "ignored"

        node.emit = _emit  # type: ignore[method-assign]
        node.pull = _pull  # type: ignore[method-assign]

        try:
            await node.on_exec("e1")
            with self.assertRaises(TimeoutError):
                await asyncio.to_thread(recv_sock.recvfrom, 4096)
            self.assertIn(("isOpen", False), emitted)
            self.assertIn(("sentBytes", 0), emitted)
        finally:
            recv_sock.close()
            await node.close()

    async def test_validate_state_rejects_invalid_port(self) -> None:
        _bus, node = await self._build_node(
            state_values={"enabled": True, "host": "127.0.0.1", "port": 9000, "appendNewline": False, "forceText": True}
        )
        try:
            with self.assertRaises(ValueError):
                await node.validate_state("port", 0, ts_ms=1, meta={})
            with self.assertRaises(ValueError):
                await node.validate_state("host", "", ts_ms=2, meta={})
        finally:
            await node.close()

    async def test_force_text_false_rejects_non_string_non_bytes_values(self) -> None:
        port = _free_udp_port()
        _bus, node = await self._build_node(
            state_values={"enabled": True, "host": "127.0.0.1", "port": port, "appendNewline": False, "forceText": False}
        )

        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", port))
        recv_sock.settimeout(0.2)

        emitted: list[tuple[str, Any]] = []

        async def _emit(port_name: str, value: Any, *, ts_ms: int | None = None) -> None:
            del ts_ms
            emitted.append((port_name, value))

        async def _pull(_port_name: str, *, ctx_id: str | int | None = None) -> Any:
            del ctx_id
            return {"tcode": "L09999I020"}

        node.emit = _emit  # type: ignore[method-assign]
        node.pull = _pull  # type: ignore[method-assign]

        try:
            await node.on_exec("e1")
            with self.assertRaises(TimeoutError):
                await asyncio.to_thread(recv_sock.recvfrom, 4096)
            self.assertIn(("sentBytes", 0), emitted)
            error_values = [value for port_name, value in emitted if port_name == "error"]
            self.assertTrue(error_values)
            self.assertIn("forceText is disabled", str(error_values[-1]))
        finally:
            recv_sock.close()
            await node.close()

    async def test_force_text_false_sends_bytes_without_text_conversion(self) -> None:
        port = _free_udp_port()
        _bus, node = await self._build_node(
            state_values={"enabled": True, "host": "127.0.0.1", "port": port, "appendNewline": False, "forceText": False}
        )

        recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        recv_sock.bind(("127.0.0.1", port))
        recv_sock.settimeout(1.0)

        async def _pull(_port_name: str, *, ctx_id: str | int | None = None) -> Any:
            del ctx_id
            return b"L09999I020"

        node.pull = _pull  # type: ignore[method-assign]

        try:
            await node.on_exec("e1")
            packet, _addr = await asyncio.to_thread(recv_sock.recvfrom, 4096)
            self.assertEqual(packet, b"L09999I020")
        finally:
            recv_sock.close()
            await node.close()


if __name__ == "__main__":
    unittest.main()
