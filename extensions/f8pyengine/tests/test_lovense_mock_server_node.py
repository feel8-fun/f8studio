import asyncio
import base64
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
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8RuntimeGraph,
    F8RuntimeNode,
)
from f8pysdk.nodes import OperatorNode  # noqa: E402
from f8pysdk.registry import Registry, create_runtime_node_registry  # noqa: E402
from f8pysdk.host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
from f8pyengine.pyengine_node_registry import register_pyengine_specs  # noqa: E402
from f8pyengine.pyengine_service import PyEngineService  # noqa: E402
from f8pyengine.operators.lovense_mock_server import (  # noqa: E402
    LovenseMockServerRuntimeNode,
    register_operator,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _http_post_json(*, host: str, port: int, path: str, payload: dict[str, object]) -> tuple[int, str]:
    body = json.dumps(payload).encode("utf-8")
    req = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Connection: close\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "\r\n"
    ).encode("ascii") + body
    reader, writer = await asyncio.open_connection(host, int(port))
    try:
        writer.write(req)
        await writer.drain()
        raw = await reader.read()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    head = raw.split(b"\r\n", 1)[0].decode("ascii", errors="replace")
    parts = head.split(" ", 2)
    code = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 0
    return code, head


async def _ws_open_and_send_large_text(*, host: str, port: int, path: str, payload_size: int) -> bytes:
    key = base64.b64encode(b"0123456789012345").decode("ascii")
    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode("ascii")
    reader, writer = await asyncio.open_connection(host, int(port))
    try:
        writer.write(req)
        await writer.drain()
        _ = await reader.readuntil(b"\r\n\r\n")
        _ = await reader.read(4096)

        payload = b"a" * int(payload_size)
        frame = bytearray([0x81])
        ln = len(payload)
        if ln < 126:
            frame.append(ln)
        elif ln < 65536:
            frame.append(126)
            frame.extend(int(ln).to_bytes(2, "big"))
        else:
            frame.append(127)
            frame.extend((0).to_bytes(4, "big"))
            frame.extend(int(ln).to_bytes(4, "big"))
        writer.write(bytes(frame) + payload)
        await writer.drain()
        return await reader.read(4096)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


def _exec_edge(*, edge_id: str, from_node: str, from_port: str, to_node: str, to_port: str) -> F8Edge:
    return F8Edge(
        edgeId=edge_id,
        fromServiceId="svcA",
        fromOperatorId=from_node,
        fromPort=from_port,
        toServiceId="svcA",
        toOperatorId=to_node,
        toPort=to_port,
        kind=F8EdgeKindEnum.exec,
        strategy=F8EdgeStrategyEnum.latest,
    )


@dataclass
class _RuntimeStub:
    bus: object


class _ProbeExecRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        del initial_state
        super().__init__(
            node_id=node_id,
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
            exec_in_ports=list(node.execInPorts or []),
            exec_out_ports=list(node.execOutPorts or []),
        )
        self.calls = 0
        self.exec_ids: list[str] = []
        self.delay_s = 0.05

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        _ = in_port
        self.calls += 1
        self.exec_ids.append(str(exec_id))
        await asyncio.sleep(float(self.delay_s))
        return []


class LovenseMockServerNodeTests(unittest.IsolatedAsyncioTestCase):
    def test_spec_exposes_event_as_data_not_state(self) -> None:
        state_names = {str(field.name or "") for field in list(LovenseMockServerRuntimeNode.SPEC.stateFields or [])}
        data_names = {str(port.name or "") for port in list(LovenseMockServerRuntimeNode.SPEC.dataOutPorts or [])}
        self.assertNotIn("event", state_names)
        self.assertIn("event", data_names)

    async def test_publishes_event_output_and_survives_rungraph_redeploy(self) -> None:
        port = _free_port()
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")

        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))

        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="lov1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=LovenseMockServerRuntimeNode.SPEC.operatorClass,
            stateFields=list(LovenseMockServerRuntimeNode.SPEC.stateFields or []),
            stateValues={"bindAddress": "127.0.0.1", "port": port},
            dataOutPorts=list(LovenseMockServerRuntimeNode.SPEC.dataOutPorts or []),
        )
        graph_v1 = F8RuntimeGraph(graphId="g1", revision="r1", nodes=[op], edges=[])
        await bus.set_rungraph(graph_v1)

        node1 = bus.get_node("lov1")
        self.assertIsInstance(node1, LovenseMockServerRuntimeNode)

        # Wait for server to start (listening=True).
        for _ in range(50):
            st = await bus.get_state("lov1", "listening")
            if st.found and st.value is True:
                break
            await asyncio.sleep(0.02)
        st = await bus.get_state("lov1", "listening")
        self.assertTrue(st.value is True)

        code, _ = await _http_post_json(
            host="127.0.0.1",
            port=port,
            path="/command",
            payload={"command": "Pattern", "toy": "f082c00246fa", "timeSec": 1, "strength": 20, "apiVer": 1},
        )
        self.assertEqual(code, 200)

        ev1 = await node1.compute_output("event")
        self.assertIsInstance(ev1, dict)
        self.assertEqual(ev1.get("seq"), 1)
        self.assertEqual(((ev1.get("command") or {}).get("kind") or ""), "vibration_pattern")
        self.assertIn("Lush", ((ev1.get("toys") or {}).get("names") or []))

        # Ping should NOT replace the latest command event.
        code, _ = await _http_post_json(
            host="127.0.0.1",
            port=port,
            path="/command",
            payload={"type": "ping"},
        )
        self.assertEqual(code, 200)
        ev_after_ping = await node1.compute_output("event")
        self.assertIsInstance(ev_after_ping, dict)
        self.assertEqual(ev_after_ping.get("seq"), 1)

        # Redeploy rungraph (same ports/state): node instance should be preserved.
        graph_v2 = F8RuntimeGraph(graphId="g1", revision="r2", nodes=[op], edges=[])
        await bus.set_rungraph(graph_v2)
        node2 = bus.get_node("lov1")
        self.assertIs(node1, node2)

        code, _ = await _http_post_json(
            host="127.0.0.1",
            port=port,
            path="/command",
            payload={"command": "Function", "toy": "ff922f7fd345", "timeSec": 2, "action": "Stop", "apiVer": 1},
        )
        self.assertEqual(code, 200)

        assert isinstance(node2, LovenseMockServerRuntimeNode)
        ev2 = await node2.compute_output("event")
        self.assertIsInstance(ev2, dict)
        self.assertEqual(ev2.get("seq"), 2)
        self.assertEqual(((ev2.get("command") or {}).get("kind") or ""), "stop")
        self.assertIn("Solace Pro", ((ev2.get("toys") or {}).get("names") or []))

        if isinstance(node2, LovenseMockServerRuntimeNode):
            await node2.close()

    async def test_websocket_large_frame_is_rejected(self) -> None:
        port = _free_port()
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")

        reg = create_runtime_node_registry()
        register_operator(Registry.wrap(reg))
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="lov_ws",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=LovenseMockServerRuntimeNode.SPEC.operatorClass,
            stateFields=list(LovenseMockServerRuntimeNode.SPEC.stateFields or []),
            stateValues={"bindAddress": "127.0.0.1", "port": port},
            dataOutPorts=list(LovenseMockServerRuntimeNode.SPEC.dataOutPorts or []),
        )
        await bus.set_rungraph(F8RuntimeGraph(graphId="g_ws", revision="r1", nodes=[op], edges=[]))

        for _ in range(50):
            st = await bus.get_state("lov_ws", "listening")
            if st.found and st.value is True:
                break
            await asyncio.sleep(0.02)

        data = await _ws_open_and_send_large_text(host="127.0.0.1", port=port, path="/v1", payload_size=70_000)
        self.assertTrue((data == b"") or (data.startswith(b"\x88")))

        node = bus.get_node("lov_ws")
        self.assertIsInstance(node, LovenseMockServerRuntimeNode)
        if isinstance(node, LovenseMockServerRuntimeNode):
            await node.close()


class LovenseMockServerExecTriggerTests(unittest.IsolatedAsyncioTestCase):
    async def _setup_exec_runtime(self, *, port: int) -> tuple[object, PyEngineService, _RuntimeStub]:
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")
        reg = create_runtime_node_registry()
        register_pyengine_specs(Registry.wrap(reg))
        reg.register_operator_factory(
            SERVICE_CLASS,
            "f8.test_probe_exec",
            lambda node_id, node, initial_state: _ProbeExecRuntimeNode(
                node_id=node_id, node=node, initial_state=initial_state
            ),
            overwrite=True,
        )
        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)
        service = PyEngineService()
        runtime = _RuntimeStub(bus=bus)
        await service.setup(runtime)  # type: ignore[arg-type]

        lov = F8RuntimeNode(
            nodeId="lov_exec",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=LovenseMockServerRuntimeNode.SPEC.operatorClass,
            stateFields=list(LovenseMockServerRuntimeNode.SPEC.stateFields or []),
            stateValues={"bindAddress": "127.0.0.1", "port": port},
            execInPorts=list(LovenseMockServerRuntimeNode.SPEC.execInPorts or []),
            execOutPorts=list(LovenseMockServerRuntimeNode.SPEC.execOutPorts or []),
            dataOutPorts=list(LovenseMockServerRuntimeNode.SPEC.dataOutPorts or []),
        )
        probe = F8RuntimeNode(
            nodeId="probe_exec",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass="f8.test_probe_exec",
            execInPorts=["exec"],
            execOutPorts=[],
        )
        graph = F8RuntimeGraph(
            graphId="g_exec",
            revision="r1",
            nodes=[lov, probe],
            edges=[_exec_edge(edge_id="e1", from_node="lov_exec", from_port="event", to_node="probe_exec", to_port="exec")],
        )
        await bus.set_rungraph(graph)  # type: ignore[attr-defined]
        for _ in range(50):
            st = await bus.get_state("lov_exec", "listening")
            if st.found and st.value is True:
                break
            await asyncio.sleep(0.02)
        return bus, service, runtime

    async def _teardown_exec_runtime(self, service: PyEngineService, runtime: _RuntimeStub) -> None:
        bus = runtime.bus
        node = bus.get_node("lov_exec")
        if isinstance(node, LovenseMockServerRuntimeNode):
            await node.close()
        await service.teardown(runtime)  # type: ignore[arg-type]

    async def test_event_exec_trigger_and_ping_no_exec(self) -> None:
        port = _free_port()
        bus, service, runtime = await self._setup_exec_runtime(port=port)
        try:
            code, _ = await _http_post_json(
                host="127.0.0.1",
                port=port,
                path="/command",
                payload={"command": "Function", "toy": "lush", "timeSec": 1, "action": "Vibrate:4", "apiVer": 1},
            )
            self.assertEqual(code, 200)

            probe_node = bus.get_node("probe_exec")
            self.assertIsInstance(probe_node, _ProbeExecRuntimeNode)
            assert isinstance(probe_node, _ProbeExecRuntimeNode)
            for _ in range(100):
                if probe_node.calls >= 1:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(probe_node.calls, 1)

            code, _ = await _http_post_json(
                host="127.0.0.1",
                port=port,
                path="/command",
                payload={"type": "ping"},
            )
            self.assertEqual(code, 200)
            await asyncio.sleep(0.2)
            self.assertEqual(probe_node.calls, 1)
        finally:
            await self._teardown_exec_runtime(service, runtime)

    async def test_event_exec_burst_coalesces_to_latest(self) -> None:
        port = _free_port()
        bus, service, runtime = await self._setup_exec_runtime(port=port)
        try:
            tasks = []
            for i in range(20):
                task = asyncio.create_task(
                    _http_post_json(
                        host="127.0.0.1",
                        port=port,
                        path="/command",
                        payload={
                            "command": "Function",
                            "toy": "lush",
                            "timeSec": 1,
                            "action": f"Vibrate:{(i % 20) + 1}",
                            "apiVer": 1,
                        },
                    )
                )
                tasks.append(task)
            results = await asyncio.gather(*tasks)
            self.assertTrue(all(code == 200 for code, _ in results))

            probe_node = bus.get_node("probe_exec")
            self.assertIsInstance(probe_node, _ProbeExecRuntimeNode)
            assert isinstance(probe_node, _ProbeExecRuntimeNode)
            # Wait until call count settles, instead of sleeping a fixed long window.
            end = asyncio.get_running_loop().time() + 1.2
            last_calls = -1
            stable_ticks = 0
            while asyncio.get_running_loop().time() < end:
                current = probe_node.calls
                if current > 0 and current == last_calls:
                    stable_ticks += 1
                    if stable_ticks >= 5:
                        break
                else:
                    stable_ticks = 0
                    last_calls = current
                await asyncio.sleep(0.02)
            self.assertGreater(probe_node.calls, 0)
            self.assertLess(probe_node.calls, 20)
            self.assertTrue(probe_node.exec_ids[-1].endswith(":20"))
        finally:
            await self._teardown_exec_runtime(service, runtime)


if __name__ == "__main__":
    unittest.main()
