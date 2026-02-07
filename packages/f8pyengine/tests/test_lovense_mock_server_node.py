import asyncio
import json
import os
import socket
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SDK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "f8pysdk"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SDK_ROOT not in sys.path:
    sys.path.insert(0, SDK_ROOT)

from f8pysdk.generated import F8RuntimeGraph, F8RuntimeNode  # noqa: E402
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry  # noqa: E402
from f8pysdk.service_host import ServiceHost, ServiceHostConfig  # noqa: E402
from f8pysdk.testing import ServiceBusHarness  # noqa: E402

from f8pyengine.constants import SERVICE_CLASS  # noqa: E402
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


class LovenseMockServerNodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_publishes_event_and_survives_rungraph_redeploy(self) -> None:
        port = _free_port()
        harness = ServiceBusHarness()
        bus = harness.create_bus("svcA")

        reg = RuntimeNodeRegistry.instance()
        register_operator(reg)

        _ = ServiceHost(bus, config=ServiceHostConfig(service_class=SERVICE_CLASS), registry=reg)

        op = F8RuntimeNode(
            nodeId="lov1",
            serviceId="svcA",
            serviceClass=SERVICE_CLASS,
            operatorClass=LovenseMockServerRuntimeNode.SPEC.operatorClass,
            stateFields=list(LovenseMockServerRuntimeNode.SPEC.stateFields or []),
            stateValues={"bindAddress": "127.0.0.1", "port": port},
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
            payload={"command": "Pattern", "toy": "lush", "timeSec": 1, "strength": 20, "apiVer": 1},
        )
        self.assertEqual(code, 200)

        ev1 = (await bus.get_state("lov1", "event")).value
        self.assertIsInstance(ev1, dict)
        self.assertEqual(ev1.get("seq"), 1)
        self.assertEqual((ev1.get("summary") or {}).get("type"), "vibration_pattern")

        # Redeploy rungraph (same ports/state): node instance should be preserved.
        graph_v2 = F8RuntimeGraph(graphId="g1", revision="r2", nodes=[op], edges=[])
        await bus.set_rungraph(graph_v2)
        node2 = bus.get_node("lov1")
        self.assertIs(node1, node2)

        code, _ = await _http_post_json(
            host="127.0.0.1",
            port=port,
            path="/command",
            payload={"command": "Function", "toy": "solace", "timeSec": 2, "action": "Stop", "apiVer": 1},
        )
        self.assertEqual(code, 200)

        ev2 = (await bus.get_state("lov1", "event")).value
        self.assertIsInstance(ev2, dict)
        self.assertEqual(ev2.get("seq"), 2)
        self.assertEqual((ev2.get("summary") or {}).get("type"), "stop")

        if isinstance(node2, LovenseMockServerRuntimeNode):
            await node2.close()


if __name__ == "__main__":
    unittest.main()

