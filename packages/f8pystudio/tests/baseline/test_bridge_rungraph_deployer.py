from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from _bootstrap import ensure_package_importable

ensure_package_importable()

from f8pystudio.bridge.rungraph_deployer import (
    NatsRungraphGateway,
    RungraphDeployConfig,
    RungraphDeployRequest,
)


class _FakeGraph:
    def model_dump(self, *, mode: str, by_alias: bool) -> dict[str, object]:
        _ = (mode, by_alias)
        return {"graphId": "g1", "meta": {}}


class _FakeTransport:
    def __init__(self, response_payload: dict[str, object]) -> None:
        self._response_payload = response_payload
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True

    async def request(self, subject: str, payload: bytes, timeout: float, raise_on_error: bool) -> bytes:
        _ = (subject, payload, timeout, raise_on_error)
        return json.dumps(self._response_payload).encode("utf-8")


class RungraphDeployerTests(unittest.TestCase):
    def test_deploy_runtime_graph_success(self) -> None:
        async def _run() -> None:
            transport = _FakeTransport({"ok": True, "result": {}})

            async def _wait_ready(fake_transport: _FakeTransport, timeout_s: float) -> None:
                _ = (fake_transport, timeout_s)

            gateway = NatsRungraphGateway(RungraphDeployConfig(nats_url="nats://127.0.0.1:4222"))
            with patch("f8pystudio.bridge.rungraph_deployer.NatsTransport", return_value=transport):
                with patch("f8pystudio.bridge.rungraph_deployer.wait_service_ready", side_effect=_wait_ready):
                    result = await gateway.deploy_runtime_graph(
                        RungraphDeployRequest(service_id="svc_demo", graph=_FakeGraph())
                    )
            self.assertEqual(result.success, True)
            self.assertEqual(result.error_message, "")
            self.assertEqual(transport.connected, True)
            self.assertEqual(transport.closed, True)

        asyncio.run(_run())

    def test_deploy_runtime_graph_rejected(self) -> None:
        async def _run() -> None:
            transport = _FakeTransport({"ok": False, "error": {"message": "bad graph"}})

            async def _wait_ready(fake_transport: _FakeTransport, timeout_s: float) -> None:
                _ = (fake_transport, timeout_s)

            gateway = NatsRungraphGateway(RungraphDeployConfig(nats_url="nats://127.0.0.1:4222"))
            with patch("f8pystudio.bridge.rungraph_deployer.NatsTransport", return_value=transport):
                with patch("f8pystudio.bridge.rungraph_deployer.wait_service_ready", side_effect=_wait_ready):
                    result = await gateway.deploy_runtime_graph(
                        RungraphDeployRequest(service_id="svc_demo", graph=_FakeGraph())
                    )
            self.assertEqual(result.success, False)
            self.assertEqual(result.error_message, "bad graph")

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
