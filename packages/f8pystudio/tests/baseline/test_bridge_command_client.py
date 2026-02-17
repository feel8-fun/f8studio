from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import dataclass

from _bootstrap import ensure_package_importable

ensure_package_importable()

from f8pystudio.bridge.command_client import CommandRequest, NatsCommandGateway


@dataclass
class _Msg:
    data: bytes


class _FakeNatsClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.last_subject = ""

    async def request(self, subject: str, payload: bytes, timeout: float) -> _Msg:
        _ = (payload, timeout)
        self.last_subject = subject
        return _Msg(data=json.dumps(self.payload).encode("utf-8"))


class _TestGateway(NatsCommandGateway):
    def __init__(self, fake_client: _FakeNatsClient) -> None:
        super().__init__(nats_url="nats://127.0.0.1:4222")
        self._fake_client = fake_client

    async def ensure_connected(self) -> _FakeNatsClient:
        return self._fake_client


class BridgeCommandClientTests(unittest.TestCase):
    def test_request_command_success(self) -> None:
        async def _run() -> None:
            fake_client = _FakeNatsClient({"ok": True, "result": {"hello": "world"}})
            gateway = _TestGateway(fake_client)
            response = await gateway.request_command(
                CommandRequest(service_id="svc_demo", call="ping", args={"x": 1})
            )
            self.assertEqual(response.ok, True)
            self.assertEqual(response.result, {"hello": "world"})
            self.assertIn(".cmd", fake_client.last_subject)

        asyncio.run(_run())

    def test_request_command_rejected(self) -> None:
        async def _run() -> None:
            fake_client = _FakeNatsClient({"ok": False, "error": {"message": "rejected"}})
            gateway = _TestGateway(fake_client)
            response = await gateway.request_command(
                CommandRequest(service_id="svc_demo", call="ping", args={})
            )
            self.assertEqual(response.ok, False)
            self.assertEqual(response.error_message, "rejected")

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
