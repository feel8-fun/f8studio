from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import dataclass
from typing import Any

from _bootstrap import ensure_package_importable

ensure_package_importable()

from f8pystudio.bridge.nats_request import (
    RequestJsonInput,
    parse_ok_envelope,
    request_json,
)


@dataclass
class _Msg:
    data: bytes


class _FakeNatsClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def request(self, subject: str, payload: bytes, timeout: float) -> _Msg:
        _ = (subject, payload, timeout)
        return _Msg(data=json.dumps(self._payload).encode("utf-8"))


class _FakeEmptyNatsClient:
    async def request(self, subject: str, payload: bytes, timeout: float) -> _Msg:
        _ = (subject, payload, timeout)
        return _Msg(data=b"")


class BridgeNatsRequestTests(unittest.TestCase):
    def test_request_json_success(self) -> None:
        async def _run() -> None:
            client = _FakeNatsClient({"ok": True, "result": {"active": True}})
            decoded = await request_json(
                client,
                RequestJsonInput(
                    subject="svc.test.status",
                    payload={"reqId": "1"},
                    timeout_s=0.2,
                ),
            )
            self.assertEqual(decoded["ok"], True)
            self.assertEqual(decoded["result"]["active"], True)

        asyncio.run(_run())

    def test_request_json_empty_response_raises(self) -> None:
        async def _run() -> None:
            client = _FakeEmptyNatsClient()
            with self.assertRaisesRegex(RuntimeError, "empty response"):
                await request_json(
                    client,
                    RequestJsonInput(
                        subject="svc.test.status",
                        payload={"reqId": "1"},
                        timeout_s=0.2,
                    ),
                )

        asyncio.run(_run())

    def test_parse_ok_envelope(self) -> None:
        ok = parse_ok_envelope({"ok": True, "result": {"a": 1}})
        self.assertEqual(ok.ok, True)
        self.assertEqual(ok.result, {"a": 1})

        rejected = parse_ok_envelope({"ok": False, "error": {"message": "bad"}})
        self.assertEqual(rejected.ok, False)
        self.assertEqual(rejected.error_message, "bad")


if __name__ == "__main__":
    unittest.main()
