from __future__ import annotations

import asyncio
from typing import Any

from f8pysdk.f8_naming import svc_endpoint_key
from f8pysdk.codec import decode_obj, encode_obj

from f8pystudio.bridge.service_endpoint_client import (
    request_service_status,
    request_service_terminate,
    request_set_remote_state,
    request_set_service_active,
)


class _FakeMessage:
    def __init__(self, data: bytes) -> None:
        self.data = data


class _FakeRequester:
    def __init__(self, scripted: list[object]) -> None:
        self._scripted = list(scripted)
        self.calls: list[tuple[str, bytes, float]] = []

    async def request(self, key: str, payload: bytes, timeout: float) -> Any:
        self.calls.append((str(key), bytes(payload), float(timeout)))
        if not self._scripted:
            raise AssertionError("unexpected request call")
        current = self._scripted.pop(0)
        if isinstance(current, Exception):
            raise current
        if isinstance(current, bytes):
            return _FakeMessage(current)
        raise AssertionError("script item must be bytes or Exception")


def test_request_service_status_success() -> None:
    requester = _FakeRequester(
        [
            encode_obj(
                {
                    "reqId": "r1",
                    "ok": True,
                    "result": {
                        "serviceId": "svc_demo",
                        "serviceClass": "f8.tests.demo",
                        "runtimeInstanceId": "inst_demo",
                        "active": False,
                    },
                    "error": None,
                }
            )
        ]
    )

    result = asyncio.run(request_service_status(requester, service_id="svc_demo", timeout_s=0.4))

    assert result == {
        "alive": True,
        "identityValid": True,
        "serviceId": "svc_demo",
        "serviceClass": "f8.tests.demo",
        "runtimeInstanceId": "inst_demo",
        "active": False,
    }
    assert requester.calls[0][0] == svc_endpoint_key("svc_demo", "status")


def test_request_service_status_marks_old_protocol_identity_invalid() -> None:
    requester = _FakeRequester(
        [encode_obj({"reqId": "r1", "ok": True, "result": {"serviceId": "svc_demo", "active": False}, "error": None})]
    )

    result = asyncio.run(request_service_status(requester, service_id="svc_demo", timeout_s=0.4))

    assert result == {
        "alive": True,
        "identityValid": False,
        "serviceId": "svc_demo",
        "active": False,
    }
    assert requester.calls[0][0] == svc_endpoint_key("svc_demo", "status")


def test_request_service_status_fallback_preserves_identity() -> None:
    requester = _FakeRequester(
        [
            encode_obj(
                {
                    "reqId": "r1",
                    "ok": True,
                    "result": {
                        "serviceId": "svc_demo",
                        "serviceClass": "f8.tests.demo",
                        "runtimeInstanceId": "inst_demo",
                        "futureField": "ignored",
                    },
                    "error": None,
                    "futureWrapperField": "ignored",
                }
            )
        ]
    )

    result = asyncio.run(request_service_status(requester, service_id="svc_demo", timeout_s=0.4))

    assert result == {
        "alive": True,
        "identityValid": True,
        "serviceId": "svc_demo",
        "serviceClass": "f8.tests.demo",
        "runtimeInstanceId": "inst_demo",
    }
    assert requester.calls[0][0] == svc_endpoint_key("svc_demo", "status")


def test_request_service_status_retries_after_exception() -> None:
    requester = _FakeRequester(
        [
            TimeoutError("transient"),
            encode_obj(
                {
                    "reqId": "r2",
                    "ok": True,
                    "result": {
                        "serviceId": "svc_demo",
                        "serviceClass": "f8.tests.demo",
                        "runtimeInstanceId": "inst_demo",
                        "active": True,
                    },
                    "error": None,
                }
            ),
        ]
    )

    result = asyncio.run(
        request_service_status(
            requester,
            service_id="svc_demo",
            timeout_s=0.4,
            attempts=2,
            retry_sleep_s=0.0,
        )
    )

    assert result is not None
    assert result["identityValid"] is True
    assert result["active"] is True
    assert len(requester.calls) == 2
    assert requester.calls[0][0] == svc_endpoint_key("svc_demo", "status")


def test_request_set_service_active_retries_after_exception() -> None:
    requester = _FakeRequester(
        [
            RuntimeError("transient"),
            encode_obj({"reqId": "r1", "ok": True, "result": {"active": True}, "error": None}),
        ]
    )

    ok = asyncio.run(
            request_set_service_active(
                requester,
                service_id="svc_demo",
                active=True,
                attempts=2,
                timeout_s=0.5,
                retry_sleep_s=0.0,
            )
    )

    assert ok is True
    assert len(requester.calls) == 2
    assert requester.calls[0][0] == svc_endpoint_key("svc_demo", "activate")


def test_request_service_terminate_reject_stops_retry() -> None:
    requester = _FakeRequester(
        [
            encode_obj(
                {"reqId": "r1", "ok": False, "result": None, "error": {"code": "INTERNAL", "message": "rejected"}}
            ),
            encode_obj({"reqId": "r2", "ok": True, "result": {"terminating": True}, "error": None}),
        ]
    )

    ok = asyncio.run(
            request_service_terminate(
                requester,
                service_id="svc_demo",
                attempts=2,
                timeout_s=0.4,
                retry_sleep_s=0.0,
            )
    )

    assert ok is False
    assert len(requester.calls) == 1
    assert requester.calls[0][0] == svc_endpoint_key("svc_demo", "terminate")


def test_request_set_remote_state_returns_reject_details() -> None:
    requester = _FakeRequester(
        [
            encode_obj(
                {
                    "reqId": "r1",
                    "ok": False,
                    "result": None,
                    "error": {"code": "INVALID_ARGS", "message": "invalid field"},
                }
            ),
        ]
    )

    result = asyncio.run(
            request_set_remote_state(
                requester,
                service_id="svc_demo",
                node_id="node_demo",
                field="gain",
                value=1,
                attempts=3,
                timeout_s=0.5,
            retry_sleep_s=0.0,
        )
    )

    assert result.accepted is False
    assert result.rejected is True
    assert result.reject_code == "INVALID_ARGS"
    assert result.reject_message == "invalid field"
    assert requester.calls[0][0] == svc_endpoint_key("svc_demo", "set_state")
    request_payload = decode_obj(requester.calls[0][1])
    assert request_payload["args"]["field"] == "gain"
