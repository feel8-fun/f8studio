from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol


class NatsRequester(Protocol):
    async def request(self, subject: str, payload: bytes, timeout: float) -> Any: ...


@dataclass(frozen=True)
class RequestJsonInput:
    subject: str
    payload: dict[str, Any]
    timeout_s: float


@dataclass(frozen=True)
class OkEnvelope:
    ok: bool
    result: dict[str, Any]
    error_message: str
    payload: dict[str, Any]


async def request_json(nc: NatsRequester, req: RequestJsonInput) -> dict[str, Any]:
    """
    Send request and parse JSON response.
    """
    raw_payload = json.dumps(req.payload, ensure_ascii=False, default=str).encode("utf-8")
    message = await nc.request(str(req.subject), raw_payload, timeout=float(req.timeout_s))
    raw = bytes(message.data or b"")
    if not raw:
        raise RuntimeError("empty response")
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("response is not a JSON object")
    return decoded


def parse_ok_envelope(payload: dict[str, Any]) -> OkEnvelope:
    """
    Parse unified `{ok,result,error}` response envelope.
    """
    ok = bool(payload.get("ok") is True)
    result = payload.get("result")
    err = payload.get("error")

    result_obj = result if isinstance(result, dict) else {}
    if ok:
        return OkEnvelope(ok=True, result=result_obj, error_message="", payload=payload)

    err_obj = err if isinstance(err, dict) else {}
    msg = str(err_obj.get("message") or "").strip()
    return OkEnvelope(ok=False, result=result_obj, error_message=msg, payload=payload)


def parse_ok_response(payload: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
    """
    Backward-compatible tuple adapter for older call sites.
    """
    parsed = parse_ok_envelope(payload)
    return parsed.ok, parsed.result, parsed.error_message
