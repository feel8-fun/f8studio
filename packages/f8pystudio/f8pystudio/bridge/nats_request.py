from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from f8pysdk.service_bus.codec import decode_obj, encode_obj


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
    Send request and parse MsgPack response.
    """
    raw_payload = encode_obj(req.payload)
    message = await nc.request(str(req.subject), raw_payload, timeout=float(req.timeout_s))
    raw = bytes(message.data or b"")
    if not raw:
        raise RuntimeError("empty response")
    try:
        return decode_obj(raw)
    except ValueError as exc:
        raise RuntimeError(f"response is not a valid MsgPack object: {exc}") from exc


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
