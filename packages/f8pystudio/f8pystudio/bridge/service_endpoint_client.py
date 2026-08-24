from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import msgspec
from f8pysdk.specs import (
    F8ActivateRequest,
    F8ActiveReply,
    F8CommandError,
    F8DeactivateRequest,
    F8EmptyArgs,
    F8SetStateArgs,
    F8SetStateReply,
    F8SetStateRequest,
    F8StatusReply,
    F8StatusRequest,
    F8TerminateReply,
    F8TerminateRequest,
)
from f8pysdk.f8_naming import ensure_token, new_id, svc_endpoint_key
from f8pysdk.codec import decode_as, decode_obj, encode_obj, validate_as

from .runtime_request import RuntimeRequester

logger = logging.getLogger(__name__)
_SERVICE_ENDPOINT_REQUEST_ERRORS = (Exception,)


@dataclass(frozen=True)
class SetStateRequestResult:
    accepted: bool
    rejected: bool
    reject_code: str
    reject_message: str


def message_data_bytes(message: Any) -> bytes:
    try:
        data = message.data
    except AttributeError:
        return b""
    try:
        return bytes(data or b"")
    except (TypeError, ValueError):
        return b""


def _error_message(error: F8CommandError | None | msgspec.UnsetType) -> str:
    if error is None or isinstance(error, msgspec.UnsetType):
        return ""
    return str(error.message or "")


def _error_code(error: F8CommandError | None | msgspec.UnsetType) -> str:
    if error is None or isinstance(error, msgspec.UnsetType):
        return ""
    return str(error.code.value)


def _status_identity_from_mapping(result: dict[str, Any]) -> dict[str, Any]:
    service_id = str(result.get("serviceId") or "").strip()
    service_class = str(result.get("serviceClass") or "").strip()
    runtime_instance_id = str(result.get("runtimeInstanceId") or "").strip()
    output: dict[str, Any] = {
        "alive": True,
        "identityValid": bool(service_class and runtime_instance_id),
    }
    if service_id:
        output["serviceId"] = service_id
    if service_class:
        output["serviceClass"] = service_class
    if runtime_instance_id:
        output["runtimeInstanceId"] = runtime_instance_id
    if "active" in result:
        output["active"] = bool(result.get("active"))
    return output


async def request_service_status(
    requester: RuntimeRequester,
    *,
    service_id: str,
    timeout_s: float = 0.4,
    attempts: int = 1,
    retry_sleep_s: float = 0.0,
) -> dict[str, Any] | None:
    sid = ensure_token(str(service_id), label="service_id")
    attempt_count = max(int(attempts), 1)
    for attempt_index in range(attempt_count):
        payload = encode_obj(
            F8StatusRequest(
                reqId=new_id(),
                args=F8EmptyArgs(),
                meta={"actor": "studio", "cmd": "status"},
            )
        )
        try:
            message = await requester.request(svc_endpoint_key(sid, "status"), payload, timeout=float(timeout_s))
        except _SERVICE_ENDPOINT_REQUEST_ERRORS as exc:
            logger.debug(
                "service status request failed service_id=%s attempt=%s/%s",
                service_id,
                attempt_index + 1,
                attempt_count,
                exc_info=exc,
            )
            if attempt_index + 1 < attempt_count:
                await asyncio.sleep(float(retry_sleep_s))
                continue
            return None
        raw = message_data_bytes(message)
        if not raw:
            return None
        try:
            response = decode_as(raw, F8StatusReply)
        except ValueError as exc:
            logger.debug("strict service status decode failed service_id=%s", service_id, exc_info=exc)
            try:
                fallback = decode_obj(raw)
            except ValueError:
                return None
            if not isinstance(fallback, dict):
                return None
            if not bool(fallback.get("ok")):
                return None
            result = fallback.get("result")
            if not isinstance(result, dict):
                return None
            return _status_identity_from_mapping(result)
        if not response.ok:
            return None
        result = response.result
        if result is None or isinstance(result, msgspec.UnsetType):
            return None
        output: dict[str, Any] = {
            "alive": True,
            "identityValid": True,
            "serviceId": str(result.serviceId),
            "serviceClass": str(result.serviceClass),
            "runtimeInstanceId": str(result.runtimeInstanceId),
        }
        output["active"] = bool(result.active)
        return output
    return None


async def request_set_service_active(
    requester: RuntimeRequester,
    *,
    service_id: str,
    active: bool,
    attempts: int,
    timeout_s: float,
    retry_sleep_s: float,
) -> bool:
    sid = ensure_token(str(service_id), label="service_id")
    cmd = "activate" if bool(active) else "deactivate"
    if cmd == "activate":
        payload = encode_obj(
            F8ActivateRequest(reqId=new_id(), args=F8EmptyArgs(), meta={"actor": "studio", "cmd": cmd})
        )
    else:
        payload = encode_obj(
            F8DeactivateRequest(reqId=new_id(), args=F8EmptyArgs(), meta={"actor": "studio", "cmd": cmd})
        )
    for _ in range(max(int(attempts), 1)):
        try:
            message = await requester.request(svc_endpoint_key(sid, cmd), payload, timeout=float(timeout_s))
            data = message_data_bytes(message)
            if data:
                response = decode_as(data, F8ActiveReply)
                if response.ok:
                    return True
        except _SERVICE_ENDPOINT_REQUEST_ERRORS as exc:
            logger.debug("set service active request failed service_id=%s active=%s", service_id, active, exc_info=exc)
            await asyncio.sleep(float(retry_sleep_s))
            continue
    return False


async def request_service_terminate(
    requester: RuntimeRequester,
    *,
    service_id: str,
    attempts: int,
    timeout_s: float,
    retry_sleep_s: float,
) -> bool:
    sid = ensure_token(str(service_id), label="service_id")
    key = svc_endpoint_key(sid, "terminate")
    payload = encode_obj(
        F8TerminateRequest(
            reqId=new_id(),
            args=F8EmptyArgs(),
            meta={"actor": "studio", "cmd": "terminate"},
        )
    )
    for _ in range(max(int(attempts), 1)):
        try:
            message = await requester.request(key, payload, timeout=float(timeout_s))
            raw = message_data_bytes(message)
            if not raw:
                continue
            response = decode_as(raw, F8TerminateReply)
            if response.ok:
                return True
            return False
        except _SERVICE_ENDPOINT_REQUEST_ERRORS as exc:
            logger.debug("service terminate request failed service_id=%s", service_id, exc_info=exc)
            await asyncio.sleep(float(retry_sleep_s))
            continue
    return False


async def request_set_remote_state(
    requester: RuntimeRequester,
    *,
    service_id: str,
    node_id: str,
    field: str,
    value: Any,
    attempts: int,
    timeout_s: float,
    retry_sleep_s: float,
) -> SetStateRequestResult:
    sid = ensure_token(str(service_id), label="service_id")
    nid = ensure_token(str(node_id), label="node_id")
    state_field = str(field or "").strip()
    if not state_field:
        return SetStateRequestResult(
            accepted=False,
            rejected=False,
            reject_code="",
            reject_message="",
        )
    payload = encode_obj(
        F8SetStateRequest(
            reqId=new_id(),
            args=validate_as(
                F8SetStateArgs,
                {"nodeId": nid, "field": state_field, "value": value},
            ),
            meta={"actor": "studio", "source": "ui"},
        )
    )
    key = svc_endpoint_key(sid, "set_state")
    for _ in range(max(int(attempts), 1)):
        try:
            message = await requester.request(key, payload, timeout=float(timeout_s))
            raw = message_data_bytes(message)
            if not raw:
                continue
            response = decode_as(raw, F8SetStateReply)
            if response.ok:
                return SetStateRequestResult(
                    accepted=True,
                    rejected=False,
                    reject_code="",
                    reject_message="",
                )
            if not response.ok:
                error = response.error
                return SetStateRequestResult(
                    accepted=False,
                    rejected=True,
                    reject_code=_error_code(error),
                    reject_message=_error_message(error),
                )
        except _SERVICE_ENDPOINT_REQUEST_ERRORS as exc:
            logger.debug(
                "set remote state request failed service_id=%s node_id=%s field=%s",
                service_id,
                node_id,
                field,
                exc_info=exc,
            )
            await asyncio.sleep(float(retry_sleep_s))
            continue
    return SetStateRequestResult(
        accepted=False,
        rejected=False,
        reject_code="",
        reject_message="",
    )


async def request_service_debug_data(
    requester: RuntimeRequester,
    *,
    service_id: str,
    node_id: str = "",
    port: str = "",
    limit: int = 100,
    include_value: bool = True,
    max_value_bytes: int = 65536,
    timeout_s: float = 1.0,
) -> dict[str, Any]:
    sid = ensure_token(str(service_id), label="service_id")
    node_id_s = str(node_id or "").strip()
    port_s = str(port or "").strip()
    if node_id_s:
        node_id_s = ensure_token(node_id_s, label="node_id")
    if port_s:
        port_s = ensure_token(port_s, label="port")
    payload = encode_obj(
        {
            "reqId": new_id(),
            "args": {
                "nodeId": node_id_s,
                "port": port_s,
                "limit": max(1, min(int(limit), 1000)),
                "includeValue": bool(include_value),
                "maxValueBytes": max(0, int(max_value_bytes)),
            },
            "meta": {"actor": "studio", "cmd": "debug_data"},
        }
    )
    message = await requester.request(svc_endpoint_key(sid, "debug_data"), payload, timeout=float(timeout_s))
    raw = message_data_bytes(message)
    if not raw:
        return {"ok": False, "result": {}, "error": "empty response"}
    try:
        response = decode_obj(raw)
    except ValueError as exc:
        return {"ok": False, "result": {}, "error": str(exc)}
    if not bool(response.get("ok")):
        error = response.get("error")
        if isinstance(error, dict):
            message_text = str(error.get("message") or "")
            code_text = str(error.get("code") or "")
            return {"ok": False, "result": {}, "error": message_text or code_text}
        return {"ok": False, "result": {}, "error": str(error or "debug_data failed")}
    result = response.get("result")
    if isinstance(result, dict):
        return {"ok": True, "result": result, "error": ""}
    return {"ok": False, "result": {}, "error": "debug_data response result must be an object"}
