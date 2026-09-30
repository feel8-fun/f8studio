from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from ...codec import decode_obj
from ...runtime_transport import RuntimeTransport, SubscriptionHandle
from ...time_utils import now_ms
from ...f8_naming import ensure_token


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RungraphDeployStatus:
    service_id: str
    req_id: str
    graph_id: str
    revision: str
    phase: str
    ok: bool
    error_message: str
    ts_ms: int
    target_fingerprint: str = ""
    applied_fingerprint: str = ""
    runtime_instance_id: str = ""


class RungraphDeployStatusTimeout(Exception):
    """Raised when a rungraph deploy request reports progress but no final status."""


RUNGRAPH_DEPLOY_PROTOCOL = "f8.rungraphDeployStatus/1"
RUNGRAPH_DEPLOY_PROTOCOL_V2 = "f8.rungraphDeployStatus/2"


def rungraph_deploy_status_key(service_id: str) -> str:
    service_id_s = ensure_token(service_id, label="service_id")
    return f"f8/svc/{service_id_s}/status/rungraph"


def rungraph_deploy_request_status_key(service_id: str, req_id: str) -> str:
    service_id_s = ensure_token(service_id, label="service_id")
    req_id_s = str(req_id or "").strip()
    if not req_id_s:
        raise ValueError("req_id is required")
    req_hash = hashlib.sha256(req_id_s.encode("utf-8")).hexdigest()
    return f"f8/svc/{service_id_s}/status/rungraph/requests/{req_hash}"


def _decode_payload_or_empty(raw: bytes | None) -> Any:
    if not raw:
        return {}
    try:
        return decode_obj(raw)
    except ValueError:
        return {}


async def _retained_payload_or_empty(tr: RuntimeTransport, key: str, *, context: str) -> Any:
    try:
        raw = await tr.retained_get(key)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        log.debug("%s retained_get failed key=%s", context, key, exc_info=exc)
        return {}
    return _decode_payload_or_empty(raw)


async def _stop_retained_watch(watch: SubscriptionHandle, *, key: str, context: str) -> None:
    try:
        await watch.unsubscribe()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        log.error("%s watch stop failed key=%s", context, key, exc_info=exc)


async def wait_service_ready(
    tr: RuntimeTransport,
    *,
    timeout_s: float = 6.0,
    min_ts_ms: int | None = None,
    max_age_ms: int | None = None,
    service_id: str | None = None,
) -> None:
    """
    Wait until a service announces readiness.

    Readiness is published as a retained status sample at
    `f8/svc/{service_id}/status/ready`. This function waits with a retained
    watch after an initial read.
    """
    if service_id is None:
        raise ValueError("service_id is required")
    service_id_s = ensure_token(service_id, label="service_id")
    min_ts = int(min_ts_ms) if min_ts_ms is not None else None
    max_age = int(max_age_ms) if max_age_ms is not None else None

    def _accept(payload: Any) -> bool:
        if not isinstance(payload, dict) or payload.get("ready") is not True:
            return False
        try:
            ts = int(payload.get("ts") or 0)
        except (TypeError, ValueError):
            ts = 0
        if min_ts is not None and ts < min_ts:
            return False
        if max_age is not None:
            age = int(now_ms()) - ts
            if ts <= 0 or age > max_age:
                return False
        return True

    key = f"f8/svc/{service_id_s}/status/ready"
    if _accept(await _retained_payload_or_empty(tr, key, context="initial ready")):
        return

    loop = asyncio.get_running_loop()
    fut: asyncio.Future[None] = loop.create_future()

    async def _on_ready(_key: str, value: bytes) -> None:
        if fut.done():
            return
        try:
            payload = decode_obj(value or b"")
        except ValueError:
            payload = {}
        if _accept(payload):
            fut.set_result(None)

    watch = None
    try:
        watch = await tr.retained_watch(key, cb=_on_ready, with_initial=True)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        log.debug("ready retained_watch failed key=%s", key, exc_info=exc)
        watch = None

    try:
        if _accept(await _retained_payload_or_empty(tr, key, context="second ready")):
            return
        await asyncio.wait_for(fut, timeout=float(timeout_s))
    finally:
        if watch is not None:
            await _stop_retained_watch(watch, key=key, context="ready")


def _rungraph_status_from_payload(payload: Any) -> RungraphDeployStatus | None:
    if not isinstance(payload, dict):
        return None
    service_id = str(payload.get("serviceId") or "").strip()
    req_id = str(payload.get("reqId") or "").strip()
    graph_id = str(payload.get("graphId") or "").strip()
    revision = str(payload.get("revision") or "").strip()
    phase = str(payload.get("phase") or "").strip()
    error_message = str(payload.get("errorMessage") or "").strip()
    target_fingerprint = str(payload.get("targetFingerprint") or "").strip()
    applied_fingerprint = str(payload.get("appliedFingerprint") or "").strip()
    runtime_instance_id = str(payload.get("runtimeInstanceId") or "").strip()
    try:
        ts_ms = int(payload.get("ts") or 0)
    except (TypeError, ValueError):
        ts_ms = 0
    return RungraphDeployStatus(
        service_id=service_id,
        req_id=req_id,
        graph_id=graph_id,
        revision=revision,
        phase=phase,
        ok=bool(payload.get("ok") is True),
        error_message=error_message,
        ts_ms=ts_ms,
        target_fingerprint=target_fingerprint,
        applied_fingerprint=applied_fingerprint,
        runtime_instance_id=runtime_instance_id,
    )


async def wait_rungraph_deploy_status(
    tr: RuntimeTransport,
    *,
    service_id: str,
    req_id: str,
    graph_id: str = "",
    revision: str = "",
    target_fingerprint: str = "",
    expected_runtime_instance_id: str = "",
    timeout_s: float = 15.0,
) -> RungraphDeployStatus:
    """
    Wait for the retained final status of an accepted rungraph deployment.

    `set_rungraph` request/reply acknowledges acceptance only. The final
    deployment state is published at `f8/svc/{serviceId}/status/rungraph`.
    """
    service_id_s = ensure_token(service_id, label="service_id")
    req_id_s = str(req_id or "").strip()
    if not req_id_s:
        raise ValueError("req_id is required")
    graph_id_s = str(graph_id or "").strip()
    revision_s = str(revision or "").strip()
    target_fingerprint_s = str(target_fingerprint or "").strip()
    expected_runtime_instance_id_s = str(expected_runtime_instance_id or "").strip()
    key = rungraph_deploy_request_status_key(service_id_s, req_id_s)
    last_status: RungraphDeployStatus | None = None

    def _accept(payload: Any) -> RungraphDeployStatus | None:
        nonlocal last_status
        status = _rungraph_status_from_payload(payload)
        if status is None:
            return None
        if status.service_id != service_id_s:
            return None
        if status.req_id != req_id_s:
            return None
        if graph_id_s and status.graph_id != graph_id_s:
            return None
        if revision_s and status.revision != revision_s:
            return None
        if target_fingerprint_s and status.target_fingerprint and status.target_fingerprint != target_fingerprint_s:
            return None
        if target_fingerprint_s and not status.target_fingerprint:
            return None
        if target_fingerprint_s and status.applied_fingerprint != target_fingerprint_s:
            return None
        if expected_runtime_instance_id_s and status.runtime_instance_id != expected_runtime_instance_id_s:
            return None
        last_status = status
        if status.phase not in ("applied", "failed"):
            return None
        return status

    status = _accept(await _retained_payload_or_empty(tr, key, context="initial rungraph status"))
    if status is not None:
        return status

    loop = asyncio.get_running_loop()
    fut: asyncio.Future[RungraphDeployStatus] = loop.create_future()

    async def _on_status(_key: str, value: bytes) -> None:
        if fut.done():
            return
        try:
            payload = decode_obj(value or b"")
        except ValueError:
            payload = {}
        status = _accept(payload)
        if status is not None:
            fut.set_result(status)

    watch = None
    try:
        watch = await tr.retained_watch(key, cb=_on_status, with_initial=True)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        log.debug("rungraph status retained_watch failed key=%s", key, exc_info=exc)
        watch = None

    try:
        status2 = _accept(await _retained_payload_or_empty(tr, key, context="second rungraph status"))
        if status2 is not None:
            return status2
        try:
            return await asyncio.wait_for(fut, timeout=float(timeout_s))
        except asyncio.TimeoutError as exc:
            if last_status is None:
                raise RungraphDeployStatusTimeout(
                    f"rungraph apply status not received within {float(timeout_s):g}s (key={key})"
                ) from exc
            raise RungraphDeployStatusTimeout(
                f"rungraph apply status not final within {float(timeout_s):g}s "
                f"(last phase={last_status.phase or '<empty>'}, key={key})"
            ) from exc
    finally:
        if watch is not None:
            await _stop_retained_watch(watch, key=key, context="rungraph status")


__all__ = [
    "RUNGRAPH_DEPLOY_PROTOCOL",
    "RungraphDeployStatus",
    "RungraphDeployStatusTimeout",
    "rungraph_deploy_request_status_key",
    "rungraph_deploy_status_key",
    "wait_rungraph_deploy_status",
    "wait_service_ready",
]
