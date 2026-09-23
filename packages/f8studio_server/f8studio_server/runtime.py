from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Protocol, cast

import msgspec

from f8pysdk.bus import BusBackend
from f8pysdk.codec import decode_as, decode_obj, encode_obj
from f8pysdk.f8_naming import cmd_channel_key, ensure_token, new_id, svc_endpoint_key
from f8pysdk.rungraph_fingerprint import build_rungraph_deploy_fingerprint
from f8pysdk.runtime_transport import RuntimeTransport
from f8pysdk.service_runtime_tools.deploy.readiness import (
    RungraphDeployStatusTimeout,
    wait_rungraph_deploy_status,
)
from f8pysdk.specs import (
    F8ActivateRequest,
    F8ActiveReply,
    F8CommandError,
    F8CommandInvokeReply,
    F8CommandInvokeRequest,
    F8DeactivateRequest,
    F8EmptyArgs,
    F8JsonValue,
    F8RuntimeGraph,
    F8SetRungraphArgs,
    F8SetRungraphReply,
    F8SetRungraphRequest,
    F8SetStateArgs,
    F8SetStateReply,
    F8SetStateRequest,
    F8StatusReply,
    F8StatusRequest,
    F8TerminateReply,
    F8TerminateRequest,
)
from f8pysdk.zenoh_transport import ZenohTransport, ZenohTransportConfig
from f8pysdk.zenoh_naming import zenoh_state_key

from .models import RuntimeActionResult, RuntimeStateField, ServiceDeployResult, ServiceRuntimeStatus


RuntimeMonitorCallback = Callable[[str, bytes], Awaitable[None]]


class RuntimeSubscription(Protocol):
    async def unsubscribe(self) -> None: ...


class RuntimeGateway(Protocol):
    async def start_monitoring(self, callback: RuntimeMonitorCallback) -> None: ...

    async def deploy(
        self,
        *,
        service_id: str,
        graph: F8RuntimeGraph,
        force_apply: bool,
    ) -> ServiceDeployResult: ...

    async def status(self, service_id: str) -> ServiceRuntimeStatus: ...

    async def set_active(self, service_id: str, *, active: bool) -> RuntimeActionResult: ...

    async def set_state(
        self,
        service_id: str,
        *,
        node_id: str,
        field: str,
        value: F8JsonValue,
    ) -> RuntimeActionResult: ...

    async def read_state(self, service_id: str, *, node_id: str, field: str) -> RuntimeStateField: ...

    async def invoke_command(
        self,
        service_id: str,
        *,
        call: str,
        params: dict[str, F8JsonValue],
    ) -> RuntimeActionResult: ...

    async def terminate(self, service_id: str) -> RuntimeActionResult: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class RuntimeConfig:
    bus_backend: BusBackend = "zenoh"
    client_service_id: str = "webstudio"
    zenoh_config_path: str | None = None
    zenoh_connect: tuple[str, ...] = ()
    zenoh_listen: tuple[str, ...] = ()
    zenoh_shm_pool_bytes: int = 256 * 1024 * 1024
    endpoint_ready_timeout_s: float = 4.0
    request_timeout_s: float = 1.0
    request_attempts: int = 3
    deploy_timeout_s: float = 15.0


def _error_message(error: F8CommandError | None | msgspec.UnsetType) -> str:
    if error is None or isinstance(error, msgspec.UnsetType):
        return ""
    return str(error.message)


@dataclass
class ZenohRuntimeGateway:
    config: RuntimeConfig = field(default_factory=RuntimeConfig)
    _transport: RuntimeTransport | None = field(default=None, init=False, repr=False)
    _monitor_subscription: RuntimeSubscription | None = field(default=None, init=False, repr=False)
    _state_subscription: RuntimeSubscription | None = field(default=None, init=False, repr=False)
    _state_values: dict[str, bytes] = field(default_factory=dict, init=False, repr=False)
    _connect_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def _build_transport(self) -> RuntimeTransport:
        if self.config.bus_backend == "mem":
            from f8pysdk.testing import InMemoryCluster, InMemoryTransport

            return InMemoryTransport(cluster=InMemoryCluster())
        if self.config.bus_backend != "zenoh":
            raise ValueError("runtime gateway supports only zenoh or mem bus backends")
        return ZenohTransport(
            ZenohTransportConfig(
                service_id=self.config.client_service_id,
                config_path=self.config.zenoh_config_path,
                connect=self.config.zenoh_connect,
                listen=self.config.zenoh_listen,
                shm_pool_bytes=self.config.zenoh_shm_pool_bytes,
            )
        )

    async def _connected_transport(self) -> RuntimeTransport:
        transport = self._transport
        if transport is not None:
            return transport
        async with self._connect_lock:
            transport = self._transport
            if transport is not None:
                return transport
            transport = self._build_transport()
            await transport.connect()
            self._transport = transport
            return transport

    async def close(self) -> None:
        async with self._connect_lock:
            transport = self._transport
            self._transport = None
            monitor_subscription = self._monitor_subscription
            self._monitor_subscription = None
            state_subscription = self._state_subscription
            self._state_subscription = None
            self._state_values.clear()
        if monitor_subscription is not None:
            await monitor_subscription.unsubscribe()
        if state_subscription is not None:
            await state_subscription.unsubscribe()
        if transport is not None:
            await transport.close()

    async def start_monitoring(self, callback: RuntimeMonitorCallback) -> None:
        transport = await self._connected_transport()
        if self._monitor_subscription is None:
            subscription = await transport.subscribe(
                "f8/svc/*/nodes/*/data/monitor",
                cb=callback,
            )
            self._monitor_subscription = cast(RuntimeSubscription, subscription)
        if self._state_subscription is None:
            state_subscription = await transport.retained_watch(
                "f8/svc/*/state/nodes/*/state/**",
                cb=self._ingest_state,
                with_initial=True,
            )
            self._state_subscription = cast(RuntimeSubscription, state_subscription)

    async def _ingest_state(self, key: str, payload: bytes) -> None:
        self._state_values[key] = bytes(payload)

    async def _request(self, key: str, payload: bytes, *, timeout_s: float | None = None) -> bytes:
        transport = await self._connected_transport()
        raw = await transport.request(
            key,
            payload,
            timeout=self.config.request_timeout_s if timeout_s is None else timeout_s,
            raise_on_error=True,
        )
        if not raw:
            raise RuntimeError(f"empty runtime response from {key}")
        return raw

    async def status(self, service_id: str) -> ServiceRuntimeStatus:
        service_id = ensure_token(service_id, label="service_id")
        request = F8StatusRequest(
            reqId=new_id(),
            args=F8EmptyArgs(),
            meta={"actor": "webstudio", "source": "api"},
        )
        response = decode_as(
            await self._request(svc_endpoint_key(service_id, "status"), encode_obj(request)),
            F8StatusReply,
        )
        if not response.ok or response.result is None or isinstance(response.result, msgspec.UnsetType):
            raise RuntimeError(_error_message(response.error) or f"status rejected by {service_id}")
        result = response.result
        return ServiceRuntimeStatus(
            service_id=str(result.serviceId),
            service_class=str(result.serviceClass),
            runtime_instance_id=str(result.runtimeInstanceId),
            active=bool(result.active),
            rungraph_graph_id="" if isinstance(result.rungraphGraphId, msgspec.UnsetType) else str(result.rungraphGraphId),
            rungraph_revision="" if isinstance(result.rungraphRevision, msgspec.UnsetType) else str(result.rungraphRevision),
            rungraph_fingerprint=(
                "" if isinstance(result.rungraphFingerprint, msgspec.UnsetType) else str(result.rungraphFingerprint)
            ),
        )

    async def _wait_until_ready(self, service_id: str) -> ServiceRuntimeStatus:
        deadline = asyncio.get_running_loop().time() + self.config.endpoint_ready_timeout_s
        last_error = ""
        while True:
            try:
                return await self.status(service_id)
            except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"service endpoint not ready: {service_id}: {last_error}")
            await asyncio.sleep(0.1)

    async def deploy(
        self,
        *,
        service_id: str,
        graph: F8RuntimeGraph,
        force_apply: bool,
    ) -> ServiceDeployResult:
        service_id = ensure_token(service_id, label="service_id")
        try:
            status = await self._wait_until_ready(service_id)
        except TimeoutError as exc:
            return ServiceDeployResult(service_id=service_id, success=False, error_message=str(exc))
        target_fingerprint = build_rungraph_deploy_fingerprint(graph)
        if not force_apply and status.rungraph_fingerprint == target_fingerprint:
            return ServiceDeployResult(service_id=service_id, success=True)

        request_id = new_id()
        request = F8SetRungraphRequest(
            reqId=request_id,
            args=F8SetRungraphArgs(graph=graph),
            meta={
                "actor": "webstudio",
                "source": f"webstudio:{request_id}",
                "targetFingerprint": target_fingerprint,
                "forceApply": force_apply,
            },
        )
        last_error = ""
        for attempt in range(max(1, self.config.request_attempts)):
            try:
                response = decode_as(
                    await self._request(
                        svc_endpoint_key(service_id, "set_rungraph"),
                        encode_obj(request),
                    ),
                    F8SetRungraphReply,
                )
            except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt + 1 < self.config.request_attempts:
                    await asyncio.sleep(0.15)
                continue
            if not response.ok:
                return ServiceDeployResult(
                    service_id=service_id,
                    success=False,
                    error_message=_error_message(response.error) or "set_rungraph rejected",
                )
            break
        else:
            return ServiceDeployResult(
                service_id=service_id,
                success=False,
                error_message=f"set_rungraph request failed: {last_error}",
            )

        transport = await self._connected_transport()
        try:
            final = await wait_rungraph_deploy_status(
                transport,
                service_id=service_id,
                req_id=request_id,
                graph_id=str(graph.graphId),
                revision=str(graph.revision),
                target_fingerprint=target_fingerprint,
                expected_runtime_instance_id=status.runtime_instance_id if force_apply else "",
                timeout_s=self.config.deploy_timeout_s,
            )
        except RungraphDeployStatusTimeout as exc:
            return ServiceDeployResult(service_id=service_id, success=False, error_message=str(exc))
        return ServiceDeployResult(
            service_id=service_id,
            success=final.ok,
            error_message="" if final.ok else final.error_message or "rungraph apply failed",
        )

    async def set_active(self, service_id: str, *, active: bool) -> RuntimeActionResult:
        service_id = ensure_token(service_id, label="service_id")
        request_id = new_id()
        request: F8ActivateRequest | F8DeactivateRequest
        endpoint: str
        if active:
            endpoint = "activate"
            request = F8ActivateRequest(reqId=request_id, args=F8EmptyArgs(), meta={"actor": "webstudio"})
        else:
            endpoint = "deactivate"
            request = F8DeactivateRequest(reqId=request_id, args=F8EmptyArgs(), meta={"actor": "webstudio"})
        response = decode_as(
            await self._request(svc_endpoint_key(service_id, endpoint), encode_obj(request)),
            F8ActiveReply,
        )
        return RuntimeActionResult(
            success=response.ok,
            result={"active": active} if response.ok else None,
            error_message="" if response.ok else _error_message(response.error) or f"{endpoint} rejected",
        )

    async def set_state(
        self,
        service_id: str,
        *,
        node_id: str,
        field: str,
        value: F8JsonValue,
    ) -> RuntimeActionResult:
        service_id = ensure_token(service_id, label="service_id")
        node_id = ensure_token(node_id, label="node_id")
        if not field.strip():
            raise ValueError("state field must be non-empty")
        request = F8SetStateRequest(
            reqId=new_id(),
            args=F8SetStateArgs(nodeId=node_id, field=field, value=value),
            meta={"actor": "webstudio", "source": "api"},
        )
        response = decode_as(
            await self._request(svc_endpoint_key(service_id, "set_state"), encode_obj(request)),
            F8SetStateReply,
        )
        return RuntimeActionResult(
            success=response.ok,
            result={"nodeId": node_id, "field": field} if response.ok else None,
            error_message="" if response.ok else _error_message(response.error) or "set_state rejected",
        )

    async def read_state(self, service_id: str, *, node_id: str, field: str) -> RuntimeStateField:
        service_id = ensure_token(service_id, label="service_id")
        node_id = ensure_token(node_id, label="node_id")
        normalized_field = field.strip()
        if not normalized_field:
            raise ValueError("state field must be non-empty")
        await self._connected_transport()
        raw = self._state_values.get(zenoh_state_key(service_id, node_id=node_id, field=normalized_field))
        if raw is None:
            return RuntimeStateField(field=normalized_field, found=False)
        decoded = decode_obj(raw)
        if "value" not in decoded:
            raise ValueError(
                f"invalid retained state envelope service_id={service_id} "
                f"node_id={node_id} field={normalized_field}"
            )
        timestamp = decoded.get("tsMs", decoded.get("ts", decoded.get("ts_ms")))
        ts_ms = int(timestamp) if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool) else None
        return RuntimeStateField(
            field=normalized_field,
            found=True,
            value=cast(F8JsonValue, decoded["value"]),
            ts_ms=ts_ms,
        )

    async def invoke_command(
        self,
        service_id: str,
        *,
        call: str,
        params: dict[str, F8JsonValue],
    ) -> RuntimeActionResult:
        service_id = ensure_token(service_id, label="service_id")
        if not call.strip():
            raise ValueError("command call must be non-empty")
        request = F8CommandInvokeRequest(
            reqId=new_id(),
            call=call,
            args=params,
            meta={"actor": "webstudio", "source": "api"},
        )
        response = decode_as(
            await self._request(cmd_channel_key(service_id), encode_obj(request), timeout_s=2.0),
            F8CommandInvokeReply,
        )
        return RuntimeActionResult(
            success=response.ok,
            result=response.result if response.ok else None,
            error_message="" if response.ok else _error_message(response.error) or "command rejected",
        )

    async def terminate(self, service_id: str) -> RuntimeActionResult:
        service_id = ensure_token(service_id, label="service_id")
        request = F8TerminateRequest(reqId=new_id(), args=F8EmptyArgs(), meta={"actor": "webstudio"})
        response = decode_as(
            await self._request(svc_endpoint_key(service_id, "terminate"), encode_obj(request)),
            F8TerminateReply,
        )
        return RuntimeActionResult(
            success=response.ok,
            result={"terminating": True} if response.ok else None,
            error_message="" if response.ok else _error_message(response.error) or "terminate rejected",
        )


__all__ = [
    "RuntimeConfig",
    "RuntimeGateway",
    "RuntimeMonitorCallback",
    "RuntimeSubscription",
    "ZenohRuntimeGateway",
]
