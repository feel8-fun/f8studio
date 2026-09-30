from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ...generated.control import ControlEndpoint
from ...codec import encode_obj
from ...generated import Code, F8CommandError, F8CommandInvokeReply
from ...f8_naming import cmd_channel_key, svc_endpoint_key
from .micro import ServiceBusControlHandlers

log = logging.getLogger(__name__)


class ServiceControlEndpointServer(Protocol):
    async def start(self) -> Any: ...

    async def stop(self) -> None: ...


@dataclass
class _TransportEndpointRequest:
    data: bytes
    response: bytes | None = None

    async def respond(self, payload: bytes) -> None:
        self.response = bytes(payload)


class RuntimeTransportServiceControlEndpointServer:
    """
    Backend-neutral control endpoint server using RuntimeTransport request/serve.

    It reuses the canonical request handlers so Zenoh command streams preserve
    identical wire payload semantics across Python and C++ services.
    """

    def __init__(self, bus: Any) -> None:
        self._bus = bus
        self._handlers = ServiceBusControlHandlers(bus)
        self._handles: list[Any] = []

    async def start(self) -> Any:
        sid = str(self._bus.service_id)
        registrations: tuple[tuple[str, Callable[[Any], Awaitable[None]]], ...] = (
            (svc_endpoint_key(sid, ControlEndpoint.activate), self._handlers._activate),
            (svc_endpoint_key(sid, ControlEndpoint.deactivate), self._handlers._deactivate),
            (svc_endpoint_key(sid, ControlEndpoint.set_active), self._handlers._set_active),
            (svc_endpoint_key(sid, ControlEndpoint.status), self._handlers._status),
            (svc_endpoint_key(sid, ControlEndpoint.terminate), self._handlers._terminate),
            (svc_endpoint_key(sid, ControlEndpoint.quit), self._handlers._terminate),
            (cmd_channel_key(sid), self._handlers._cmd),
            (svc_endpoint_key(sid, ControlEndpoint.set_state), self._handlers._set_state),
            (svc_endpoint_key(sid, ControlEndpoint.set_rungraph), self._handlers._set_rungraph),
            (svc_endpoint_key(sid, ControlEndpoint.debug_data), self._handlers._debug_data),
        )
        for key, handler in registrations:
            self._handles.append(await self._bus._transport.serve(key, self._wrap_handler(key, handler)))
        return self

    async def stop(self) -> None:
        for handle in list(self._handles):
            try:
                await handle.unsubscribe()
            except Exception as exc:
                log.error("failed to stop control endpoint service_id=%s", self._bus.service_id, exc_info=exc)
        self._handles.clear()

    @staticmethod
    def _empty_internal_error(req_id: str) -> bytes:
        return encode_obj(
            F8CommandInvokeReply(
                reqId=str(req_id or ""),
                ok=False,
                result=None,
                error=F8CommandError(code=Code.INTERNAL, message="control endpoint failed", details={}),
            )
        )

    def _wrap_handler(
        self,
        key: str,
        handler: Callable[[Any], Awaitable[None]],
    ) -> Callable[[bytes], Awaitable[bytes | None]]:
        async def _handle(payload: bytes) -> bytes | None:
            req = _TransportEndpointRequest(data=bytes(payload))
            try:
                await handler(req)
            except Exception as exc:
                log.error("control endpoint handler failed key=%s", key, exc_info=exc)
                return self._empty_internal_error("")
            return req.response or b""

        return _handle


def create_service_control_endpoint_server(bus: Any) -> ServiceControlEndpointServer:
    return RuntimeTransportServiceControlEndpointServer(bus)


ZenohServiceControlEndpointServer = RuntimeTransportServiceControlEndpointServer


__all__ = [
    "RuntimeTransportServiceControlEndpointServer",
    "ServiceControlEndpointServer",
    "ZenohServiceControlEndpointServer",
    "create_service_control_endpoint_server",
]
