from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from nats.micro import ServiceConfig, add_service  # type: ignore[import-not-found]
from nats.micro.service import EndpointConfig  # type: ignore[import-not-found]

from ..capabilities import CommandableNode
from ..generated import F8RuntimeGraph
from ..nats_naming import cmd_channel_subject, ensure_token, new_id, svc_endpoint_subject, svc_micro_name
from .state_write import StateWriteError, StateWriteOrigin, StateWriteSource

if TYPE_CHECKING:
    from .bus import ServiceBus


log = logging.getLogger(__name__)


class _ServiceBusMicroEndpoints:
    def __init__(self, bus: "ServiceBus") -> None:
        self._bus = bus
        self._micro: Any | None = None

    async def start(self) -> Any:
        nc = await self._bus._transport.require_client()
        service_name = str(self._bus._service_name or "") or self._bus.service_id
        service_class = str(self._bus._service_class or "")
        description_parts = [f"serviceName={service_name}", f"serviceId={self._bus.service_id}"]
        if service_class:
            description_parts.insert(1, f"serviceClass={service_class}")
        description = "F8 service runtime control plane (%s)." % ", ".join(description_parts)
        metadata: dict[str, Any] = {"serviceId": self._bus.service_id, "serviceName": service_name}
        if service_class:
            metadata["serviceClass"] = service_class
        self._micro = await add_service(
            nc,
            ServiceConfig(
                name=svc_micro_name(self._bus.service_id),
                version="0.0.1",
                description=description,
                metadata=metadata,
            ),
        )
        await self._register_endpoints()
        return self._micro

    async def stop(self) -> None:
        if self._micro is None:
            return
        try:
            await self._micro.stop()
        except Exception:
            pass
        self._micro = None

    def _parse_envelope(self, data: bytes) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
        req: dict[str, Any] = {}
        if data:
            try:
                req = json.loads(data.decode("utf-8"))
            except Exception:
                req = {}
        if not isinstance(req, dict):
            req = {}
        req_id = str(req.get("reqId") or "") or new_id()
        args = req.get("args") if isinstance(req.get("args"), dict) else {}
        meta = req.get("meta") if isinstance(req.get("meta"), dict) else {}
        return req_id, req, dict(args), dict(meta)

    async def _respond(
        self, req: Any, *, req_id: str, ok: bool, result: Any = None, error: dict[str, Any] | None = None
    ) -> None:
        payload = {
            "reqId": req_id,
            "ok": bool(ok),
            "result": result if ok else None,
            "error": error if not ok else None,
        }
        await req.respond(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"))

    async def _set_active_req(self, req: Any, active: bool, *, cmd: str) -> None:
        req_id, _raw, _args, meta = self._parse_envelope(req.data)
        want_active = bool(active)
        await self._bus.set_active(want_active, source=StateWriteSource.cmd, meta={"cmd": cmd, **meta})
        await self._respond(req, req_id=req_id, ok=True, result={"active": self._bus.active})

    async def _activate(self, req: Any) -> None:
        await self._set_active_req(req, True, cmd="activate")

    async def _deactivate(self, req: Any) -> None:
        await self._set_active_req(req, False, cmd="deactivate")

    async def _set_active(self, req: Any) -> None:
        req_id, raw, args, _meta = self._parse_envelope(req.data)
        want_active = args.get("active")
        if want_active is None:
            want_active = raw.get("active")
        if want_active is None:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "missing active"})
            return
        await self._set_active_req(req, bool(want_active), cmd="set_active")

    async def _status(self, req: Any) -> None:
        req_id, _raw, _args, _meta = self._parse_envelope(req.data)
        await self._respond(req, req_id=req_id, ok=True, result={"serviceId": self._bus.service_id, "active": self._bus.active})

    async def _terminate(self, req: Any) -> None:
        req_id, _raw, _args, meta = self._parse_envelope(req.data)
        try:
            log.info("terminate requested serviceId=%s meta=%s", self._bus.service_id, dict(meta or {}))
        except Exception:
            pass
        try:
            self._bus._terminate_event.set()
        except Exception:
            pass
        await self._respond(req, req_id=req_id, ok=True, result={"terminating": True})

    async def _cmd(self, req: Any) -> None:
        req_id, raw, args, meta = self._parse_envelope(req.data)
        call = str(raw.get("call") or "").strip()
        if not call:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "missing call"})
            return
        service_node = self._bus.get_node(self._bus.service_id)
        if service_node is None or not isinstance(service_node, CommandableNode):
            await self._respond(req, req_id=req_id, ok=False, error={"code": "UNKNOWN_CALL", "message": f"unknown call: {call}"})
            return
        try:
            out = await service_node.on_command(call, args, meta=meta)  # type: ignore[misc]
        except Exception as exc:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INTERNAL", "message": str(exc)})
            return
        await self._respond(req, req_id=req_id, ok=True, result=out)

    async def _set_state(self, req: Any) -> None:
        req_id, raw, args, meta = self._parse_envelope(req.data)
        node_id = args.get("nodeId") or raw.get("nodeId")
        field = args.get("field") or raw.get("field")
        value = args.get("value") if "value" in args else raw.get("value")
        if node_id is None or field is None:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "missing nodeId/field"})
            return
        node_id_s = str(node_id).strip()
        field_s = str(field).strip()
        if not node_id_s or not field_s:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "empty nodeId/field"})
            return

        try:
            node_id_s = ensure_token(node_id_s, label="node_id")
        except Exception:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "invalid nodeId"})
            return

        user_meta = dict(meta)
        user_meta.pop("source", None)
        user_meta.pop("origin", None)
        try:
            await self._bus.publish_state_external(
                node_id_s,
                field_s,
                value,
                source=StateWriteSource.endpoint,
                meta=user_meta,
            )
        except StateWriteError as exc:
            await self._respond(
                req,
                req_id=req_id,
                ok=False,
                error={"code": exc.code, "message": exc.message, "details": exc.details},
            )
            return
        except Exception as exc:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INTERNAL", "message": str(exc)})
            return
        await self._respond(req, req_id=req_id, ok=True, result={"nodeId": node_id_s, "field": field_s})

    async def _set_rungraph(self, req: Any) -> None:
        req_id, raw, args, _meta = self._parse_envelope(req.data)
        graph_obj = args.get("graph") if isinstance(args.get("graph"), dict) else raw.get("graph")
        if graph_obj is None and isinstance(raw, dict):
            graph_obj = raw if "nodes" in raw and "edges" in raw else None
        if not isinstance(graph_obj, dict):
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INVALID_ARGS", "message": "missing graph"})
            return

        try:
            graph = F8RuntimeGraph.model_validate(graph_obj)
        except Exception as exc:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INVALID_RUNGRAPH", "message": str(exc)})
            return

        try:
            await self._bus.set_rungraph(graph)
        except Exception as exc:
            await self._respond(req, req_id=req_id, ok=False, error={"code": "INTERNAL", "message": str(exc)})
            return
        await self._respond(req, req_id=req_id, ok=True, result={"graphId": str(graph.graphId)})

    async def _register_endpoints(self) -> None:
        micro = self._micro
        if micro is None:
            return
        sid = self._bus.service_id
        await micro.add_endpoint(
            EndpointConfig(
                name="activate",
                subject=svc_endpoint_subject(sid, "activate"),
                handler=self._activate,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="deactivate",
                subject=svc_endpoint_subject(sid, "deactivate"),
                handler=self._deactivate,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="set_active",
                subject=svc_endpoint_subject(sid, "set_active"),
                handler=self._set_active,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="status",
                subject=svc_endpoint_subject(sid, "status"),
                handler=self._status,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="terminate",
                subject=svc_endpoint_subject(sid, "terminate"),
                handler=self._terminate,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="quit",
                subject=svc_endpoint_subject(sid, "quit"),
                handler=self._terminate,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="cmd",
                subject=cmd_channel_subject(sid),
                handler=self._cmd,
                metadata={"builtin": "false"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="set_state",
                subject=svc_endpoint_subject(sid, "set_state"),
                handler=self._set_state,
                metadata={"builtin": "true"},
            )
        )
        await micro.add_endpoint(
            EndpointConfig(
                name="set_rungraph",
                subject=svc_endpoint_subject(sid, "set_rungraph"),
                handler=self._set_rungraph,
                metadata={"builtin": "true"},
            )
        )
