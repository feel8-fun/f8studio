from __future__ import annotations

"""Internal-only service control endpoint handlers for `service_bus`."""

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import msgspec

from ...codec import decode_as, decode_obj, dump_json, encode_obj, validate_as
from ...generated import (
    Code,
    F8ActivateRequest,
    F8ActiveReply,
    F8ActiveReplyResult,
    F8CommandError,
    F8CommandInvokeReply,
    F8DeactivateRequest,
    F8SetActiveRequest,
    F8SetRungraphReply,
    F8SetRungraphReplyResult,
    F8SetRungraphRequest,
    F8SetStateReply,
    F8SetStateReplyResult,
    F8SetStateRequest,
    F8StatusRequest,
    F8TerminateReply,
    F8TerminateReplyResult,
    F8TerminateRequest,
)
from ...f8_naming import ensure_token, new_id
from ...state import StateWriteError, StateWriteSource
from .command import (
    CommandExecutionErrorKind,
    CommandInvocation,
    CommandInvokeOptions,
    CommandOutputPolicy,
    execute_command,
)

if TYPE_CHECKING:
    from ..runtime import ServiceBus


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DecodedCommandRequest:
    req_id: str
    call: str
    args: Any
    meta: dict[str, Any]


class ServiceBusControlHandlers:
    def __init__(self, bus: "ServiceBus") -> None:
        self._bus = bus

    async def start(self) -> Any:
        return self

    async def stop(self) -> None:
        return None

    @staticmethod
    def _req_id(req_id: str) -> str:
        out = str(req_id or "").strip()
        return out or new_id()

    @staticmethod
    def _error(*, code: str, message: str, details: Any = None) -> F8CommandError:
        code_text = str(code or "").strip()
        try:
            enum_code = Code(code_text)
        except ValueError:
            enum_code = Code.INTERNAL
        return F8CommandError(code=enum_code, message=str(message), details=details or {})

    @staticmethod
    def _meta_dict(meta: dict[str, Any] | msgspec.UnsetType | None, *, cmd: str | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if cmd:
            out["cmd"] = str(cmd)
        if meta is None or isinstance(meta, msgspec.UnsetType):
            return out
        for key, value in dict(meta).items():
            out[str(key)] = value
        return out

    def _decode_command_request(self, raw: bytes) -> _DecodedCommandRequest:
        payload = decode_obj(raw)
        req_id = self._req_id(str(payload.get("reqId") or ""))
        call = str(payload.get("call") or "").strip()
        raw_meta = payload.get("meta")
        if raw_meta is None:
            meta = {}
        elif isinstance(raw_meta, dict):
            meta = self._meta_dict(raw_meta)
        else:
            raise ValueError("msgpack decode failed: field meta must be an object")
        return _DecodedCommandRequest(
            req_id=req_id,
            call=call,
            args=payload.get("args"),
            meta=meta,
        )

    async def _set_active_req(
        self,
        req: Any,
        *,
        req_id: str,
        active: bool,
        cmd: str,
        meta: dict[str, Any] | msgspec.UnsetType | None,
    ) -> None:
        want_active = bool(active)
        await self._bus.set_active(want_active, source=StateWriteSource.cmd, meta=self._meta_dict(meta, cmd=cmd))
        await req.respond(
            encode_obj(
                F8ActiveReply(
                    reqId=req_id,
                    ok=True,
                    result=F8ActiveReplyResult(active=self._bus.active),
                    error=None,
                )
            )
        )

    async def _activate(self, req: Any) -> None:
        try:
            payload = decode_as(req.data, F8ActivateRequest)
        except ValueError as exc:
            await req.respond(
                encode_obj(
                    F8ActiveReply(
                        reqId=new_id(),
                        ok=False,
                        result=None,
                        error=self._error(code="INVALID_ARGS", message=str(exc)),
                    )
                )
            )
            return
        await self._set_active_req(
            req,
            req_id=self._req_id(payload.reqId),
            active=True,
            cmd="activate",
            meta=payload.meta,
        )

    async def _deactivate(self, req: Any) -> None:
        try:
            payload = decode_as(req.data, F8DeactivateRequest)
        except ValueError as exc:
            await req.respond(
                encode_obj(
                    F8ActiveReply(
                        reqId=new_id(),
                        ok=False,
                        result=None,
                        error=self._error(code="INVALID_ARGS", message=str(exc)),
                    )
                )
            )
            return
        await self._set_active_req(
            req,
            req_id=self._req_id(payload.reqId),
            active=False,
            cmd="deactivate",
            meta=payload.meta,
        )

    async def _set_active(self, req: Any) -> None:
        try:
            payload = decode_as(req.data, F8SetActiveRequest)
        except ValueError as exc:
            await req.respond(
                encode_obj(
                    F8ActiveReply(
                        reqId=new_id(),
                        ok=False,
                        result=None,
                        error=self._error(code="INVALID_ARGS", message=str(exc)),
                    )
                )
            )
            return
        await self._set_active_req(
            req,
            req_id=self._req_id(payload.reqId),
            active=payload.args.active,
            cmd="set_active",
            meta=payload.meta,
        )

    async def _status(self, req: Any) -> None:
        try:
            payload = decode_as(req.data, F8StatusRequest)
            req_id = self._req_id(payload.reqId)
        except ValueError:
            req_id = new_id()
        await req.respond(
            encode_obj(
                {
                    "reqId": req_id,
                    "ok": True,
                    "result": {
                        "serviceId": self._bus.service_id,
                        "serviceClass": self._bus.service_class,
                        "runtimeInstanceId": self._bus.runtime_instance_id,
                        "active": self._bus.active,
                        "rungraphGraphId": str(self._bus._graph.graphId or "") if self._bus._graph is not None else "",
                        "rungraphRevision": str(self._bus._graph.revision or "") if self._bus._graph is not None else "",
                        "rungraphFingerprint": self._bus._rungraph_fingerprint,
                    },
                    "error": None,
                }
            )
        )

    async def _debug_data(self, req: Any) -> None:
        req_id = new_id()
        try:
            payload = decode_obj(req.data)
            req_id = self._req_id(str(payload.get("reqId") or ""))
            args = payload.get("args")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise ValueError("msgpack decode failed: field args must be an object")
            node_id = str(args.get("nodeId") or "").strip()
            port = str(args.get("port") or "").strip()
            if node_id:
                node_id = ensure_token(node_id, label="node_id")
            if port:
                port = ensure_token(port, label="port")
            result = self._bus.data_router.debug_input_buffers(
                node_id=node_id,
                port=port,
                limit=int(args.get("limit") or 100),
                include_value=bool(args.get("includeValue", True)),
                max_value_bytes=int(args.get("maxValueBytes") or 65536),
            )
        except (TypeError, ValueError) as exc:
            await req.respond(
                encode_obj(
                    {
                        "reqId": req_id,
                        "ok": False,
                        "result": None,
                        "error": self._error(code="INVALID_ARGS", message=str(exc)),
                    }
                )
            )
            return
        await req.respond(
            encode_obj(
                {
                    "reqId": req_id,
                    "ok": True,
                    "result": result,
                    "error": None,
                }
            )
        )

    async def _terminate(self, req: Any) -> None:
        try:
            payload = decode_as(req.data, F8TerminateRequest)
            req_id = self._req_id(payload.reqId)
            meta = payload.meta
        except ValueError:
            req_id = new_id()
            meta = None
        log.info("terminate requested serviceId=%s meta=%s", self._bus.service_id, self._meta_dict(meta))
        self._bus._terminate_event.set()
        await req.respond(
            encode_obj(
                F8TerminateReply(
                    reqId=req_id,
                    ok=True,
                    result=F8TerminateReplyResult(terminating=True),
                    error=None,
                )
            )
        )

    async def _cmd(self, req: Any) -> None:
        try:
            payload = self._decode_command_request(req.data)
        except ValueError as exc:
            await req.respond(
                encode_obj(
                    F8CommandInvokeReply(
                        reqId=new_id(),
                        ok=False,
                        result=None,
                        error=self._error(code="INVALID_ARGS", message=str(exc)),
                    )
                )
            )
            return
        req_id = payload.req_id
        call = payload.call
        if not call:
            await req.respond(
                encode_obj(
                    F8CommandInvokeReply(
                        reqId=req_id,
                        ok=False,
                        result=None,
                        error=self._error(code="INVALID_ARGS", message="missing call"),
                    )
                )
            )
            return
        result = await execute_command(
            self._bus,
            invocation=CommandInvocation(
                node_id=self._bus.service_id,
                call=call,
                args=payload.args,
            ),
            options=CommandInvokeOptions(
                call_meta=payload.meta,
                output_policy=CommandOutputPolicy.none,
                output_ts_ms=None,
                output_meta={},
            ),
        )
        if not result.ok:
            error_code = "INTERNAL"
            if result.error_kind == CommandExecutionErrorKind.missing_target:
                error_code = "UNKNOWN_CALL"
            elif result.error_kind == CommandExecutionErrorKind.invalid_args:
                error_code = "INVALID_ARGS"
            await req.respond(
                encode_obj(
                    F8CommandInvokeReply(
                        reqId=req_id,
                        ok=False,
                        result=None,
                        error=self._error(code=error_code, message=str(result.error_message or "")),
                    )
                )
            )
            return
        await req.respond(encode_obj(F8CommandInvokeReply(reqId=req_id, ok=True, result=result.value, error=None)))

    async def _set_state(self, req: Any) -> None:
        try:
            payload = decode_as(req.data, F8SetStateRequest)
        except ValueError as exc:
            await req.respond(
                encode_obj(
                    F8SetStateReply(
                        reqId=new_id(),
                        ok=False,
                        result=None,
                        error=self._error(code="INVALID_ARGS", message=str(exc)),
                    )
                )
            )
            return
        req_id = self._req_id(payload.reqId)
        node_id_s = str(payload.args.nodeId or "").strip()
        args_mapping = dump_json(payload.args)
        field_s = str(args_mapping.get("field") or "").strip() if isinstance(args_mapping, dict) else ""
        value = payload.args.value
        if not node_id_s or not field_s:
            await req.respond(
                encode_obj(
                    F8SetStateReply(
                        reqId=req_id,
                        ok=False,
                        result=None,
                        error=self._error(code="INVALID_ARGS", message="empty nodeId/field"),
                    )
                )
            )
            return

        try:
            node_id_s = ensure_token(node_id_s, label="node_id")
        except ValueError:
            await req.respond(
                encode_obj(
                    F8SetStateReply(
                        reqId=req_id,
                        ok=False,
                        result=None,
                        error=self._error(code="INVALID_ARGS", message="invalid nodeId"),
                    )
                )
            )
            return

        user_meta = self._meta_dict(payload.meta)
        user_meta.pop("source", None)
        try:
            await self._bus.publish_state_external(
                node_id_s,
                field_s,
                value,
                source=StateWriteSource.endpoint,
                meta=user_meta,
            )
        except StateWriteError as exc:
            await req.respond(
                encode_obj(
                    F8SetStateReply(
                        reqId=req_id,
                        ok=False,
                        result=None,
                        error=self._error(code=exc.code, message=exc.message, details=exc.details),
                    )
                )
            )
            return
        except Exception as exc:
            await req.respond(
                encode_obj(
                    F8SetStateReply(
                        reqId=req_id,
                        ok=False,
                        result=None,
                        error=self._error(code="INTERNAL", message=str(exc)),
                    )
                )
            )
            return
        await req.respond(
            encode_obj(
                F8SetStateReply(
                    reqId=req_id,
                    ok=True,
                    result=validate_as(
                        F8SetStateReplyResult,
                        {"nodeId": node_id_s, "field": field_s},
                    ),
                    error=None,
                )
            )
        )

    async def _set_rungraph(self, req: Any) -> None:
        try:
            decoded_req = decode_as(req.data, F8SetRungraphRequest)
            graph = decoded_req.args.graph
        except ValueError as exc:
            await req.respond(
                encode_obj(
                    F8SetRungraphReply(
                        reqId=new_id(),
                        ok=False,
                        result=None,
                        error=self._error(code="INVALID_RUNGRAPH", message=str(exc)),
                    )
                )
            )
            return
        req_id = self._req_id(decoded_req.reqId)

        try:
            source = "control"
            target_fingerprint = ""
            force_apply = False
            meta = decoded_req.meta
            if isinstance(meta, dict):
                source = str(meta.get("source") or "control")
                target_fingerprint = str(meta.get("targetFingerprint") or "")
                force_apply = bool(meta.get("forceApply"))
            await self._bus.submit_rungraph(
                graph,
                req_id=req_id,
                source=source,
                target_fingerprint=target_fingerprint,
                force_apply=force_apply,
            )
        except Exception as exc:
            await req.respond(
                encode_obj(
                    F8SetRungraphReply(
                        reqId=req_id,
                        ok=False,
                        result=None,
                        error=self._error(code="INTERNAL", message=str(exc)),
                    )
                )
            )
            return
        await req.respond(
            encode_obj(
                F8SetRungraphReply(
                    reqId=req_id,
                    ok=True,
                    result=F8SetRungraphReplyResult(graphId=str(graph.graphId)),
                    error=None,
                )
            )
        )

ServiceBusMicroEndpoints = ServiceBusControlHandlers


__all__ = ["ServiceBusControlHandlers", "ServiceBusMicroEndpoints"]
