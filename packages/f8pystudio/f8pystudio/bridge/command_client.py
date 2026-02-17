from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from f8pysdk.nats_naming import cmd_channel_subject, ensure_token, new_id

from .json_codec import coerce_json_dict
from .nats_request import OkEnvelope, RequestJsonInput, request_json


class CommandGateway(Protocol):
    async def request_command(self, req: "CommandRequest") -> "CommandResponse": ...


@dataclass(frozen=True)
class CommandRequest:
    service_id: str
    call: str
    args: dict[str, Any] | None = None
    timeout_s: float = 2.0
    source: str = "ui"
    actor: str = "studio"


@dataclass(frozen=True)
class CommandResponse:
    ok: bool
    result: dict[str, Any]
    error_message: str
    payload: dict[str, Any]

    @staticmethod
    def from_envelope(envelope: OkEnvelope) -> "CommandResponse":
        return CommandResponse(
            ok=bool(envelope.ok),
            result=dict(envelope.result),
            error_message=str(envelope.error_message),
            payload=dict(envelope.payload),
        )


@dataclass
class NatsCommandGateway:
    nats_url: str
    _nc: Any | None = None

    async def ensure_connected(self) -> Any:
        if self._nc is not None:
            return self._nc
        import nats

        self._nc = await nats.connect(servers=[str(self.nats_url)], connect_timeout=2)
        return self._nc

    async def close(self) -> None:
        if self._nc is None:
            return
        await self._nc.close()
        self._nc = None

    async def request_command(self, req: CommandRequest) -> CommandResponse:
        sid = ensure_token(str(req.service_id), label="service_id")
        call_name = str(req.call or "").strip()
        if not call_name:
            raise ValueError("call is empty")

        nc = await self.ensure_connected()
        payload: dict[str, Any] = {
            "reqId": new_id(),
            "call": call_name,
            "args": coerce_json_dict(req.args or {}),
            "meta": {"actor": str(req.actor), "source": str(req.source)},
        }
        raw_payload = await request_json(
            nc,
            RequestJsonInput(
                subject=cmd_channel_subject(sid),
                payload=payload,
                timeout_s=float(req.timeout_s),
            ),
        )
        result = raw_payload.get("result")
        result_obj = result if isinstance(result, dict) else {"value": result}
        err_obj = raw_payload.get("error")
        err_message = str(err_obj.get("message") or "").strip() if isinstance(err_obj, dict) else ""
        ok = bool(raw_payload.get("ok") is True)
        if ok:
            err_message = ""
        return CommandResponse(ok=ok, result=result_obj, error_message=err_message, payload=raw_payload)
