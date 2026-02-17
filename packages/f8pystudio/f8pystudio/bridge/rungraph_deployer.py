from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from f8pysdk import F8RuntimeGraph
from f8pysdk.nats_naming import kv_bucket_for_service, new_id, svc_endpoint_subject
from f8pysdk.nats_transport import NatsTransport, NatsTransportConfig
from f8pysdk.service_ready import wait_service_ready

from .nats_request import parse_ok_envelope


class RungraphGateway(Protocol):
    async def deploy_runtime_graph(self, req: "RungraphDeployRequest") -> "RungraphDeployResult": ...


@dataclass(frozen=True)
class RungraphDeployConfig:
    nats_url: str
    ready_timeout_s: float = 6.0
    request_timeout_s: float = 2.0


@dataclass(frozen=True)
class RungraphDeployRequest:
    service_id: str
    graph: F8RuntimeGraph
    source: str = "studio"


@dataclass(frozen=True)
class RungraphDeployResult:
    service_id: str
    success: bool
    error_message: str = ""


@dataclass(frozen=True)
class NatsRungraphGateway:
    config: RungraphDeployConfig

    async def deploy_runtime_graph(self, req: RungraphDeployRequest) -> RungraphDeployResult:
        service_id = str(req.service_id)
        bucket = kv_bucket_for_service(service_id)
        transport = NatsTransport(NatsTransportConfig(url=str(self.config.nats_url), kv_bucket=bucket))
        await transport.connect()
        try:
            await wait_service_ready(transport, timeout_s=float(self.config.ready_timeout_s))
            payload = req.graph.model_dump(mode="json", by_alias=True)
            meta = dict(payload.get("meta") or {})
            if not str(meta.get("source") or "").strip():
                meta["source"] = str(req.source or "studio")
            payload["meta"] = meta

            request_payload = {
                "reqId": new_id(),
                "args": {"graph": payload},
                "meta": {"source": str(req.source or "studio")},
            }
            request_bytes = json.dumps(request_payload, ensure_ascii=False, default=str).encode("utf-8")
            response_bytes = await transport.request(
                svc_endpoint_subject(service_id, "set_rungraph"),
                request_bytes,
                timeout=float(self.config.request_timeout_s),
                raise_on_error=True,
            )
            if not response_bytes:
                return RungraphDeployResult(service_id=service_id, success=False, error_message="empty response")
            response_payload = json.loads(response_bytes.decode("utf-8"))
            if not isinstance(response_payload, dict):
                return RungraphDeployResult(service_id=service_id, success=False, error_message="invalid response")
            parsed = parse_ok_envelope(response_payload)
            return RungraphDeployResult(
                service_id=service_id,
                success=bool(parsed.ok),
                error_message=str(parsed.error_message),
            )
        finally:
            await transport.close()
