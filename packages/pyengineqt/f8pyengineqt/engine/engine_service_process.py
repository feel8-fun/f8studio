from __future__ import annotations

import asyncio
import os
import json
import uuid
from dataclasses import dataclass
from typing import Any

from .nats_naming import ensure_token, kv_bucket_for_service
from .engine_executor import EngineExecutor
from .nats_naming import cmd_subject
from .operator_runtime_registry import OperatorRuntimeRegistry
from ..runtime import ServiceRuntime, ServiceRuntimeConfig


@dataclass(frozen=True)
class EngineServiceProcessConfig:
    """
    Minimal engine service process (v1).

    Today it only:
    - watches `svc.<serviceId>.topology` in bucket `svc_<serviceId>`
    - registers runtime nodes for topology nodes so cross-edge subscriptions work
    - keeps a shared KV/state cache (ServiceRuntime)
    """

    service_id: str
    nats_url: str
    kv_bucket: str | None = None
    actor_id: str | None = None


class EngineServiceProcess:
    def __init__(self, config: EngineServiceProcessConfig) -> None:
        self._config = config
        self._runtime = ServiceRuntime(
            ServiceRuntimeConfig(
                service_id=ensure_token(config.service_id, label="service_id"),
                nats_url=str(config.nats_url).strip(),
                kv_bucket=(config.kv_bucket or "").strip() or kv_bucket_for_service(config.service_id),
                actor_id=(config.actor_id or "").strip() or None,
            )
        )
        self._executor = EngineExecutor(self._runtime)
        self._cmd_sub: Any | None = None

        self._runtime.add_topology_listener(self._on_topology)

    async def _on_topology(self, graph: Any) -> None:
        try:
            await self._executor.apply_topology(graph)
        except Exception:
            return

    async def _on_cmd_run(self, _subject: str, payload: bytes) -> None:
        try:
            msg = json.loads(payload.decode("utf-8")) if payload else {}
        except Exception:
            msg = {}
        mode = str(msg.get("mode") or "once")
        if mode != "once":
            return
        await self._executor.run_once()

    async def run(self) -> None:
        await self._runtime.start()
        try:
            subj = cmd_subject(self._runtime.service_id, "run")
            self._cmd_sub = await self._runtime.subscribe(subj, cb=self._on_cmd_run)
        except Exception:
            self._cmd_sub = None
        try:
            await asyncio.Event().wait()
        finally:
            if self._cmd_sub is not None:
                try:
                    await self._cmd_sub.unsubscribe()
                except Exception:
                    pass
            await self._runtime.stop()


def _env_or(default: str, key: str) -> str:
    v = os.environ.get(key)
    return v.strip() if v and v.strip() else default


async def main() -> None:
    service_id = _env_or(uuid.uuid4().hex, "F8_SERVICE_ID")
    nats_url = _env_or("nats://127.0.0.1:4222", "F8_NATS_URL")
    bucket = os.environ.get("F8_NATS_BUCKET")
    actor_id = os.environ.get("F8_ACTOR_ID") or uuid.uuid4().hex

    modules = (os.environ.get("F8_OPERATOR_RUNTIME_MODULES") or "").strip()
    if modules:
        try:
            OperatorRuntimeRegistry.instance().load_modules([m.strip() for m in modules.split(",")])
        except Exception as exc:
            print(f"[engine] failed to load operator runtimes: {exc}")

    proc = EngineServiceProcess(
        EngineServiceProcessConfig(
            service_id=service_id,
            nats_url=nats_url,
            kv_bucket=bucket,
            actor_id=actor_id,
        )
    )
    await proc.run()


if __name__ == "__main__":
    asyncio.run(main())
