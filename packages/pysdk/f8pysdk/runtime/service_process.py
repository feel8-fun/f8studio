from __future__ import annotations

import asyncio
import os
import uuid

from .service_host import ServiceHost, ServiceHostConfig
from .service_operator_runtime_registry import ServiceOperatorRuntimeRegistry
from .service_runtime import ServiceRuntime, ServiceRuntimeConfig


def _env_or(default: str, key: str) -> str:
    v = os.environ.get(key)
    return v.strip() if v and v.strip() else default


async def main() -> None:
    """
    Generic service process entrypoint.

    Environment:
    - `F8_SERVICE_ID` (required): service instance id (token-safe)
    - `F8_SERVICE_CLASS` (required): service class key for runtime registry lookup
    - `F8_NATS_URL` (optional): defaults to `nats://127.0.0.1:4222`
    - `F8_NATS_BUCKET` (optional): overrides per-service default bucket
    - `F8_ACTOR_ID` (optional): defaults to random uuid4 hex
    - `F8_SERVICE_MODULES` (optional): comma-separated python modules to import and register runtimes
    """
    service_id = _env_or("", "F8_SERVICE_ID")
    service_class = _env_or("", "F8_SERVICE_CLASS")
    if not service_id:
        raise RuntimeError("F8_SERVICE_ID is required")
    if not service_class:
        raise RuntimeError("F8_SERVICE_CLASS is required")

    nats_url = _env_or("nats://127.0.0.1:4222", "F8_NATS_URL")
    bucket = os.environ.get("F8_NATS_BUCKET")
    actor_id = _env_or(uuid.uuid4().hex, "F8_ACTOR_ID")

    reg = ServiceOperatorRuntimeRegistry.instance()
    modules = _env_or("", "F8_SERVICE_MODULES")
    if modules:
        reg.load_modules([m.strip() for m in modules.split(",") if m.strip()])

    runtime = ServiceRuntime(ServiceRuntimeConfig(service_id=service_id, nats_url=nats_url, kv_bucket=bucket, actor_id=actor_id))
    ServiceHost(runtime, config=ServiceHostConfig(service_class=service_class), registry=reg)

    await runtime.start()
    try:
        await asyncio.Event().wait()
    finally:
        await runtime.stop()


if __name__ == "__main__":
    asyncio.run(main())

