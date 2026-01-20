from __future__ import annotations

import argparse
import asyncio
import json
import os

from f8pysdk.runtime_node_registry import RuntimeNodeRegistry
from f8pysdk.service_app import ServiceApp, ServiceAppConfig

from f8pysdk.executors.exec_flow import ExecFlowExecutor
from f8pyengine.engine_binder import EngineBinder
from f8pyengine.pyengine_node_registry import register_pyengine_specs


def _env_or(default: str, key: str) -> str:
    v = os.environ.get(key)
    return v.strip() if v and v.strip() else default


async def _run_service(*, service_id: str, nats_url: str) -> None:
    registry = RuntimeNodeRegistry.instance()
    register_pyengine_specs(registry)

    service_class = "f8.pyengine"
    app = ServiceApp(
        ServiceAppConfig(
            service_id=service_id,
            service_class=service_class,
            nats_url=nats_url,
        ),
        registry=registry,
    )
    executor = ExecFlowExecutor(app.bus)
    _binder = EngineBinder(bus=app.bus, executor=executor, service_class=service_class)

    async def _on_lifecycle(active: bool, _meta: dict[str, object]) -> None:
        await executor.set_active(active)

    app.bus.add_lifecycle_listener(_on_lifecycle)

    await app.start()

    try:
        await asyncio.Event().wait()
    finally:
        try:
            await executor.stop_entrypoint()
        except Exception:
            pass
        await app.stop()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="F8PyEngine")
    parser.add_argument("--describe", action="store_true", help="Output the service description in JSON format")
    parser.add_argument("--service-id", default=_env_or("", "F8_SERVICE_ID"), help="Service instance id (required)")
    parser.add_argument("--nats-url", default=_env_or("nats://127.0.0.1:4222", "F8_NATS_URL"), help="NATS server URL")
    args = parser.parse_args(argv)

    if args.describe:
        registry = RuntimeNodeRegistry.instance()
        register_pyengine_specs(registry)
        describe = registry.describe("f8.pyengine").model_dump(mode="json")
        print(json.dumps(describe, ensure_ascii=False, indent=1))
        return 0

    service_id = str(args.service_id or "").strip()
    if not service_id:
        raise SystemExit("Missing --service-id (or env F8_SERVICE_ID)")

    asyncio.run(_run_service(service_id=service_id, nats_url=str(args.nats_url).strip()))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
