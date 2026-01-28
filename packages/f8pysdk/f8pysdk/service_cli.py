from __future__ import annotations

import argparse
import asyncio
import json
import os
from abc import ABC, abstractmethod
from typing import Any

from .runtime_node_registry import RuntimeNodeRegistry
from .service_runtime import ServiceRuntime, ServiceRuntimeConfig


def _env_or(default: str, key: str) -> str:
    v = os.environ.get(key)
    return v.strip() if v and v.strip() else default


class ServiceCliTemplate(ABC):
    """
    Opinionated CLI entrypoint template for a service process.

    Goal: make each service entrypoint a small "fill the blanks" class:
    - register runtime node specs
    - attach service-specific runtime wiring (executors, binders, listeners)
    - run forever with a consistent CLI (`--describe`, `--service-id`, `--nats-url`)
    """

    @property
    @abstractmethod
    def service_class(self) -> str:
        raise NotImplementedError

    # ---- registry/app construction ------------------------------------
    def build_registry(self) -> RuntimeNodeRegistry:
        return RuntimeNodeRegistry.instance()

    @abstractmethod
    def register_specs(self, registry: RuntimeNodeRegistry) -> None:
        raise NotImplementedError

    def build_runtime_config(self, *, service_id: str, nats_url: str) -> ServiceRuntimeConfig:
        return ServiceRuntimeConfig.from_values(service_id=service_id, service_class=self.service_class, nats_url=nats_url)

    def build_runtime(self, *, service_id: str, nats_url: str, registry: RuntimeNodeRegistry) -> ServiceRuntime:
        return ServiceRuntime(self.build_runtime_config(service_id=service_id, nats_url=nats_url), registry=registry)

    # ---- lifecycle hooks ----------------------------------------------
    async def setup(self, runtime: ServiceRuntime) -> None:
        """
        Hook point for wiring service-specific runtime pieces.
        Called before `runtime.start()`.
        """

    async def teardown(self, runtime: ServiceRuntime) -> None:
        """
        Hook point for stopping service-specific runtime pieces.
        Called before `runtime.stop()` (best-effort).
        """

    # ---- running -------------------------------------------------------
    async def run_forever(self, *, service_id: str, nats_url: str) -> None:
        registry = self.build_registry()
        self.register_specs(registry)

        runtime = self.build_runtime(service_id=service_id, nats_url=nats_url, registry=registry)
        await self.setup(runtime)
        await runtime.start()

        try:
            await runtime.bus.wait_terminate()
        finally:
            try:
                await self.teardown(runtime)
            except Exception:
                pass
            await runtime.stop()

    def describe_json(self) -> dict[str, Any]:
        registry = self.build_registry()
        self.register_specs(registry)
        return registry.describe(self.service_class).model_dump(mode="json")

    # ---- CLI -----------------------------------------------------------
    def cli(self, argv: list[str] | None = None, *, program_name: str | None = None) -> int:
        parser = argparse.ArgumentParser(description=program_name or self.service_class)
        parser.add_argument("--describe", action="store_true", help="Output the service description in JSON format")
        parser.add_argument("--service-id", default=_env_or("", "F8_SERVICE_ID"), help="Service instance id (required)")
        parser.add_argument("--nats-url", default=_env_or("nats://127.0.0.1:4222", "F8_NATS_URL"), help="NATS server URL")
        args = parser.parse_args(argv)

        if args.describe:
            print(json.dumps(self.describe_json(), ensure_ascii=False, indent=1))
            return 0

        service_id = str(args.service_id or "").strip()
        if not service_id:
            raise SystemExit("Missing --service-id (or env F8_SERVICE_ID)")

        asyncio.run(self.run_forever(service_id=service_id, nats_url=str(args.nats_url).strip()))
        return 0
