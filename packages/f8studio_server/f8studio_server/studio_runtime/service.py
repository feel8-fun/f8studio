from __future__ import annotations

from dataclasses import dataclass

from f8pysdk.bus import BusBackend, ServiceBusConfig
from f8pysdk.registry import RuntimeNodeRegistry
from f8pysdk.runtime import ServiceRuntime, ServiceRuntimeConfig
from f8pysdk.specs import F8ServiceDescribe

from .identifiers import SERVICE_CLASS, STUDIO_SERVICE_ID
from .presentation import PresentationOutlet
from .registry import create_studio_registry, describe_studio_registry


@dataclass(frozen=True)
class StudioRuntimeConfig:
    bus_backend: BusBackend = "zenoh"
    service_id: str = STUDIO_SERVICE_ID
    zenoh_config_path: str | None = None
    zenoh_connect: tuple[str, ...] = ()
    zenoh_listen: tuple[str, ...] = ()
    zenoh_shm_pool_bytes: int = 256 * 1024 * 1024


class StudioRuntimeService:
    def __init__(
        self,
        config: StudioRuntimeConfig,
        *,
        presentation: PresentationOutlet,
        registry: RuntimeNodeRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry or create_studio_registry(presentation=presentation)
        self._runtime: ServiceRuntime | None = None

    @property
    def service_id(self) -> str:
        return self._config.service_id

    @property
    def describe(self) -> F8ServiceDescribe:
        return describe_studio_registry(self._registry)

    async def start(self) -> None:
        if self._runtime is not None:
            return
        runtime = ServiceRuntime(
            ServiceRuntimeConfig(
                bus=ServiceBusConfig(
                    service_id=self._config.service_id,
                    service_class=SERVICE_CLASS,
                    bus_backend=self._config.bus_backend,
                    zenoh_config_path=self._config.zenoh_config_path,
                    zenoh_connect=self._config.zenoh_connect,
                    zenoh_listen=self._config.zenoh_listen,
                    zenoh_shm_pool_bytes=self._config.zenoh_shm_pool_bytes,
                    cross_publish_policy="routed",
                    data_delivery="callback",
                )
            ),
            registry=self._registry,
        )
        await runtime.start()
        self._runtime = runtime

    async def stop(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            await runtime.stop()


__all__ = ["StudioRuntimeConfig", "StudioRuntimeService"]
