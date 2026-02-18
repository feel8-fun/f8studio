from __future__ import annotations

from dataclasses import dataclass

from nats.js.api import StorageType  # type: ignore[import-not-found]

from .runtime_node_registry import RuntimeNodeRegistry
from .service_bus.bus import DataDeliveryMode, ServiceBus, ServiceBusConfig
from .service_host import ServiceHost, ServiceHostConfig


@dataclass(frozen=True)
class ServiceRuntimeConfig:
    """
    Runtime facade for a service process.

    This bundles:
    - `ServiceBus`: NATS+KV transport, routing, state cache
    - `ServiceHost`: rungraph-driven node creation and registration
    - `RuntimeNodeRegistry`: node factory registry (optionally loaded from modules)
    """

    bus: ServiceBusConfig
    host: ServiceHostConfig
    registry_modules: tuple[str, ...] = ()

    @classmethod
    def from_values(
        cls,
        *,
        service_id: str,
        service_class: str,
        service_name: str | None = None,
        nats_url: str = "nats://127.0.0.1:4222",
        publish_all_data: bool = True,
        kv_storage: StorageType = StorageType.MEMORY,
        delete_bucket_on_start: bool = False,
        delete_bucket_on_stop: bool = False,
        data_delivery: DataDeliveryMode = "pull",
        state_sync_concurrency: int = 8,
        state_cache_max_entries: int = 8192,
        data_input_max_buffers: int = 4096,
        data_input_default_queue_size: int = 256,
        registry_modules: list[str] | tuple[str, ...] | None = None,
    ) -> "ServiceRuntimeConfig":
        bus = ServiceBusConfig(
            service_id=str(service_id),
            service_name=str(service_name or "") or None,
            service_class=str(service_class or "") or None,
            nats_url=str(nats_url),
            publish_all_data=bool(publish_all_data),
            kv_storage=kv_storage,
            delete_bucket_on_start=bool(delete_bucket_on_start),
            delete_bucket_on_stop=bool(delete_bucket_on_stop),
            data_delivery=data_delivery,
            state_sync_concurrency=max(1, int(state_sync_concurrency)),
            state_cache_max_entries=max(0, int(state_cache_max_entries)),
            data_input_max_buffers=max(0, int(data_input_max_buffers)),
            data_input_default_queue_size=max(1, int(data_input_default_queue_size)),
        )
        host = ServiceHostConfig(service_class=str(service_class))
        modules = tuple(str(m).strip() for m in (registry_modules or ()) if str(m).strip())
        return cls(bus=bus, host=host, registry_modules=modules)

    @property
    def service_id(self) -> str:
        return str(self.bus.service_id)

    @property
    def service_class(self) -> str:
        return str(self.host.service_class)

    @property
    def nats_url(self) -> str:
        return str(self.bus.nats_url)

    @property
    def publish_all_data(self) -> bool:
        return bool(self.bus.publish_all_data)

    @property
    def kv_storage(self) -> StorageType:
        return self.bus.kv_storage

    @property
    def delete_bucket_on_start(self) -> bool:
        return bool(self.bus.delete_bucket_on_start)

    @property
    def delete_bucket_on_stop(self) -> bool:
        return bool(self.bus.delete_bucket_on_stop)


class ServiceRuntime:
    """
    Process-level runtime facade that wires together `ServiceBus` and `ServiceHost`.
    """

    def __init__(
        self,
        config: ServiceRuntimeConfig,
        *,
        registry: RuntimeNodeRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry or RuntimeNodeRegistry.instance()

        for module in config.registry_modules:
            self._registry.load_modules([str(module)])

        self.bus = ServiceBus(config.bus)
        self.host = ServiceHost(self.bus, config=config.host, registry=self._registry)

    async def start(self) -> None:
        await self.host.start()
        await self.bus.start()

    async def stop(self) -> None:
        await self.bus.stop()
