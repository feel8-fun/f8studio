from __future__ import annotations

from dataclasses import dataclass, field

from nats.js.api import StorageType  # type: ignore[import-not-found]

from .service_host import ServiceHost, ServiceHostConfig
from .service_operator_runtime_registry import ServiceOperatorRuntimeRegistry
from .service_bus import ServiceBus, ServiceBusConfig


@dataclass(frozen=True)
class ServiceAppConfig:
    """
    Facade for a service process.

    This bundles:
    - `ServiceBus`: NATS+KV transport, routing, state cache
    - `ServiceHost`: rungraph-driven node creation and registration
    - `ServiceOperatorRuntimeRegistry`: node factory registry (optionally loaded from modules)
    """

    service_id: str
    service_class: str
    nats_url: str = "nats://127.0.0.1:4222"

    publish_all_data: bool = True
    kv_storage: StorageType = StorageType.MEMORY
    delete_bucket_on_start: bool = False
    delete_bucket_on_stop: bool = False

    registry_modules: list[str] = field(default_factory=list)


class ServiceApp:
    """
    Process-level facade that wires together `ServiceBus` and `ServiceHost`.
    """

    def __init__(
        self,
        config: ServiceAppConfig,
        *,
        registry: ServiceOperatorRuntimeRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry or ServiceOperatorRuntimeRegistry.instance()

        for module in list(getattr(config, "registry_modules", []) or []):
            try:
                self._registry.load_modules([str(module)])
            except Exception:
                continue

        self.bus = ServiceBus(
            ServiceBusConfig(
                service_id=str(config.service_id),
                nats_url=str(config.nats_url),
                publish_all_data=bool(config.publish_all_data),
                kv_storage=config.kv_storage,
                delete_bucket_on_start=bool(config.delete_bucket_on_start),
                delete_bucket_on_stop=bool(config.delete_bucket_on_stop),
            )
        )
        self.host = ServiceHost(
            self.bus,
            config=ServiceHostConfig(service_class=str(config.service_class)),
            registry=self._registry,
        )

    async def start(self) -> None:
        await self.bus.start()

    async def stop(self) -> None:
        await self.bus.stop()
