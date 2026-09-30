from __future__ import annotations

from dataclasses import dataclass

from .bus import ServiceBus, ServiceBusConfig
from .data import CrossPublishPolicy
from .host import ServiceHost
from .registry import RuntimeNodeRegistry, create_runtime_node_registry


@dataclass(frozen=True)
class ServiceRuntimeConfig:
    """
    Runtime facade for a service process.

    This bundles:
    - `ServiceBus`: runtime transport, routing, state cache
    - `ServiceHost`: rungraph-driven node creation and registration
    - `RuntimeNodeRegistry`: node factory registry (explicitly registered factories)
    """

    bus: ServiceBusConfig

    def __post_init__(self) -> None:
        object.__setattr__(self, "bus", self.bus.normalized())

    @property
    def service_id(self) -> str:
        return str(self.bus.service_id)

    @property
    def service_class(self) -> str:
        return str(self.bus.service_class or "")

    @property
    def bus_backend(self) -> str:
        return str(self.bus.bus_backend)

    @property
    def cross_publish_policy(self) -> CrossPublishPolicy:
        return self.bus.cross_publish_policy


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
        self._registry = registry if registry is not None else create_runtime_node_registry()
        self._closed = False

        self.bus = ServiceBus(config.bus)
        self.host = ServiceHost(self.bus, registry=self._registry)

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("ServiceRuntime is not restartable after stop(); create a new instance")
        await self.host.start()
        await self.bus.start()

    async def stop(self) -> None:
        try:
            await self.host.stop()
        finally:
            try:
                await self.bus.stop()
            finally:
                self._closed = True

__all__ = ["ServiceRuntime", "ServiceRuntimeConfig"]
