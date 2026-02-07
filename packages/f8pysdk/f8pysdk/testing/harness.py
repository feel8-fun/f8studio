from __future__ import annotations

from dataclasses import dataclass, field

from ..nats_naming import kv_bucket_for_service
from ..service_bus.bus import ServiceBus, ServiceBusConfig
from .in_memory_transport import InMemoryCluster, InMemoryTransport


@dataclass
class ServiceBusHarness:
    """
    In-process harness for spinning up multiple ServiceBus instances.
    """

    cluster: InMemoryCluster = field(default_factory=InMemoryCluster)

    def create_bus(self, service_id: str) -> ServiceBus:
        cfg = ServiceBusConfig(service_id=str(service_id), nats_url="mem://")
        transport = InMemoryTransport(cluster=self.cluster, kv_bucket=kv_bucket_for_service(cfg.service_id))
        return ServiceBus(cfg, transport=transport)
