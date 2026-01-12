from __future__ import annotations

"""
Runtime subpackage public surface.

Keep this file light: importing `f8pysdk.runtime.*` must not eagerly import
modules that depend on other heavy subsystems to avoid circular imports.
"""

from typing import TYPE_CHECKING

from .nats_naming import (
    cmd_subject,
    data_subject,
    edge_subject,
    ensure_token,
    kv_bucket_for_service,
    kv_key_node_state,
    kv_key_topology,
    new_id,
)

if TYPE_CHECKING:
    from .nats_transport import NatsTransport, NatsTransportConfig
    from .service_host import ServiceHost, ServiceHostConfig
    from .service_operator_runtime_registry import ServiceOperatorRuntimeRegistry
    from .service_runtime import ServiceRuntime, ServiceRuntimeConfig
    from .service_runtime_node import ServiceRuntimeNode

__all__ = [
    "NatsTransport",
    "NatsTransportConfig",
    "ServiceRuntime",
    "ServiceRuntimeConfig",
    "ServiceRuntimeNode",
    "ServiceHost",
    "ServiceHostConfig",
    "ServiceOperatorRuntimeRegistry",
    "cmd_subject",
    "data_subject",
    "edge_subject",
    "ensure_token",
    "kv_bucket_for_service",
    "kv_key_node_state",
    "kv_key_topology",
    "new_id",
]


def __getattr__(name: str):
    if name in ("NatsTransport", "NatsTransportConfig"):
        from .nats_transport import NatsTransport, NatsTransportConfig

        return {"NatsTransport": NatsTransport, "NatsTransportConfig": NatsTransportConfig}[name]

    if name in ("ServiceRuntime", "ServiceRuntimeConfig"):
        from .service_runtime import ServiceRuntime, ServiceRuntimeConfig

        return {"ServiceRuntime": ServiceRuntime, "ServiceRuntimeConfig": ServiceRuntimeConfig}[name]

    if name == "ServiceRuntimeNode":
        from .service_runtime_node import ServiceRuntimeNode

        return ServiceRuntimeNode

    if name in ("ServiceHost", "ServiceHostConfig"):
        from .service_host import ServiceHost, ServiceHostConfig

        return {"ServiceHost": ServiceHost, "ServiceHostConfig": ServiceHostConfig}[name]

    if name == "ServiceOperatorRuntimeRegistry":
        from .service_operator_runtime_registry import ServiceOperatorRuntimeRegistry

        return ServiceOperatorRuntimeRegistry

    raise AttributeError(name)
