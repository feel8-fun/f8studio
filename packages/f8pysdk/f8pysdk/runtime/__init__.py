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
    kv_key_rungraph,
    new_id,
)

if TYPE_CHECKING:
    from .nats_transport import NatsTransport, NatsTransportConfig
    from .service_app import ServiceApp, ServiceAppConfig
    from .service_host import ServiceHost, ServiceHostConfig
    from .service_operator_runtime_registry import ServiceOperatorRuntimeRegistry
    from .service_bus import ServiceBus, ServiceBusConfig
    from .service_runtime_node import OperatorRuntimeNode, RuntimeNode, ServiceNodeRuntimeNode

__all__ = [
    "NatsTransport",
    "NatsTransportConfig",
    "ServiceApp",
    "ServiceAppConfig",
    "ServiceBus",
    "ServiceBusConfig",
    "RuntimeNode",
    "ServiceNodeRuntimeNode",
    "OperatorRuntimeNode",
    "ServiceHost",
    "ServiceHostConfig",
    "ServiceOperatorRuntimeRegistry",
    "cmd_subject",
    "data_subject",
    "edge_subject",
    "ensure_token",
    "kv_bucket_for_service",
    "kv_key_node_state",
    "kv_key_rungraph",
    "new_id",
]


def __getattr__(name: str):
    if name in ("NatsTransport", "NatsTransportConfig"):
        from .nats_transport import NatsTransport, NatsTransportConfig

        return {"NatsTransport": NatsTransport, "NatsTransportConfig": NatsTransportConfig}[name]

    if name in ("ServiceApp", "ServiceAppConfig"):
        from .service_app import ServiceApp, ServiceAppConfig

        return {"ServiceApp": ServiceApp, "ServiceAppConfig": ServiceAppConfig}[name]

    if name in ("ServiceBus", "ServiceBusConfig"):
        from .service_bus import ServiceBus, ServiceBusConfig

        return {
            "ServiceBus": ServiceBus,
            "ServiceBusConfig": ServiceBusConfig,
        }[name]

    if name in ("RuntimeNode", "ServiceNodeRuntimeNode", "OperatorRuntimeNode"):
        from .service_runtime_node import OperatorRuntimeNode, RuntimeNode, ServiceNodeRuntimeNode

        return {
            "RuntimeNode": RuntimeNode,
            "ServiceNodeRuntimeNode": ServiceNodeRuntimeNode,
            "OperatorRuntimeNode": OperatorRuntimeNode,
        }[name]

    if name in ("ServiceHost", "ServiceHostConfig"):
        from .service_host import ServiceHost, ServiceHostConfig

        return {"ServiceHost": ServiceHost, "ServiceHostConfig": ServiceHostConfig}[name]

    if name == "ServiceOperatorRuntimeRegistry":
        from .service_operator_runtime_registry import ServiceOperatorRuntimeRegistry

        return ServiceOperatorRuntimeRegistry

    raise AttributeError(name)
