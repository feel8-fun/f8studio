from __future__ import annotations

from typing import Any

from f8pysdk.nodes import ServiceNode
from f8pysdk.registry import Registry, RuntimeNodeRegistry, create_runtime_node_registry
from f8pysdk.specs import (
    F8RuntimeNode,
    F8ServiceDescribe,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    integer_schema,
)

from .identifiers import SERVICE_CLASS
from .operators import register_operator
from .presentation import PresentationOutlet


class StudioServiceNode(ServiceNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=node_id,
            data_in_ports=[port.name for port in (node.dataInPorts or [])],
            data_out_ports=[port.name for port in (node.dataOutPorts or [])],
            state_fields=[field.name for field in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})


def register_studio_runtime(registry: Registry, *, presentation: PresentationOutlet) -> Registry:
    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=SERVICE_CLASS,
            version="0.1.0",
            label="Web Studio Runtime",
            description="Backend runtime for Studio computation and presentation operators.",
            tags=["editor", "web", "runtime"],
            paletteCategory="svc",
            hiddenInPalette=True,
            rendererClass="default_svc",
            stateFields=[
                F8StateSpec(
                    name="tickMs",
                    label="Refresh Interval (ms)",
                    description="Default refresh interval for Studio presentation operators.",
                    valueSchema=integer_schema(default=100, minimum=16, maximum=5000),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=True,
                )
            ],
        ),
        StudioServiceNode,
        overwrite=True,
    )
    return register_operator(registry, presentation=presentation)


def create_studio_registry(*, presentation: PresentationOutlet) -> RuntimeNodeRegistry:
    runtime_registry = create_runtime_node_registry()
    register_studio_runtime(Registry.wrap(runtime_registry), presentation=presentation)
    return runtime_registry


def describe_studio_registry(registry: RuntimeNodeRegistry) -> F8ServiceDescribe:
    return Registry.wrap(registry).describe(SERVICE_CLASS)


__all__ = [
    "StudioServiceNode",
    "create_studio_registry",
    "describe_studio_registry",
    "register_studio_runtime",
]
