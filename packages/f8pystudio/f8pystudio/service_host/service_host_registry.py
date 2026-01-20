from __future__ import annotations

from f8pysdk import F8OperatorSchemaVersion, F8OperatorSpec, F8ServiceSchemaVersion, F8ServiceSpec
from f8pysdk import F8StateAccess, F8StateSpec, integer_schema
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from .constants import SERVICE_CLASS, STUDIO_SERVICE_ID
from ..runtime_nodes.print_node import register_print_node


def register_pystudio_specs(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    """
    Register f8.pystudio service/operator specs for discovery / `--describe`.

    This uses the shared `RuntimeNodeRegistry` so other tools/services can call
    `RuntimeNodeRegistry.instance().describe("f8.pystudio")` uniformly.
    """
    reg = registry or RuntimeNodeRegistry.instance()

    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=SERVICE_CLASS,
            version="0.0.1",
            label="PyStudio",
            description="Service Graph Editor in Python and Qt.",
            tags=["editor", "ui", "python", "py"],
            rendererClass="",
            stateFields=[
                F8StateSpec(
                    name="tickMs",
                    label="Refresh Interval (ms)",
                    description="Interval in milliseconds for refreshing the UI nodes in the editor.",
                    valueSchema=integer_schema(default=100, minimum=16, maximum=5000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
            ],
            editableStateFields=False,
            editableDataInPorts=False,
            editableDataOutPorts=False,
            editableCommands=False,
        ),
        overwrite=True,
    )

    # debug
    register_print_node(reg)

    return reg


class ServiceHostRegistry:
    """Registry for service host specifications and operator specifications."""

    def __init__(self):
        self._registry = register_pystudio_specs()

    @staticmethod
    def instance() -> "ServiceHostRegistry":
        # Singleton instance accessor.
        if not hasattr(ServiceHostRegistry, "_instance"):
            ServiceHostRegistry._instance = ServiceHostRegistry()
        return ServiceHostRegistry._instance

    @property
    def serviceClass(self) -> str:
        return SERVICE_CLASS

    def service_spec(self) -> F8ServiceSpec:
        spec = self._registry.service_spec(SERVICE_CLASS)
        if spec is None:
            raise KeyError(f"service spec not registered: {SERVICE_CLASS}")
        return spec

    def operator_specs(self) -> list[F8OperatorSpec]:
        return list(self._registry.operator_specs(SERVICE_CLASS))

    def register_operator(self, spec: F8OperatorSpec) -> None:
        if self.serviceClass != spec.serviceClass:
            raise ValueError("Cannot register operator spec for different service class.")
        self._registry.register_operator_spec(spec, overwrite=True)
