from __future__ import annotations

from f8pysdk import F8ServiceSchemaVersion, F8ServiceSpec
from f8pysdk import F8StateAccess, F8StateSpec, integer_schema
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from .constants import SERVICE_CLASS, STUDIO_SERVICE_ID
from .operators import register_operator


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
            tags=["__hidden__", "editor", "ui", "python", "py"],
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
    register_operator(reg)

    return reg
