from __future__ import annotations

from f8pysdk.specs import F8ServiceSchemaVersion, F8ServiceSpec, F8StateAccess, F8StateSpec, boolean_schema, string_schema
from f8pysdk.registry import Registry, RuntimeNodeRegistry, create_runtime_node_registry, shared_runtime_node_registry

from .constants import SERVICE_CLASS
from .proclauncher_service_node import ProcLauncherServiceNode


def register_proclauncher_specs(registry: Registry) -> Registry:
    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=SERVICE_CLASS,
            paletteCategory="svc",
            version="0.0.1",
            label="Proc Launcher",
            description="Launches an external OS process (optionally detached).",
            tags=["utility", "process", "launcher"],
            rendererClass="default_svc",
            stateFields=[
                F8StateSpec(
                    name="programPath",
                    label="Program Path",
                    description="Executable path or command line (quoted if it contains spaces). Cleared when exporting publish JSON.",
                    valueSchema=string_schema(default=""),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=True,
                    redactOnPublish=True,
                ),
                F8StateSpec(
                    name="singleton",
                    label="Singleton",
                    description="If true, do not start if a previous launch for the same command is still running.",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="detached",
                    label="Detached",
                    description="If true, do not stop the launched process when this service stops.",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
            ],
        ),
        ProcLauncherServiceNode,
        overwrite=True,
    )
    return registry


def create_proclauncher_registry() -> RuntimeNodeRegistry:
    runtime_registry = create_runtime_node_registry()
    register_proclauncher_specs(Registry.wrap(runtime_registry))
    return runtime_registry


def shared_proclauncher_registry() -> RuntimeNodeRegistry:
    runtime_registry = shared_runtime_node_registry()
    register_proclauncher_specs(Registry.wrap(runtime_registry))
    return runtime_registry
