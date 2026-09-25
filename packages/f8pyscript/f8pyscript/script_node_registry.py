from __future__ import annotations

from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
    F8Command,
    F8CommandParam,
    F8DataPortSpec,
    F8SpecEditPolicy,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    editable_collection_edit_policy,
    integer_schema,
    string_schema,
)
from f8pysdk.registry import Registry, RuntimeNodeRegistry, create_runtime_node_registry, shared_runtime_node_registry

from .constants import SERVICE_CLASS
from .editor_assist_payload import pyscript_code_field_editor_assist_payload
from .script_service_node import PythonScriptServiceNode
from .script_templates import DEFAULT_CODE


def register_specs(registry: Registry) -> Registry:
    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=SERVICE_CLASS,
            paletteCategory="svc",
            version="0.0.1",
            label="Python Script Service",
            description="Standalone python script runtime service with lifecycle/tick/command hooks.",
            tags=["python", "script", "service"],
            rendererClass="default_svc",
            stateFields=[
                F8StateSpec(
                    name="code",
                    label="Code",
                    description="Python source code.",
                    valueSchema=string_schema(default=DEFAULT_CODE),
                    access=F8StateAccess.rw,
                    control=F8UiControlSpec(kind=F8UiControlKind.code, language="python"),
                    valueRequired=True,
                    showOnNode=False,
                    editorAssist=pyscript_code_field_editor_assist_payload(),
                ),
                F8StateSpec(
                    name="tickEnabled",
                    label="Tick Enabled",
                    description="Enable onTick scheduler.",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="tickMs",
                    label="Tick Interval (ms)",
                    description="onTick interval in milliseconds.",
                    valueSchema=integer_schema(default=100, minimum=1),
                    access=F8StateAccess.rw,
                    valueRequired=True,
                    showOnNode=False,
                ),
            ],
            commands=[
                F8Command(
                    name="grant_local_exec",
                    description="Grant local execution for this script session.",
                    definitionProtected=True,
                    showOnNode=True,
                    params=[
                        F8CommandParam(
                            name="ttlMs",
                            description="Optional grant TTL in milliseconds.",
                            valueSchema=integer_schema(default=60000, minimum=1),
                            valueRequired=False,
                        )
                    ],
                ),
                F8Command(
                    name="revoke_local_exec",
                    description="Revoke local execution grant.",
                    definitionProtected=True,
                    showOnNode=True,
                    params=[],
                ),
            ],
            dataInPorts=[F8DataPortSpec(name="in", description="Default data input", valueSchema=any_schema(), definitionProtected=False)],
            dataOutPorts=[F8DataPortSpec(name="out", description="Default data output", valueSchema=any_schema(), definitionProtected=False)],
            editPolicy=F8SpecEditPolicy(
                stateFields=editable_collection_edit_policy(),
                commands=editable_collection_edit_policy(),
                dataInPorts=editable_collection_edit_policy(),
                dataOutPorts=editable_collection_edit_policy(),
            ),
        ),
        PythonScriptServiceNode,
        overwrite=True,
    )
    return registry


def create_pyscript_registry() -> RuntimeNodeRegistry:
    runtime_registry = create_runtime_node_registry()
    register_specs(Registry.wrap(runtime_registry))
    return runtime_registry


def shared_pyscript_registry() -> RuntimeNodeRegistry:
    runtime_registry = shared_runtime_node_registry()
    register_specs(Registry.wrap(runtime_registry))
    return runtime_registry
