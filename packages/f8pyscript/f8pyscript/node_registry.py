from __future__ import annotations

from typing import Any

from f8pysdk import (
    F8Command,
    F8CommandParam,
    F8DataPortSpec,
    F8RuntimeNode,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    array_schema,
    boolean_schema,
    integer_schema,
    string_schema,
)
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from .constants import SERVICE_CLASS
from .service_node import DEFAULT_CODE, PythonScriptServiceNode


def register_specs(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=SERVICE_CLASS,
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
                    uiControl="code",
                    uiLanguage="python",
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="lastError",
                    label="Last Error",
                    description="Last script compile/runtime error.",
                    valueSchema=string_schema(default=""),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="active",
                    label="Active",
                    description="Service lifecycle active flag.",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="tickEnabled",
                    label="Tick Enabled",
                    description="Enable onTick scheduler.",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="tickMs",
                    label="Tick Interval (ms)",
                    description="onTick interval in milliseconds.",
                    valueSchema=integer_schema(default=100, minimum=1),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="commands",
                    label="Command Declarations",
                    description="Optional UI command declaration list.",
                    valueSchema=array_schema(items=any_schema()),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="localExecGranted",
                    label="Local Exec Granted",
                    description="Readonly local execution grant state.",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.ro,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="localExecGrantTsMs",
                    label="Local Exec Grant Ts (ms)",
                    description="Grant timestamp in milliseconds.",
                    valueSchema=integer_schema(default=0, minimum=0),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="execCount",
                    label="Exec Count",
                    description="Total local exec invocations.",
                    valueSchema=integer_schema(default=0, minimum=0),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="lastExecTsMs",
                    label="Last Exec Ts (ms)",
                    description="Last local exec timestamp.",
                    valueSchema=integer_schema(default=0, minimum=0),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                ),
            ],
            commands=[
                F8Command(
                    name="grant_local_exec",
                    description="Grant local execution for this script session.",
                    required=True,
                    showOnNode=True,
                    params=[
                        F8CommandParam(
                            name="ttlMs",
                            description="Optional grant TTL in milliseconds.",
                            valueSchema=integer_schema(default=60000, minimum=1),
                            required=False,
                        )
                    ],
                ),
                F8Command(
                    name="revoke_local_exec",
                    description="Revoke local execution grant.",
                    required=True,
                    showOnNode=True,
                    params=[],
                ),
                F8Command(name="restart_script", description="Recompile and restart script.", required=True, showOnNode=False, params=[]),
                F8Command(name="status", description="Get runtime status.", required=True, showOnNode=False, params=[]),
            ],
            dataInPorts=[F8DataPortSpec(name="in", description="Default data input", valueSchema=any_schema())],
            dataOutPorts=[F8DataPortSpec(name="out", description="Default data output", valueSchema=any_schema())],
            editableStateFields=False,
            editableCommands=True,
            editableDataInPorts=True,
            editableDataOutPorts=True,
        ),
        overwrite=True,
    )

    def _service_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return PythonScriptServiceNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register_service(SERVICE_CLASS, _service_factory, overwrite=True)
    return reg
