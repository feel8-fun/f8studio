from __future__ import annotations

from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8RuntimeNode,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    string_schema,
)
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from .constants import EXPR_SERVICE_CLASS
from .expr_service_node import DEFAULT_CODE, PythonExprServiceNode


def register_expr_specs(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()
    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=EXPR_SERVICE_CLASS,
            version="0.0.1",
            label="Python Expr Service",
            description="Standalone expression runtime service for simplified data-flow transforms.",
            tags=["python", "expr", "service"],
            rendererClass="default_svc",
            stateFields=[
                F8StateSpec(
                    name="code",
                    label="Expr",
                    description="Single-line expression. Available names: inputs + identifier-safe input ports.",
                    valueSchema=string_schema(default=DEFAULT_CODE),
                    access=F8StateAccess.rw,
                    uiControl="wrapline",
                    uiLanguage="python",
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="allowNumpy",
                    label="Allow Numpy",
                    description="Enable numpy calls in expressions (np.*, numpy.*).",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.wo,
                    uiControl="toggle",
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="unpackDictOutputs",
                    label="Unpack Dict Outputs",
                    description="When enabled, dict results are emitted per matching output port key.",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.wo,
                    uiControl="toggle",
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="lastError",
                    label="Last Error",
                    description="Last expression compile/eval error.",
                    valueSchema=string_schema(default=""),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                ),
            ],
            dataInPorts=[F8DataPortSpec(name="msg", description="Default input value.", valueSchema=any_schema(), required=True)],
            dataOutPorts=[
                F8DataPortSpec(name="out", description="Default expression output value.", valueSchema=any_schema(), required=True)
            ],
            editableStateFields=False,
            editableCommands=False,
            editableDataInPorts=True,
            editableDataOutPorts=True,
        ),
        overwrite=True,
    )

    def _service_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return PythonExprServiceNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register_service(EXPR_SERVICE_CLASS, _service_factory, overwrite=True)
    return reg
