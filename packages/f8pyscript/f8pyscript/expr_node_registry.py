from __future__ import annotations

from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
    F8DataPortSpec,
    F8SpecEditPolicy,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    editable_collection_edit_policy,
    string_schema,
)
from f8pysdk.registry import Registry, RuntimeNodeRegistry, create_runtime_node_registry, shared_runtime_node_registry

from .constants import EXPR_SERVICE_CLASS
from .expr_service_node import PythonExprServiceNode
from .expr_templates import DEFAULT_CODE


def register_expr_specs(registry: Registry) -> Registry:
    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=EXPR_SERVICE_CLASS,
            paletteCategory="svc",
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
                    required=True,
                    control=F8UiControlSpec(kind=F8UiControlKind.textarea, language="python"),
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="allowNumpy",
                    label="Allow Numpy",
                    description="Enable numpy calls in expressions (np.*, numpy.*).",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.wo,
                    required=True,
                    control=F8UiControlSpec(kind=F8UiControlKind.toggle),
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="unpackDictOutputs",
                    label="Unpack Dict Outputs",
                    description="When enabled, dict results are emitted per matching output port key.",
                    valueSchema=boolean_schema(default=False),
                    access=F8StateAccess.wo,
                    required=True,
                    control=F8UiControlSpec(kind=F8UiControlKind.toggle),
                    showOnNode=False,
                ),
            ],
            dataInPorts=[F8DataPortSpec(name="msg", description="Default input value.", valueSchema=any_schema(), required=False)],
            dataOutPorts=[
                F8DataPortSpec(name="out", description="Default expression output value.", valueSchema=any_schema(), required=False)
            ],
            editPolicy=F8SpecEditPolicy(
                dataInPorts=editable_collection_edit_policy(),
                dataOutPorts=editable_collection_edit_policy(),
            ),
        ),
        PythonExprServiceNode,
        overwrite=True,
    )
    return registry


def create_pyexpr_registry() -> RuntimeNodeRegistry:
    runtime_registry = create_runtime_node_registry()
    register_expr_specs(Registry.wrap(runtime_registry))
    return runtime_registry


def shared_pyexpr_registry() -> RuntimeNodeRegistry:
    runtime_registry = shared_runtime_node_registry()
    register_expr_specs(Registry.wrap(runtime_registry))
    return runtime_registry
