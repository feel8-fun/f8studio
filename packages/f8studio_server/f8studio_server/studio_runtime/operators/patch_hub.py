from __future__ import annotations

from f8pysdk.specs import exec_port_specs

from typing import Any

import msgspec
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    editable_collection_edit_policy,
)
from f8pysdk.codec import copy_model
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..identifiers import SERVICE_CLASS
from .categories import PALETTE_CATEGORY_ROUTING

OPERATOR_CLASS = "f8.patch_hub"
RENDERER_CLASS = "patch_hub"


def _coerce_port_spec(port: F8DataPortSpec | None, *, name: str) -> F8DataPortSpec:
    base = port or F8DataPortSpec(name=name, valueSchema=any_schema(), definitionProtected=False)
    return copy_model(
        base,
        update={
            "name": name,
            "definitionProtected": False,
            "showOnNode": True,
            "valueSchema": base.valueSchema or any_schema(),
        },
    )


def _coerce_state_terminal(field: F8StateSpec | None, *, name: str) -> F8StateSpec:
    base = field or F8StateSpec(name=name, valueSchema=any_schema(), access=F8StateAccess.rw)
    return copy_model(
        base,
        update={
            "name": name,
            "access": F8StateAccess.rw,
            "valueRequired": False,
            "showOnNode": True,
            "control": msgspec.UNSET,
            "valueSchema": base.valueSchema or any_schema(),
        },
    )


def normalize_patch_hub_spec(spec: F8OperatorSpec) -> F8OperatorSpec:
    data_name_order: list[str] = []
    data_by_name: dict[str, F8DataPortSpec] = {}

    for port in list(spec.dataInPorts or []):
        name = str(port.name or "").strip()
        if not name:
            continue
        if name not in data_by_name:
            data_name_order.append(name)
        data_by_name[name] = _coerce_port_spec(port, name=name)

    for port in list(spec.dataOutPorts or []):
        name = str(port.name or "").strip()
        if not name:
            continue
        if name not in data_by_name:
            data_name_order.append(name)
            data_by_name[name] = _coerce_port_spec(port, name=name)

    data_terminals = [data_by_name[name] for name in data_name_order if name in data_by_name]

    state_name_order: list[str] = []
    state_by_name: dict[str, F8StateSpec] = {}
    for field in list(spec.stateFields or []):
        name = str(field.name or "").strip()
        if not name:
            continue
        if name not in state_by_name:
            state_name_order.append(name)
        state_by_name[name] = _coerce_state_terminal(field, name=name)
    state_terminals = [state_by_name[name] for name in state_name_order if name in state_by_name]

    return copy_model(
        spec,
        update={
            "dataInPorts": data_terminals,
            "dataOutPorts": [copy_model(port, update={}) for port in data_terminals],
            "stateFields": state_terminals,
            "execInPorts": [],
            "execOutPorts": [],
            "rendererClass": RENDERER_CLASS,
        },
    )
class PatchHubRuntimeNode(OperatorNode):
    """
    Editor-only passthrough placeholder.

    The runtime compiler removes patch hubs before validation/deploy, but we
    still register a no-op runtime node so the spec remains self-contained.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        del initial_state
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[str(port.name or "") for port in list(node.dataInPorts or []) if str(port.name or "").strip()],
            data_out_ports=[str(port.name or "") for port in list(node.dataOutPorts or []) if str(port.name or "").strip()],
            state_fields=[str(field.name or "") for field in list(node.stateFields or []) if str(field.name or "").strip()],
            exec_in_ports=[],
            exec_out_ports=[],
        )


PatchHubRuntimeNode.SPEC = normalize_patch_hub_spec(
    F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass=SERVICE_CLASS,
        paletteCategory=PALETTE_CATEGORY_ROUTING,
        operatorClass=OPERATOR_CLASS,
        version="0.0.1",
        label="Patch Hub",
        description="Compile-time patch bay for tidying canvas wiring and fan-out.",
        tags=["studio", "hub", "patch", "connector", "routing"],
        rendererClass=RENDERER_CLASS,
        dataInPorts=[
            F8DataPortSpec(
                name="data",
                description="Starter data terminal.",
                valueSchema=any_schema(),
                definitionProtected=False,
                showOnNode=True,
            )
        ],
        dataOutPorts=[
            F8DataPortSpec(
                name="data",
                description="Starter data terminal.",
                valueSchema=any_schema(),
                definitionProtected=False,
                showOnNode=True,
            )
        ],
        execInPorts=exec_port_specs([]),
        execOutPorts=exec_port_specs([]),
        stateFields=[
            F8StateSpec(
                name="state",
                label="State",
                description="Starter state terminal.",
                valueSchema=any_schema(),
                access=F8StateAccess.rw,
                valueRequired=False,
                showOnNode=True,
            )
        ],
        editPolicy=F8SpecEditPolicy(
            dataInPorts=editable_collection_edit_policy(),
            dataOutPorts=editable_collection_edit_policy(),
            stateFields=editable_collection_edit_policy(),
        ),
    )
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(PatchHubRuntimeNode.SPEC, PatchHubRuntimeNode, overwrite=True)
    return registry
