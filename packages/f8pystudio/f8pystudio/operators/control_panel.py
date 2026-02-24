from __future__ import annotations

from typing import Any

from f8pysdk import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    array_schema,
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.control_panel"


class ControlPanelRuntimeNode(OperatorNode):
    """
    Studio-only quick state panel.

    This runtime node intentionally does not process data/exec/state callbacks.
    It serves as a state carrier so users can centralize key parameters and
    rely on existing state-edge propagation/subscription.
    """

    SPEC = F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass=SERVICE_CLASS,
        operatorClass=OPERATOR_CLASS,
        version="0.0.1",
        label="ControlPanel",
        description="Centralized state control panel for wiring key parameters through state edges.",
        tags=["panel", "state", "control", "ui"],
        dataInPorts=[],
        dataOutPorts=[],
        execInPorts=[],
        execOutPorts=[],
        rendererClass="",
        editableStateFields=True,
        stateFields=[
            F8StateSpec(
                name="value",
                label="Value",
                description="Selected/current value published to downstream subscribers.",
                valueSchema=string_schema(default=""),
                access=F8StateAccess.rw,
                uiControl="select:[options]",
                showOnNode=True,
            ),
            F8StateSpec(
                name="options",
                label="Options",
                description="Option pool used by the `value` dropdown control.",
                valueSchema=array_schema(items=string_schema()),
                access=F8StateAccess.wo,
                showOnNode=True,
            ),
        ],
    )

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        del initial_state
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
            exec_in_ports=list(node.execInPorts or []),
            exec_out_ports=list(node.execOutPorts or []),
        )


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return ControlPanelRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(ControlPanelRuntimeNode.SPEC, overwrite=True)
    return reg
