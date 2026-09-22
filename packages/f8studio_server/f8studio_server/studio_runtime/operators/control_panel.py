from __future__ import annotations

from typing import Any

from f8pysdk.specs import (
    F8CollectionEditPolicy,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry
from f8pysdk.specs import integer_schema

from ..identifiers import SERVICE_CLASS
from .categories import PALETTE_CATEGORY_CONTROL

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
        paletteCategory=PALETTE_CATEGORY_CONTROL,
        operatorClass=OPERATOR_CLASS,
        version="0.0.1",
        label="Control Panel",
        description="Centralized state control panel for wiring key parameters through state edges.",
        tags=["panel", "state", "control", "ui"],
        dataInPorts=[],
        dataOutPorts=[],
        execInPorts=[],
        execOutPorts=[],
        rendererClass="default_op",
        editPolicy=F8SpecEditPolicy(
            stateFields=F8CollectionEditPolicy(canAdd=True, canDelete=True, canEditExisting=True)
        ),
        stateFields=[
            F8StateSpec(
                name="value",
                description="The value of this control panel field.",
                valueSchema=integer_schema(),
                access=F8StateAccess.rw,
                required=False,
                showOnNode=True,
            )
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


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(ControlPanelRuntimeNode.SPEC, ControlPanelRuntimeNode, overwrite=True)
    return registry
