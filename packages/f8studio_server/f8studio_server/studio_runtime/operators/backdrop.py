from __future__ import annotations

from typing import Any

from f8pysdk.specs import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..identifiers import SERVICE_CLASS
from .categories import PALETTE_CATEGORY_CANVAS

OPERATOR_CLASS = "f8.backdrop"
RENDERER_CLASS = "backdrop"


class BackdropRuntimeNode(OperatorNode):
    """Studio-only backdrop node used to label and frame regions on the canvas."""

    SPEC = F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass=SERVICE_CLASS,
        paletteCategory=PALETTE_CATEGORY_CANVAS,
        operatorClass=OPERATOR_CLASS,
        version="0.0.1",
        label="Backdrop",
        description="A lightweight UI-only backdrop for grouping and labeling areas in the canvas.",
        tags=["backdrop", "group", "ui", "canvas", "label"],
        dataInPorts=[],
        dataOutPorts=[],
        execInPorts=[],
        execOutPorts=[],
        rendererClass=RENDERER_CLASS,
        stateFields=[],
    )

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        del initial_state
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[str(port.name or "") for port in list(node.dataInPorts or [])],
            data_out_ports=[str(port.name or "") for port in list(node.dataOutPorts or [])],
            state_fields=[],
            exec_in_ports=list(node.execInPorts or []),
            exec_out_ports=list(node.execOutPorts or []),
        )


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(BackdropRuntimeNode.SPEC, BackdropRuntimeNode, overwrite=True)
    return registry
