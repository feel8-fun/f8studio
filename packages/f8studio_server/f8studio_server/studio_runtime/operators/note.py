from __future__ import annotations

from typing import Any

from f8pysdk.specs import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    string_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..identifiers import SERVICE_CLASS
from .categories import PALETTE_CATEGORY_CANVAS

OPERATOR_CLASS = "f8.note"
RENDERER_CLASS = "note_markdown"
_DEFAULT_NOTE_CONTENT = "# Note\n\n- Write tips here.\n- Supports **Markdown**.\n"


class NoteRuntimeNode(OperatorNode):
    """
    Studio-only note node.

    This runtime node intentionally has no data/exec behavior and only serves
    as a persistent state carrier for UI note content.
    """

    SPEC = F8OperatorSpec(
        schemaVersion=F8OperatorSchemaVersion.f8operator_1,
        serviceClass=SERVICE_CLASS,
        paletteCategory=PALETTE_CATEGORY_CANVAS,
        operatorClass=OPERATOR_CLASS,
        version="0.0.1",
        label="Note",
        description="A markdown note node for tips and reminders inside the canvas.",
        tags=["note", "markdown", "ui", "tips"],
        dataInPorts=[],
        dataOutPorts=[],
        execInPorts=[],
        execOutPorts=[],
        rendererClass=RENDERER_CLASS,
        stateFields=[
            F8StateSpec(
                name="content",
                label="Content",
                description="Markdown note content shown in the node.",
                valueSchema=string_schema(default=_DEFAULT_NOTE_CONTENT),
                access=F8StateAccess.rw,
                required=True,
                showOnNode=False,
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


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(NoteRuntimeNode.SPEC, NoteRuntimeNode, overwrite=True)
    return registry
