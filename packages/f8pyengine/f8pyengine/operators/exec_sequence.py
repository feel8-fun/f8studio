from __future__ import annotations

from f8pysdk.specs import exec_port_specs

from typing import Any

from f8pysdk.specs import (
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    editable_collection_edit_policy,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from ._ports import exec_out_ports

OPERATOR_CLASS = "f8.exec_sequence"


class ExecSequenceRuntimeNode(OperatorNode):
    """
    UE-style Sequence:
    - one exec input
    - N exec outputs, executed in order (0,1,2...) under DFS scheduling
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._exec_out_ports = exec_out_ports(node)

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)


ExecSequenceRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.execution",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Sequence",
    description="Exec flow splitter: triggers its exec outputs in order (requires DFS scheduling).",
    tags=["execution", "flow", "sequence", "branch"],
    execInPorts=exec_port_specs(["exec"]),
    execOutPorts=exec_port_specs(["0", "1", "2"]),
    editPolicy=F8SpecEditPolicy(execOutPorts=editable_collection_edit_policy()),
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(ExecSequenceRuntimeNode.SPEC, ExecSequenceRuntimeNode, overwrite=True)
    return registry

