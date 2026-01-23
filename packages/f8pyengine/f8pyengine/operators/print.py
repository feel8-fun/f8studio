from __future__ import annotations

import math
import time
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    number_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.print"


class PrintRuntimeNode(RuntimeNode):
    """
    Prints incoming values.

    For the demo flow, printing happens on data arrival (no exec required).
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

    async def on_exec(self, exec_id: str | int, _in_port: str | None = None) -> list[str]:
        v = await self.pull("value", ctx_id=exec_id)
        print(f"[{self.node_id}] exec={exec_id} value={v}")
        return []


PrintRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Print",
    description="Exec-driven printer (pulls `value` and prints).",
    tags=["debug", "console", "print"],
    execInPorts=["exec"],
    dataInPorts=[F8DataPortSpec(name="value", description="value to print", valueSchema=number_schema())],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _print_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return PrintRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _print_factory, overwrite=True)

    reg.register_operator_spec(PrintRuntimeNode.SPEC, overwrite=True)
    return reg
