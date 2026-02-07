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
    any_schema,
    boolean_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.print"


class PrintRuntimeNode(OperatorNode):
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
        self._strip = self._coerce_bool((initial_state or {}).get("strip"), default=True)

    @staticmethod
    def _coerce_bool(value: Any, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        return default

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        if str(field) == "strip":
            self._strip = self._coerce_bool(value, default=self._strip)
            return

    async def on_exec(self, exec_id: str | int, _in_port: str | None = None) -> list[str]:
        v = await self.pull("value", ctx_id=exec_id)
        if self._strip:
            if isinstance(v, (bytes, bytearray)):
                try:
                    v = bytes(v).decode("utf-8", errors="replace")
                except Exception:
                    pass
            if isinstance(v, str):
                v = v.strip()
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
    dataInPorts=[F8DataPortSpec(name="value", description="value to print", valueSchema=any_schema())],
    stateFields=[
        F8StateSpec(
            name="strip",
            label="Strip",
            description="If true, strip whitespace/newlines from the start/end of string values before printing.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.wo,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _print_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return PrintRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _print_factory, overwrite=True)

    reg.register_operator_spec(PrintRuntimeNode.SPEC, overwrite=True)
    return reg
