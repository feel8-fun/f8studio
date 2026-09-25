from __future__ import annotations

from f8pysdk.specs import exec_port_specs

from typing import Any

from f8pysdk.codec import coerce_flag
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

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
        self._strip = coerce_flag((initial_state or {}).get("strip"), default=True)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        if str(field) == "strip":
            self._strip = coerce_flag(value, default=self._strip)
            return

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if port == "value":
            v = value
            if self._strip:
                if isinstance(v, (bytes, bytearray)):
                    v = bytes(v).decode("utf-8", errors="replace")
                if isinstance(v, str):
                    v = v.strip()
            print(f"[{self.node_id}] value={v}")

    async def on_exec(self, exec_id: str | int, _in_port: str | None = None) -> list[str]:
        v = await self.pull("value", ctx_id=exec_id)
        if self._strip:
            if isinstance(v, (bytes, bytearray)):
                v = bytes(v).decode("utf-8", errors="replace")
            if isinstance(v, str):
                v = v.strip()
        print(f"[{self.node_id}] exec={exec_id} value={v}")
        return []


PrintRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.debug",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Print",
    description="Exec-driven printer (pulls `value` and prints).",
    tags=["debug", "console", "print"],
    execInPorts=exec_port_specs(["exec"]),
    dataInPorts=[F8DataPortSpec(name="value", description="value to print", valueSchema=any_schema())],
    stateFields=[
        F8StateSpec(
            name="strip",
            label="Strip",
            description="If true, strip whitespace/newlines from the start/end of string values before printing.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(PrintRuntimeNode.SPEC, PrintRuntimeNode, overwrite=True)
    return registry
