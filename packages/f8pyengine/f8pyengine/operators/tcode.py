from __future__ import annotations

import math
from typing import Any, Final

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    number_schema,
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from ._ports import exec_out_ports

OPERATOR_CLASS: Final[str] = "f8.tcode"

AXES: Final[tuple[str, ...]] = ("L0", "L1", "L2", "R0", "R1", "R2", "V0", "V1", "A0", "A1")


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except Exception:
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _js_round(value: float) -> int:
    """
    Match JavaScript Math.round behavior: halves round away from zero.
    """
    if value >= 0:
        return int(math.floor(value + 0.5))
    return -int(math.floor(abs(value) + 0.5))


class TCodeRuntimeNode(OperatorNode):
    """
    Assembles a TCode v0.3 command string from normalized axis values (0..1).

    Ported from `f8flow/web/.../nodes/tcode.ts`.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._exec_out_ports = exec_out_ports(node, default=["exec"])

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        port_s = str(port)
        if port_s not in ("tcode", "f8/transform/tcode"):
            return None

        interval_ms = _coerce_number(await self.pull("intervalMs", ctx_id=ctx_id))
        if interval_ms is None:
            interval_ms = await self.get_state_value("intervalMs")
            if interval_ms is None:
                interval_ms = self._initial_state.get("intervalMs", 20)
        interval_i = max(1, _js_round(float(interval_ms)))

        commands: list[str] = []
        for axis in AXES:
            raw_value = await self.pull(axis, ctx_id=ctx_id)
            numeric = _coerce_number(raw_value)
            if numeric is None:
                continue
            clamped = min(1.0, max(0.0, float(numeric)))
            payload = _js_round(clamped * 9999.0)
            magnitude = f"{axis}{payload:04d}"
            commands.append(f"{magnitude}I{interval_i:03d}")

        if not commands:
            return ""
        return " ".join(commands) + "\n"

    async def validate_state(
        self, field: str, value: Any, *, ts_ms: int | None = None, meta: dict[str, Any] | None = None
    ) -> Any:
        name = str(field or "").strip()
        if name != "intervalMs":
            return value
        numeric = _coerce_number(value)
        if numeric is None:
            raise ValueError("intervalMs must be a number")
        interval_i = max(1, _js_round(float(numeric)))
        if interval_i > 50000:
            raise ValueError("intervalMs must be <= 50000")
        return interval_i


TCodeRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="TCode",
    description="Generates TCode v0.3 command strings from normalized axis values.",
    tags=["transform", "tcode", "osr", "command", "string"],
    execInPorts=["exec"],
    execOutPorts=["exec"],
    dataInPorts=[
        *[F8DataPortSpec(name=axis, description=f"Axis {axis} (0..1).", valueSchema=number_schema()) for axis in AXES],
        F8DataPortSpec(
            name="intervalMs",
            description="Optional interval override in milliseconds (rounded, min 1).",
            valueSchema=number_schema(default=20, minimum=1, maximum=50000),
        ),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="tcode", description="TCode v0.3 command string", valueSchema=string_schema()),
        F8DataPortSpec(
            name="f8/transform/tcode",
            description="TCode v0.3 command string (alias port id).",
            valueSchema=string_schema(),
        ),
    ],
    stateFields=[
        F8StateSpec(
            name="intervalMs",
            label="Interval (ms)",
            description="Default interval appended as `I###` when `intervalMs` input is not provided.",
            valueSchema=number_schema(default=20, minimum=1, maximum=50000),
            access=F8StateAccess.rw,
            showOnNode=False,
        )
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return TCodeRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(TCodeRuntimeNode.SPEC, overwrite=True)
    return reg
