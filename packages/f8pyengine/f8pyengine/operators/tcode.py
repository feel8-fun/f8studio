from __future__ import annotations

from f8pysdk.codec import parse_number
import math
from typing import Any, Final

from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    complex_object_schema,
    number_schema,
    string_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from ._ports import exec_out_ports

OPERATOR_CLASS: Final[str] = "f8.tcode"

AXES: Final[tuple[str, ...]] = ("L0", "L1", "L2", "R0", "R1", "R2", "V0", "V1", "A0", "A1")

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

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        port_s = str(port)
        if port_s != "tcode":
            return None

        interval_ms = parse_number(await self.pull("intervalMs", ctx_id=ctx_id))
        if interval_ms is None:
            interval_ms = await self.get_state_value("intervalMs")
            if interval_ms is None:
                interval_ms = self._initial_state.get("intervalMs", 20)
        interval_i = max(1, _js_round(float(interval_ms)))

        frame_raw = await self.pull("frame", ctx_id=ctx_id)
        frame_axes = _frame_axes(frame_raw)
        commands: list[str] = []
        for axis in AXES:
            raw_value = frame_axes.get(axis) if frame_axes is not None else await self.pull(axis, ctx_id=ctx_id)
            numeric = parse_number(raw_value)
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
        numeric = parse_number(value)
        if numeric is None:
            raise ValueError("intervalMs must be a number")
        interval_i = max(1, _js_round(float(numeric)))
        if interval_i > 50000:
            raise ValueError("intervalMs must be <= 50000")
        return interval_i

TCodeRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.signal",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="TCode",
    description="Generates TCode v0.3 command strings from normalized axis values.",
    tags=["transform", "tcode", "osr", "command", "string"],
    dataInPorts=[
        F8DataPortSpec(
            name="frame",
            description="Optional atomic axis frame. When present, scalar axis inputs are not pulled.",
            valueSchema=complex_object_schema(
                properties={"axes": complex_object_schema(properties={axis: number_schema() for axis in AXES})}
            ),
            showOnNode=True,
        ),
        *[
            F8DataPortSpec(
                name=axis,
                description=f"Axis {axis} (0..1).",
                valueSchema=number_schema(minimum=0.0, maximum=1.0),
                showOnNode=i < 6,
            )
            for i, axis in enumerate(AXES)
        ],
        F8DataPortSpec(
            name="intervalMs",
            description="Optional interval override in milliseconds (rounded, min 1).",
            valueSchema=number_schema(default=20, minimum=1, maximum=50000),
            showOnNode=False,
        ),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="tcode", description="TCode v0.3 command string", valueSchema=string_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="intervalMs",
            label="Interval (ms)",
            description="Default interval appended as `I###` when `intervalMs` input is not provided.",
            valueSchema=number_schema(default=20, minimum=1, maximum=50000),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        )
    ],
)


def _frame_axes(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    nested = value.get("axes")
    payload = nested if isinstance(nested, dict) else value
    return payload if any(axis in payload for axis in AXES) else None

def register_operator(registry: Registry) -> Registry:
    registry.register_operator(TCodeRuntimeNode.SPEC, TCodeRuntimeNode, overwrite=True)
    return registry
