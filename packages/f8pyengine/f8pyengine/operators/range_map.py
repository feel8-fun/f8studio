from __future__ import annotations

import math
from typing import Any, Callable

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

OPERATOR_CLASS = "f8.range_map"

CURVE_LINEAR = "LINEAR"
CURVE_SMOOTHSTEP = "SMOOTHSTEP"
CURVE_SMOOTHERSTEP = "SMOOTHERSTEP"
CURVE_EASE_IN = "EASE_IN"
CURVE_EASE_OUT = "EASE_OUT"
CURVE_EASE_IN_OUT = "EASE_IN_OUT"

CURVE_CHOICES = (
    CURVE_LINEAR,
    CURVE_SMOOTHSTEP,
    CURVE_SMOOTHERSTEP,
    CURVE_EASE_IN,
    CURVE_EASE_OUT,
    CURVE_EASE_IN_OUT,
)


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


def _normalize_curve(value: Any) -> str:
    name = str(value or "").strip().upper()
    if name in CURVE_CHOICES:
        return name
    return CURVE_LINEAR


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _ease_in(t: float) -> float:
    return t * t


def _ease_out(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


def _ease_in_out(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - 2.0 * (1.0 - t) * (1.0 - t)


_CURVE_FN: dict[str, Callable[[float], float]] = {
    CURVE_LINEAR: lambda t: t,
    CURVE_SMOOTHSTEP: _smoothstep,
    CURVE_SMOOTHERSTEP: _smootherstep,
    CURVE_EASE_IN: _ease_in,
    CURVE_EASE_OUT: _ease_out,
    CURVE_EASE_IN_OUT: _ease_in_out,
}


class RangeMapRuntimeNode(OperatorNode):
    """
    Clip input to [inMin, inMax], remap to [outMin, outMax] with optional curve.
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})

        self._in_min = float(_coerce_number(self._initial_state.get("inMin")) or 0.0)
        self._in_max = float(_coerce_number(self._initial_state.get("inMax")) or 1.0)
        self._out_min = float(_coerce_number(self._initial_state.get("outMin")) or 0.0)
        self._out_max = float(_coerce_number(self._initial_state.get("outMax")) or 1.0)
        self._curve = _normalize_curve(self._initial_state.get("curve"))

        self._last_output: float | None = None
        self._last_input: float | None = None
        self._last_ctx_id: str | int | None = None
        self._dirty = True

    def _apply_state(self, name: str, value: Any) -> None:
        if name == "inMin":
            numeric = _coerce_number(value)
            if numeric is not None:
                self._in_min = float(numeric)
        elif name == "inMax":
            numeric = _coerce_number(value)
            if numeric is not None:
                self._in_max = float(numeric)
        elif name == "outMin":
            numeric = _coerce_number(value)
            if numeric is not None:
                self._out_min = float(numeric)
        elif name == "outMax":
            numeric = _coerce_number(value)
            if numeric is not None:
                self._out_max = float(numeric)
        elif name == "curve":
            self._curve = _normalize_curve(value)
        self._dirty = True

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "")
        if name in {"inMin", "inMax", "outMin", "outMax", "curve"}:
            self._apply_state(name, value)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port) != "value":
            return None

        raw_value = await self.pull("value", ctx_id=ctx_id)
        numeric = _coerce_number(raw_value)
        if numeric is None:
            return self._last_output

        if not self._dirty:
            if ctx_id is not None and ctx_id == self._last_ctx_id and numeric == self._last_input:
                return self._last_output
            if ctx_id is None and numeric == self._last_input:
                return self._last_output

        in_min = self._in_min
        in_max = self._in_max
        out_min = self._out_min
        out_max = self._out_max

        if in_min > in_max:
            in_min, in_max = in_max, in_min
        if out_min > out_max:
            out_min, out_max = out_max, out_min

        if in_max - in_min == 0.0:
            output = out_min
        else:
            clipped = min(in_max, max(in_min, float(numeric)))
            t = (clipped - in_min) / (in_max - in_min)
            curve_fn = _CURVE_FN.get(self._curve, _CURVE_FN[CURVE_LINEAR])
            t = curve_fn(t)
            output = out_min + t * (out_max - out_min)

        self._last_output = float(output)
        self._last_input = float(numeric)
        self._last_ctx_id = ctx_id
        self._dirty = False
        return self._last_output


RangeMapRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Range Map",
    description="Clip input to [inMin,inMax] then remap to [outMin,outMax] with a curve.",
    tags=["map", "range", "normalize", "curve", "transform"],
    dataInPorts=[
        F8DataPortSpec(name="value", description="Input value.", valueSchema=number_schema(), required=False),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="value", description="Mapped output.", valueSchema=number_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="inMin",
            label="Input Min",
            description="Input range minimum.",
            valueSchema=number_schema(default=0.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="inMax",
            label="Input Max",
            description="Input range maximum.",
            valueSchema=number_schema(default=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="outMin",
            label="Output Min",
            description="Output range minimum.",
            valueSchema=number_schema(default=0.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="outMax",
            label="Output Max",
            description="Output range maximum.",
            valueSchema=number_schema(default=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="curve",
            label="Curve",
            description="Mapping curve.",
            valueSchema=string_schema(default=CURVE_LINEAR, enum=list(CURVE_CHOICES)),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return RangeMapRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(RangeMapRuntimeNode.SPEC, overwrite=True)
    return reg
