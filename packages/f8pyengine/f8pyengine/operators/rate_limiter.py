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
from ._ports import exec_out_ports

OPERATOR_CLASS = "f8.rate_limiter"

_EPS = 1e-9


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


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


class RateLimiterRuntimeNode(RuntimeNode):
    """
    Rate limiter for a normalized signal (typically 0..1).

    Behavior:
    - Input is optionally clipped to [inMin,inMax] before limiting.
    - Output moves toward input but cannot exceed maxRateUp/maxRateDown (units/sec).
    - Optional acceleration limiting (maxAccel, units/sec^2) smooths velocity changes.

    This preserves "trend" best-effort: rising inputs cause rising output (until it catches up),
    falling inputs cause falling output, while constraining slope.
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

        self._in_min = float(_coerce_number(self._initial_state.get("inMin")) or 0.0)
        self._in_max = float(_coerce_number(self._initial_state.get("inMax")) or 1.0)
        self._max_rate_up = float(_coerce_number(self._initial_state.get("maxRateUp")) or 2.0)
        self._max_rate_down = float(_coerce_number(self._initial_state.get("maxRateDown")) or 2.0)
        self._max_accel = float(_coerce_number(self._initial_state.get("maxAccel")) or 0.0)

        self._y: float | None = None
        self._v: float = 0.0
        self._last_time_s: float | None = None

        self._last_out: float | None = None
        self._last_in: float | None = None
        self._last_ctx_id: str | int | None = None
        self._dirty = True

    def _apply_state(self, name: str, value: Any) -> None:
        numeric = _coerce_number(value)
        if name in {"inMin", "inMax"}:
            if numeric is None:
                return
            if name == "inMin":
                self._in_min = float(numeric)
            else:
                self._in_max = float(numeric)
            self._dirty = True
            return

        if name in {"maxRateUp", "maxRateDown", "maxAccel"}:
            if numeric is None:
                return
            if name == "maxRateUp":
                self._max_rate_up = max(0.0, float(numeric))
            elif name == "maxRateDown":
                self._max_rate_down = max(0.0, float(numeric))
            else:
                self._max_accel = max(0.0, float(numeric))
            self._dirty = True
            return

    def _reset_time_base(self) -> None:
        self._last_time_s = None

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        # Freeze/resume without a large dt step.
        self._reset_time_base()

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        name = str(field or "").strip()
        if name in {"inMin", "inMax", "maxRateUp", "maxRateDown", "maxAccel"}:
            self._apply_state(name, value)

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    def _step(self, x: float) -> float:
        now_s = time.monotonic()
        if self._y is None:
            self._y = float(x)
            self._v = 0.0
            self._last_time_s = now_s
            return float(self._y)

        if self._last_time_s is None:
            # Re-attached / resumed: don't jump due to large dt.
            self._last_time_s = now_s
            return float(self._y)

        dt = max(1e-6, now_s - self._last_time_s)
        self._last_time_s = now_s

        in_min = self._in_min
        in_max = self._in_max
        if in_min > in_max:
            in_min, in_max = in_max, in_min
        x_clip = _clamp(float(x), in_min, in_max)

        y = float(self._y)
        err = x_clip - y

        # Desired velocity to reach target in one step, then rate-limit it.
        v_des = err / dt
        if v_des > 0.0:
            v_des = min(v_des, max(0.0, self._max_rate_up))
        else:
            v_des = max(v_des, -max(0.0, self._max_rate_down))

        if self._max_accel > 0.0:
            max_dv = self._max_accel * dt
            dv = _clamp(v_des - self._v, -max_dv, max_dv)
            self._v += dv
        else:
            self._v = v_des

        y_new = y + self._v * dt

        # Prevent overshoot: if we crossed the target, snap to it and stop.
        if err != 0.0 and (x_clip - y_new) * err < 0.0:
            y_new = x_clip
            self._v = 0.0

        # Clamp output to the same range (common normalized-signal expectation).
        y_new = _clamp(float(y_new), in_min, in_max)
        self._y = float(y_new)
        return float(self._y)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        if str(port) != "value":
            return None

        raw = await self.pull("value", ctx_id=ctx_id)
        numeric = _coerce_number(raw)
        if numeric is None:
            return self._last_out

        if not self._dirty:
            if ctx_id is not None and ctx_id == self._last_ctx_id and numeric == self._last_in:
                return self._last_out
            if ctx_id is None and numeric == self._last_in:
                return self._last_out

        out = self._step(float(numeric))
        self._last_out = float(out)
        self._last_in = float(numeric)
        self._last_ctx_id = ctx_id
        self._dirty = False
        return self._last_out


RateLimiterRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Rate Limiter",
    description="Limits the rate of change (and optionally acceleration) of an input signal.",
    tags=["signal", "limit", "rate", "slew", "smoothing", "transform"],
    execInPorts=["exec"],
    execOutPorts=["exec"],
    dataInPorts=[F8DataPortSpec(name="value", description="Input value.", valueSchema=number_schema(), required=False)],
    dataOutPorts=[F8DataPortSpec(name="value", description="Rate-limited output.", valueSchema=number_schema())],
    stateFields=[
        F8StateSpec(
            name="inMin",
            label="Input Min",
            description="Input/output clamp minimum (typical 0).",
            valueSchema=number_schema(default=0.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="inMax",
            label="Input Max",
            description="Input/output clamp maximum (typical 1).",
            valueSchema=number_schema(default=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="maxRateUp",
            label="Max Rate Up",
            description="Maximum rising rate (units/sec).",
            valueSchema=number_schema(default=2.0, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="maxRateDown",
            label="Max Rate Down",
            description="Maximum falling rate (units/sec).",
            valueSchema=number_schema(default=2.0, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="maxAccel",
            label="Max Accel",
            description="Maximum acceleration (units/sec^2). 0 disables acceleration limiting.",
            valueSchema=number_schema(default=0.0, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return RateLimiterRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(RateLimiterRuntimeNode.SPEC, overwrite=True)
    return reg
