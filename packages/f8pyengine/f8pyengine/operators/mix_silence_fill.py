from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Final

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    integer_schema,
    number_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from ._ports import exec_out_ports

OPERATOR_CLASS: Final[str] = "f8.mix_silence_fill"


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


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@dataclass
class _Slew:
    """
    Exponential smoothing toward a target with time-based alpha.

    dt_s drives the smoothing factor: alpha = 1 - exp(-dt/tau).
    """

    value: float
    target: float

    def step(self, *, dt_s: float, tau_s: float) -> float:
        dt = max(0.0, float(dt_s))
        tau = max(0.0, float(tau_s))
        if tau <= 0.0:
            self.value = float(self.target)
            return float(self.value)
        alpha = 1.0 - math.exp(-dt / tau)
        self.value = float(self.value + alpha * (self.target - self.value))
        return float(self.value)


class MixSilenceFillRuntimeNode(OperatorNode):
    """
    Mix node: output A by default, but when A is "silent" for some time, smoothly
    crossfade to B as filler. When A becomes active again, crossfade back to A.

    Inputs:
    - A: primary signal
    - B: filler/default signal

    Outputs:
    - out: mixed output
    - alpha: crossfade factor (0=A, 1=B)
    - silent: 1 if A considered silent else 0
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

        self._last_time_s: float | None = None
        self._last_active_s: float | None = None
        self._last_a: float | None = None

        self._alpha = _Slew(value=0.0, target=0.0)

        self._last_ctx_id: str | int | None = None
        self._cache: dict[str, float] = {}
        self._silence_s = 0.5
        self._delta_threshold = 0.001
        self._tau_s = 0.2 / 3.0
        self._refresh_runtime_params(self._initial_state)

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        _ = active
        _ = meta
        # Freeze/resume without a large dt step.
        self._last_time_s = None

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name not in ("silenceMs", "deltaThreshold", "fadeMs"):
            return
        self._refresh_runtime_params({name: value})

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        p = str(port)
        if p not in ("out", "alpha", "silent"):
            return None
        if ctx_id is not None and ctx_id == self._last_ctx_id and p in self._cache:
            return self._cache.get(p)

        out = await self._step(ctx_id=ctx_id)
        self._last_ctx_id = ctx_id
        self._cache = dict(out)
        return self._cache.get(p)

    async def _step(self, *, ctx_id: str | int | None) -> dict[str, float]:
        now_s = time.monotonic()
        if self._last_time_s is None:
            dt_s = 0.0
        else:
            dt_s = max(0.0, now_s - float(self._last_time_s))
        self._last_time_s = now_s

        a_raw = await self.pull("A", ctx_id=ctx_id)
        b_raw = await self.pull("B", ctx_id=ctx_id)
        a_num = _coerce_number(a_raw)
        b_num = _coerce_number(b_raw)

        silence_s = float(self._silence_s)
        eps = float(self._delta_threshold)
        tau_s = float(self._tau_s)

        # Activity detection: if A changes "enough", mark it active.
        if a_num is not None:
            if self._last_a is None:
                self._last_active_s = now_s
            else:
                if abs(float(a_num) - float(self._last_a)) > eps:
                    self._last_active_s = now_s
            self._last_a = float(a_num)

        # If A is absent, treat it as potentially silent (after silenceMs).
        if self._last_active_s is None:
            self._last_active_s = now_s

        silent = 0.0
        if silence_s <= 0.0:
            silent = 0.0
        else:
            if (now_s - float(self._last_active_s)) >= silence_s:
                silent = 1.0

        self._alpha.target = 1.0 if silent >= 1.0 else 0.0
        alpha = _clamp01(self._alpha.step(dt_s=dt_s, tau_s=tau_s))

        # Mix with sensible fallbacks.
        if a_num is None and b_num is None:
            out = self._cache.get("out", 0.0)
            return {"out": float(out), "alpha": float(alpha), "silent": float(silent)}
        if a_num is None:
            out = float(b_num)
            return {"out": float(out), "alpha": float(alpha), "silent": float(silent)}
        if b_num is None:
            out = float(a_num)
            return {"out": float(out), "alpha": float(alpha), "silent": float(silent)}

        out = (1.0 - alpha) * float(a_num) + alpha * float(b_num)
        return {"out": float(out), "alpha": float(alpha), "silent": float(silent)}

    def _refresh_runtime_params(self, values: dict[str, Any]) -> None:
        if "silenceMs" in values:
            silence_ms = _coerce_number(values.get("silenceMs"))
            if silence_ms is not None:
                self._silence_s = float(max(0.0, float(silence_ms)) / 1000.0)
        if "deltaThreshold" in values:
            delta = _coerce_number(values.get("deltaThreshold"))
            if delta is not None:
                self._delta_threshold = float(max(0.0, float(delta)))
        if "fadeMs" in values:
            fade_ms = _coerce_number(values.get("fadeMs"))
            if fade_ms is not None:
                fade_s = float(max(0.0, float(fade_ms)) / 1000.0)
                self._tau_s = float(fade_s / 3.0) if fade_s > 0.0 else 0.0


MixSilenceFillRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Mix (Silence Fill)",
    description="Outputs A by default; when A is silent for a while, crossfades to B as filler.",
    tags=["mix", "blend", "silence", "filler", "crossfade"],
    execInPorts=["exec"],
    execOutPorts=["exec"],
    dataInPorts=[
        F8DataPortSpec(name="A", description="Primary signal", valueSchema=number_schema()),
        F8DataPortSpec(name="B", description="Filler signal", valueSchema=number_schema()),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="out", description="Mixed output", valueSchema=number_schema()),
        F8DataPortSpec(name="alpha", description="Crossfade factor (0=A, 1=B)", valueSchema=number_schema()),
        F8DataPortSpec(name="silent", description="1 if A is considered silent else 0", valueSchema=number_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="silenceMs",
            label="Silence (ms)",
            description="If A changes less than deltaThreshold for this long, fade to B.",
            valueSchema=integer_schema(default=500, minimum=0, maximum=60_000),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="deltaThreshold",
            label="Delta Threshold",
            description="Absolute change threshold to treat A as active.",
            valueSchema=number_schema(default=0.001, minimum=0.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="fadeMs",
            label="Fade (ms)",
            description="Crossfade time (0=instant).",
            valueSchema=integer_schema(default=200, minimum=0, maximum=60_000),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
    ],
    editableStateFields=False,
    editableDataInPorts=False,
    editableDataOutPorts=False,
    editableExecInPorts=False,
    editableExecOutPorts=False,
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return MixSilenceFillRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(MixSilenceFillRuntimeNode.SPEC, overwrite=True)
    return reg
