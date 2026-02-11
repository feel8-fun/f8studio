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
    any_schema,
    boolean_schema,
    integer_schema,
    number_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS: Final[str] = "f8.sequence_player"


def _coerce_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _coerce_epoch_us(value: Any) -> int | None:
    n = _coerce_number(value)
    if n is None:
        return None
    i = int(n)
    if i <= 0:
        return None
    # 2001-09-09 in ms is ~1e12; in us is ~1e15. Use 1e13 as a safe divider.
    if i >= 10_000_000_000_000:
        return int(i)
    return int(i) * 1000


def _parse_values(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        out: list[float] = []
        for v in value:
            n = _coerce_number(v)
            if n is None:
                continue
            out.append(float(n))
        return out
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    s = str(value or "").strip()
    if not s:
        return []
    out2: list[float] = []
    for part in s.split(";"):
        frag = part.strip()
        if not frag:
            continue
        n = _coerce_number(frag)
        if n is None:
            continue
        out2.append(float(n))
    return out2


@dataclass(frozen=True)
class _Sequence:
    start_us: int
    step_s: float
    values: tuple[float, ...]
    time_s: float | None


def _parse_sequence(v: Any) -> _Sequence | None:
    if not isinstance(v, dict):
        return None

    start_us = _coerce_epoch_us(v.get("tsMs"))
    if start_us is None:
        return None

    step_ms = _coerce_number(v.get("stepMs"))
    if step_ms is None:
        step_ms = 100.0
    step_s = max(0.001, float(step_ms) / 1000.0)

    values = tuple(_parse_values(v.get("values")))
    if not values:
        return None

    time_sec = _coerce_number(v.get("timeSec"))
    time_s: float | None = None
    if time_sec is not None and time_sec > 0.0:
        time_s = float(time_sec)

    return _Sequence(
        start_us=int(start_us),
        step_s=float(step_s),
        values=values,
        time_s=time_s,
    )


def _elapsed_s(*, now_us: int, start_us: int) -> float:
    return float(now_us - start_us) / 1_000_000.0


class SequencePlayerRuntimeNode(OperatorNode):
    """
    Step-sequence player driven by an epoch-based `sequence` state dict.

    State input (`sequence`):
      tsMs: start time (ms or us since epoch)
      stepMs: step duration in ms
      values: list[float] or semicolon string
      timeSec: optional total duration (<=0/omitted => loop indefinitely; >0 => loop until timeSec then stop)

    Outputs (data ports):
      value: current step value
      index: current step index (0-based)
      active: whether still playing
      done: whether playback ended
      elapsedSec: elapsed seconds since tsMs (clamped to >=0)
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._sequence: _Sequence | None = _parse_sequence((initial_state or {}).get("sequence"))

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        _ = ts_ms
        if str(field) != "sequence":
            return
        self._sequence = _parse_sequence(value)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        _ = ctx_id
        p = str(port or "")
        if p not in ("value", "index", "active", "done", "elapsedSec"):
            return None

        seq = self._sequence
        if seq is None:
            try:
                seq = _parse_sequence(await self.get_state_value("sequence"))
            except Exception:
                seq = None
            self._sequence = seq

        if seq is None:
            if p == "value":
                return 0.0
            if p == "index":
                return 0
            if p == "active":
                return False
            if p == "done":
                return True
            if p == "elapsedSec":
                return 0.0
            return None

        now_us = int(time.time() * 1_000_000.0)
        t_s = _elapsed_s(now_us=now_us, start_us=int(seq.start_us))
        t_s_nonneg = max(0.0, float(t_s))

        values = seq.values
        step_s = float(seq.step_s)
        n = len(values)

        done = False
        if seq.time_s is not None and t_s_nonneg >= float(seq.time_s):
            done = True

        active = not done

        if done:
            idx = max(0, n - 1)
            value = 0.0
        else:
            idx_raw = int(max(0.0, t_s_nonneg) / step_s)
            idx = idx_raw % n
            value = float(values[idx])

        if p == "value":
            return float(value)
        if p == "index":
            return int(idx)
        if p == "active":
            return bool(active)
        if p == "done":
            return bool(done)
        if p == "elapsedSec":
            return float(t_s_nonneg)
        return None


SequencePlayerRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Sequence Player",
    description="Play a step-sequence over time (epoch-based), outputting the current step value.",
    tags=["signal", "sequence", "pattern", "step", "player", "lovense"],
    dataInPorts=[],
    dataOutPorts=[
        F8DataPortSpec(name="value", description="Current step value.", valueSchema=number_schema()),
        F8DataPortSpec(name="index", description="Current 0-based step index.", valueSchema=integer_schema()),
        F8DataPortSpec(name="active", description="Whether still playing.", valueSchema=boolean_schema()),
        F8DataPortSpec(name="done", description="Whether playback ended.", valueSchema=boolean_schema()),
        F8DataPortSpec(name="elapsedSec", description="Elapsed seconds since start.", valueSchema=number_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="sequence",
            label="Sequence",
            description="Dict payload defining tsMs/stepMs/values/timeSec.",
            valueSchema=any_schema(),
            access=F8StateAccess.wo,
            showOnNode=True,
            required=True,
        ),
    ],
)


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return SequencePlayerRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(SequencePlayerRuntimeNode.SPEC, overwrite=True)
    return reg
