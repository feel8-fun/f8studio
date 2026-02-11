from __future__ import annotations

import logging
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
    number_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS: Final[str] = "f8.program_wave"

logger = logging.getLogger(__name__)


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


def _coerce_epoch_us(value: Any) -> int | None:
    """
    Coerce an epoch timestamp into microseconds.

    Input can be:
    - microseconds since epoch (>= 1e13-ish for modern timestamps)
    - milliseconds since epoch (smaller), auto-upscaled to microseconds
    """
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


@dataclass(frozen=True)
class _Program:
    start_us: int
    time_s: float | None
    hz: float
    loop_running_s: float | None
    loop_pause_s: float | None


def _parse_program(v: Any) -> _Program | None:
    if not isinstance(v, dict):
        return None

    start_us = _coerce_epoch_us(v.get("tsMs"))
    if start_us is None:
        return None

    time_sec = _coerce_number(v.get("timeSec"))
    time_s: float | None = None
    if time_sec is not None and time_sec > 0.0:
        time_s = float(time_sec)

    hz = _coerce_number(v.get("hz"))
    if hz is None:
        hz_f = 1.0
    else:
        hz_f = max(0.0, float(hz))

    loop_running_sec = _coerce_number(v.get("loopRunningSec"))
    loop_pause_sec = _coerce_number(v.get("loopPauseSec"))
    loop_running_s = float(loop_running_sec) if loop_running_sec is not None and loop_running_sec > 0.0 else None
    loop_pause_s = float(loop_pause_sec) if loop_pause_sec is not None and loop_pause_sec > 0.0 else None

    return _Program(
        start_us=int(start_us),
        time_s=time_s,
        hz=float(hz_f),
        loop_running_s=loop_running_s,
        loop_pause_s=loop_pause_s,
    )


def _elapsed_s(*, now_us: int, start_us: int) -> float:
    return float(now_us - start_us) / 1_000_000.0


def _active_at(t_s: float, *, loop_running_s: float | None, loop_pause_s: float | None) -> bool:
    if t_s < 0.0:
        return False
    if loop_running_s is None:
        return True
    run_s = float(loop_running_s)
    if run_s <= 0.0:
        return True
    pause_s = float(loop_pause_s or 0.0)
    if pause_s <= 0.0:
        return True
    cycle = run_s + pause_s
    if cycle <= 0.0:
        return True
    pos = t_s % cycle
    return pos < run_s


def _active_time_s(t_s: float, *, loop_running_s: float | None, loop_pause_s: float | None) -> float:
    """
    Total accumulated active time in seconds over [0..t_s], given run/pause loops.

    If no loop is configured, active time equals t_s.
    """
    if t_s <= 0.0:
        return 0.0
    if loop_running_s is None:
        return float(t_s)
    run_s = float(loop_running_s)
    if run_s <= 0.0:
        return float(t_s)
    pause_s = float(loop_pause_s or 0.0)
    if pause_s <= 0.0:
        return float(t_s)
    cycle = run_s + pause_s
    if cycle <= 0.0:
        return float(t_s)

    full = int(t_s // cycle)
    rem = float(t_s - float(full) * cycle)
    return float(full) * run_s + min(run_s, rem)


class ProgramWaveRuntimeNode(OperatorNode):
    """
    Program-controlled phase generator from a dict state payload.

    The `program` state is expected to be a dict like:
      tsMs: start timestamp (microseconds or milliseconds since epoch)
      timeSec: total duration (0 => indefinite)
      hz: phase speed (cycles per second)
      loopRunningSec / loopPauseSec: alternating active/pause windows

    Outputs:
      - phaseTurns: unwrapped phase turns (cycles), advances only during active windows
      - phase: normalized phase (0..1) derived from phaseTurns
      - active: whether we are currently in a running window
      - done: whether the program finished (timeSec elapsed)
      - elapsedSec: elapsed time since start (clamped to >= 0)
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._program: _Program | None = _parse_program((initial_state or {}).get("program"))
        self._last_error: str | None = None

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        _ = ts_ms
        if str(field) != "program":
            return
        self._program = _parse_program(value)

    def _log_error_once(self, msg: str, *, exc: BaseException | None = None) -> None:
        s = str(msg or "")
        if not s or s == self._last_error:
            return
        self._last_error = s
        if exc is None:
            logger.error("[%s:program_wave] %s", self.node_id, s)
        else:
            logger.exception("[%s:program_wave] %s", self.node_id, s, exc_info=exc)

    async def _read_program_state(self) -> _Program | None:
        try:
            return _parse_program(await self.get_state_value("program"))
        except Exception as exc:
            # Safety boundary: state transport errors should not crash the graph.
            self._log_error_once(f"failed to read state 'program': {type(exc).__name__}: {exc}", exc=exc)
            return None

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        _ = ctx_id
        p = str(port or "")
        if p not in ("phase", "phaseTurns", "active", "done", "elapsedSec"):
            return None

        program = self._program
        if program is None:
            program = await self._read_program_state()
            self._program = program
        if program is None:
            if p == "phase":
                return 0.0
            if p == "phaseTurns":
                return 0.0
            if p == "active":
                return False
            if p == "done":
                return True
            if p == "elapsedSec":
                return 0.0
            return None

        now_us = int(time.time() * 1_000_000.0)
        t_s = _elapsed_s(now_us=now_us, start_us=int(program.start_us))
        t_s_nonneg = max(0.0, float(t_s))

        done = False
        if program.time_s is not None and t_s_nonneg >= float(program.time_s):
            done = True

        if done:
            active = False
            t_for_phase = float(program.time_s or 0.0)
        else:
            active = _active_at(t_s_nonneg, loop_running_s=program.loop_running_s, loop_pause_s=program.loop_pause_s)
            t_for_phase = t_s_nonneg

        active_time = _active_time_s(t_for_phase, loop_running_s=program.loop_running_s, loop_pause_s=program.loop_pause_s)
        phase_turns = float(active_time * float(program.hz)) if float(program.hz) > 0.0 else 0.0
        phase = float(phase_turns % 1.0)

        if p == "phase":
            return float(phase)
        if p == "phaseTurns":
            return float(phase_turns)
        if p == "active":
            return bool(active)
        if p == "done":
            return bool(done)
        if p == "elapsedSec":
            return float(t_s_nonneg)
        return None


ProgramWaveRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Program Wave",
    description="Generate a program-controlled phase/gate waveform from a dict state payload.",
    tags=["signal", "program", "wave", "phase", "lovense"],
    dataInPorts=[],
    dataOutPorts=[
        F8DataPortSpec(name="phaseTurns", description="Unwrapped phase turns (cycles).", valueSchema=number_schema()),
        F8DataPortSpec(name="phase", description="Normalized phase (0..1).", valueSchema=number_schema()),
        F8DataPortSpec(name="active", description="Whether program is in a running window.", valueSchema=boolean_schema()),
        F8DataPortSpec(name="done", description="Whether program finished (timeSec elapsed).", valueSchema=boolean_schema()),
        F8DataPortSpec(name="elapsedSec", description="Elapsed seconds since start.", valueSchema=number_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="program",
            label="Program",
            description="Dict payload defining tsMs/timeSec/hz/loopRunningSec/loopPauseSec.",
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
        return ProgramWaveRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(ProgramWaveRuntimeNode.SPEC, overwrite=True)
    return reg
