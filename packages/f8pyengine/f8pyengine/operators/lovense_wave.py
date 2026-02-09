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
    integer_schema,
    number_schema,
)
from f8pysdk.json_unwrap import unwrap_json_value as _unwrap_json_value
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from ._ports import exec_out_ports

THRUSTING_OPERATOR_CLASS: Final[str] = "f8.lovense_thrusting_wave"
VIBRATION_OPERATOR_CLASS: Final[str] = "f8.lovense_vibration_wave"

_TWO_PI = 2.0 * math.pi


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
    Exponential smoothing toward a target with time-constant-like behavior.

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


class _PhaseAccumulator:
    """
    Monotonic-time phase accumulator (normalized 0..1).
    """

    def __init__(self, *, initial_phase: float = 0.0) -> None:
        self._phase = float(initial_phase) % 1.0
        self._last_time_s: float | None = None

    def reset_time_base(self) -> None:
        self._last_time_s = None

    def phase(self) -> float:
        return float(self._phase)

    def step(self, *, hz: float) -> float:
        now_s = time.monotonic()
        if self._last_time_s is None:
            dt = 0.0
        else:
            dt = max(0.0, now_s - self._last_time_s)
        self._last_time_s = now_s
        hz_f = max(0.0, float(hz))
        self._phase = (self._phase + hz_f * dt) % 1.0
        return float(self._phase)


@dataclass(frozen=True)
class _ThrustCmd:
    event_id: str
    thrusting: float
    depth: float
    time_sec: float | None
    loop_running_sec: float | None
    loop_pause_sec: float | None


def _parse_event_id(event: dict[str, Any]) -> str:
    for k in ("eventId", "seq", "tsMs", "ts"):
        if k in event:
            return str(event.get(k) or "")
    return ""


def _parse_thrust_cmd(event: dict[str, Any]) -> _ThrustCmd | None:
    summary = event.get("summary")
    if not isinstance(summary, dict):
        return None
    typ = str(summary.get("type") or "")
    if typ != "solace_thrusting":
        return None

    thrusting = _coerce_number(summary.get("thrusting"))
    depth = _coerce_number(summary.get("depth"))
    time_sec = _coerce_number(summary.get("timeSec"))
    loop_running_sec = _coerce_number(summary.get("loopRunningSec"))
    loop_pause_sec = _coerce_number(summary.get("loopPauseSec"))
    if thrusting is None or depth is None:
        return None

    # Lovense sometimes uses timeSec=0 to mean "continuous".
    time_s: float | None = None
    if time_sec is not None and time_sec > 0.0:
        time_s = float(time_sec)

    return _ThrustCmd(
        event_id=_parse_event_id(event),
        thrusting=float(thrusting),
        depth=float(depth),
        time_sec=time_s,
        loop_running_sec=float(loop_running_sec) if loop_running_sec and loop_running_sec > 0.0 else None,
        loop_pause_sec=float(loop_pause_sec) if loop_pause_sec and loop_pause_sec > 0.0 else None,
    )


@dataclass(frozen=True)
class _VibrationCmd:
    event_id: str
    strengths: list[float]
    step_ms: float
    time_sec: float | None


def _parse_strengths(value: Any) -> list[float]:
    v = _unwrap_json_value(value)
    if v is None:
        return []
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [float(v)]
    s = str(v or "").strip()
    if not s:
        return []
    out: list[float] = []
    for part in s.split(";"):
        part_s = part.strip()
        if not part_s:
            continue
        f = _coerce_number(part_s)
        if f is None:
            continue
        out.append(float(f))
    return out


def _parse_step_ms(rule: Any, *, default_ms: float) -> float:
    # Example: "V:1;F:v,r,p,t,f,s,d,o;S:133#"
    txt = str(_unwrap_json_value(rule) or "")
    idx = txt.find("S:")
    if idx == -1:
        return float(default_ms)
    frag = txt[idx + 2 :]
    digits = ""
    for ch in frag:
        if ch.isdigit():
            digits += ch
        else:
            break
    try:
        ms = float(int(digits))
    except Exception:
        return float(default_ms)
    if ms <= 0:
        return float(default_ms)
    return float(ms)


def _parse_vibration_cmd(event: dict[str, Any]) -> _VibrationCmd | None:
    summary = event.get("summary")
    if not isinstance(summary, dict):
        return None
    typ = str(summary.get("type") or "")
    if typ != "vibration_pattern":
        return None

    strengths = _parse_strengths(summary.get("strength"))
    if not strengths:
        return None
    step_ms = _parse_step_ms(summary.get("rule"), default_ms=150.0)

    time_sec = _coerce_number(summary.get("timeSec"))
    time_s: float | None = None
    if time_sec is not None and time_sec > 0.0:
        time_s = float(time_sec)

    return _VibrationCmd(event_id=_parse_event_id(event), strengths=strengths, step_ms=float(step_ms), time_sec=time_s)


class LovenseThrustingWaveRuntimeNode(OperatorNode):
    """
    Tick-driven thrusting waveform generator (0..1) driven by Lovense "Function Thrusting" events.

    Wire a state edge:
      lovense_mock.event -> this.lovenseEvent

    Then wire a tick exec chain to call this node each tick.

    Notes:
    - Phase is continuous across parameter changes (frequency/amplitude slew).
    - Output is always clamped to 0..1.
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

        self._phase = _PhaseAccumulator(initial_phase=float(self._initial_state.get("phase", 0.0) or 0.0))
        self._last_step_time_s: float | None = None
        self._last_ctx_id: str | int | None = None
        self._last_outputs: dict[str, float] = {}

        # Smoothed parameters.
        self._freq = _Slew(value=0.0, target=0.0)
        self._amp = _Slew(value=0.0, target=0.0)

        self._last_event_id = ""
        self._last_cmd_sig: tuple[float, float, float | None, float | None, float | None] | None = None
        self._active_until_s: float | None = None
        self._loop_t0_s: float | None = None
        self._loop_run_s: float | None = None
        self._loop_pause_s: float | None = None

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        _ = active
        _ = meta
        self._phase.reset_time_base()

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        _ = ts_ms
        if str(field) != "lovenseEvent":
            return
        event = _unwrap_json_value(value)
        if not isinstance(event, dict):
            return
        await self._apply_event(event)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        p = str(port)
        if p not in ("out", "phase", "frequencyHz", "amplitude", "gate"):
            return None

        if ctx_id is not None and ctx_id == self._last_ctx_id and p in self._last_outputs:
            return self._last_outputs.get(p)

        await self._refresh_from_state()
        out = await self._step_once()
        self._last_ctx_id = ctx_id
        self._last_outputs = dict(out)
        return self._last_outputs.get(p)

    async def _refresh_from_state(self) -> None:
        ev = await self.get_state_value("lovenseEvent")
        event = _unwrap_json_value(ev)
        if isinstance(event, dict):
            await self._apply_event(event)

    async def _apply_event(self, event: dict[str, Any]) -> None:
        event_id = _parse_event_id(event)

        summary = event.get("summary")
        if isinstance(summary, dict) and str(summary.get("type") or "") == "stop":
            # Stop always applies (even if upstream metadata is flaky).
            self._last_event_id = event_id
            self._last_cmd_sig = None
            self._active_until_s = None
            self._loop_t0_s = None
            self._loop_run_s = None
            self._loop_pause_s = None
            self._freq.target = 0.0
            self._amp.target = 0.0
            return

        cmd = _parse_thrust_cmd(event)
        if cmd is None:
            return

        sig = (float(cmd.thrusting), float(cmd.depth), cmd.time_sec, cmd.loop_running_sec, cmd.loop_pause_sec)
        incoming_id = str(cmd.event_id or event_id or "")
        if incoming_id and incoming_id == self._last_event_id and self._last_cmd_sig == sig:
            return
        # If `incoming_id` is empty or unstable (e.g. two packets share the same `ts` string),
        # still accept the update when the command signature changes.
        if not incoming_id and self._last_cmd_sig == sig:
            return

        self._last_event_id = cmd.event_id or event_id
        self._last_cmd_sig = sig
        now_s = time.monotonic()

        # Map thrusting/depth -> targets.
        thrusting_max = float(await self._get_state_number("thrustingMax", default=20.0))
        depth_max = float(await self._get_state_number("depthMax", default=20.0))
        min_hz = float(await self._get_state_number("minHz", default=0.0))
        max_hz = float(await self._get_state_number("maxHz", default=3.0))
        speed_gamma = float(await self._get_state_number("speedGamma", default=1.0))

        thrust_norm = 0.0 if thrusting_max <= 0 else max(0.0, min(1.0, float(cmd.thrusting) / thrusting_max))
        depth_norm = 0.0 if depth_max <= 0 else max(0.0, min(1.0, float(cmd.depth) / depth_max))
        thrust_norm = math.pow(thrust_norm, max(0.01, speed_gamma))

        self._freq.target = float(min_hz + thrust_norm * (max_hz - min_hz))
        self._amp.target = float(depth_norm)

        if cmd.time_sec is not None:
            self._active_until_s = now_s + float(cmd.time_sec)
        else:
            self._active_until_s = None

        if cmd.loop_running_sec is not None and cmd.loop_pause_sec is not None:
            self._loop_t0_s = now_s
            self._loop_run_s = float(cmd.loop_running_sec)
            self._loop_pause_s = float(cmd.loop_pause_sec)
        else:
            self._loop_t0_s = None
            self._loop_run_s = None
            self._loop_pause_s = None

    async def _get_state_number(self, field: str, *, default: float) -> float:
        live = _unwrap_json_value(await self.get_state_value(str(field)))
        n_live = _coerce_number(live)
        if n_live is not None:
            return float(n_live)
        init = _unwrap_json_value(self._initial_state.get(str(field)))
        n_init = _coerce_number(init)
        if n_init is not None:
            return float(n_init)
        return float(default)

    async def _step_once(self) -> dict[str, float]:
        now_s = time.monotonic()
        if self._last_step_time_s is None:
            dt_s = 0.0
        else:
            dt_s = max(0.0, now_s - float(self._last_step_time_s))
        self._last_step_time_s = now_s

        slew_ms = await self.get_state_value("slewMs")
        tau_ms = _coerce_number(_unwrap_json_value(slew_ms))
        if tau_ms is None:
            tau_ms = _coerce_number(self._initial_state.get("slewMs"))
        tau_s = float(max(0.0, float(tau_ms or 0.0)) / 1000.0)

        # Detect expiry.
        if self._active_until_s is not None and now_s >= float(self._active_until_s):
            self._active_until_s = None
            self._freq.target = 0.0
            self._amp.target = 0.0

        # Loop gate (optional).
        gate = 1.0
        if self._loop_t0_s is not None and self._loop_run_s is not None and self._loop_pause_s is not None:
            run_s = max(0.0, float(self._loop_run_s))
            pause_s = max(0.0, float(self._loop_pause_s))
            total = run_s + pause_s
            if total > 0.0:
                t = (now_s - float(self._loop_t0_s)) % total
                if t >= run_s:
                    gate = 0.0

        # Smooth parameters.
        hz = self._freq.step(dt_s=dt_s, tau_s=tau_s)
        amp = self._amp.step(dt_s=dt_s, tau_s=tau_s)

        # Step phase using current frequency (phase is continuous by construction).
        phase = self._phase.step(hz=hz)

        # Shape controls.
        eccentricity = _coerce_number(_unwrap_json_value(await self.get_state_value("eccentricity")))
        if eccentricity is None:
            eccentricity = _coerce_number(self._initial_state.get("eccentricity"))
        ecc = float(max(-1.0, min(1.0, float(eccentricity or 0.0))))

        power = _coerce_number(_unwrap_json_value(await self.get_state_value("power")))
        if power is None:
            power = _coerce_number(self._initial_state.get("power"))
        pw = float(max(0.1, min(10.0, float(power or 1.0))))

        theta = _TWO_PI * float(phase)
        theta_warp = theta + (ecc * math.sin(theta))
        base = 0.5 - 0.5 * math.cos(theta_warp)  # 0..1

        # Apply "power" easing around center.
        centered = (float(base) - 0.5) * 2.0  # -1..1
        shaped = math.copysign(math.pow(abs(centered), pw), centered)
        pos = 0.5 + 0.5 * shaped  # 0..1

        # Depth amplitude around center (amp is 0..1).
        out = 0.5 + (float(amp) * 0.5) * ((float(pos) - 0.5) * 2.0)
        out = _clamp01(out)

        # Apply gate (pause): default hold-at-center behavior.
        if gate <= 0.0:
            hold = _coerce_number(_unwrap_json_value(await self.get_state_value("pauseHold")))
            if hold is None:
                hold = _coerce_number(self._initial_state.get("pauseHold"))
            if hold is None:
                out = 0.5
            else:
                out = _clamp01(float(hold))

        return {
            "out": float(out),
            "phase": float(phase),
            "frequencyHz": float(hz),
            "amplitude": float(amp),
            "gate": float(gate),
        }


LovenseThrustingWaveRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=THRUSTING_OPERATOR_CLASS,
    version="0.0.1",
    label="Lovense Thrusting Wave",
    description="Tick-driven 0..1 thrusting waveform driven by Lovense Function(Thrusting) events.",
    tags=["lovense", "signal", "waveform", "thrusting"],
    execInPorts=["exec"],
    execOutPorts=["exec"],
    dataOutPorts=[
        F8DataPortSpec(name="out", description="Thrusting position (0..1).", valueSchema=number_schema()),
        F8DataPortSpec(name="phase", description="Debug: internal phase (0..1).", valueSchema=number_schema()),
        F8DataPortSpec(name="frequencyHz", description="Debug: current frequency (Hz).", valueSchema=number_schema()),
        F8DataPortSpec(name="amplitude", description="Debug: current depth amplitude (0..1).", valueSchema=number_schema()),
        F8DataPortSpec(name="gate", description="Debug: 1=running, 0=paused.", valueSchema=number_schema()),
    ],
    stateFields=[
        F8StateSpec(
            name="lovenseEvent",
            label="Lovense Event",
            description="State-edge input from Lovense Mock Server: the latest event dict.",
            valueSchema=any_schema(),
            access=F8StateAccess.wo,
            showOnNode=True,
        ),
        F8StateSpec(
            name="minHz",
            label="Min Hz",
            description="Frequency at thrusting=0.",
            valueSchema=number_schema(default=0.0, minimum=0.0, maximum=100.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="maxHz",
            label="Max Hz",
            description="Frequency at thrusting=thrustingMax.",
            valueSchema=number_schema(default=5.0, minimum=0.0, maximum=100.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="thrustingMax",
            label="Thrusting Max",
            description="Normalize thrusting value by this maximum.",
            valueSchema=number_schema(default=8.0, minimum=1.0, maximum=100.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="depthMax",
            label="Depth Max",
            description="Normalize depth value by this maximum.",
            valueSchema=number_schema(default=8.0, minimum=1.0, maximum=100.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="speedGamma",
            label="Speed Gamma",
            description="Nonlinear curve for thrusting->speed (>=0.01).",
            valueSchema=number_schema(default=1.0, minimum=0.01, maximum=10.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="slewMs",
            label="Slew (ms)",
            description="Smoothing time constant for parameter changes (0 disables).",
            valueSchema=integer_schema(default=120, minimum=0, maximum=60_000),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="eccentricity",
            label="Eccentricity",
            description="Phase warp amount (-1..1).",
            valueSchema=number_schema(default=0.0, minimum=-1.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="power",
            label="Power",
            description="Easing exponent around center (0.1..10).",
            valueSchema=number_schema(default=1.0, minimum=0.1, maximum=10.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="pauseHold",
            label="Pause Hold",
            description="When loop is paused, hold this value (0..1). Empty => hold at 0.5.",
            valueSchema=number_schema(default=0.5, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
    ],
    editableStateFields=False,
    editableDataInPorts=False,
    editableDataOutPorts=False,
    editableExecInPorts=False,
    editableExecOutPorts=False,
)


class LovenseVibrationWaveRuntimeNode(OperatorNode):
    """
    Tick-driven vibration waveform generator (0..1) driven by Lovense Pattern events.

    This is a basic implementation: it uses the Pattern rule's S:<ms># step time
    and cycles over the strength list. (More Lovense rule features can be added later.)
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

        self._last_event_id = ""
        self._pattern_t0_s: float | None = None
        self._pattern_end_s: float | None = None
        self._strengths: list[float] = []
        self._step_ms: float = 150.0

        self._last_ctx_id: str | int | None = None
        self._last_out: float | None = None

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        _ = ts_ms
        if str(field) != "lovenseEvent":
            return
        event = _unwrap_json_value(value)
        if not isinstance(event, dict):
            return
        await self._apply_event(event)

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        p = str(port)
        if p != "out":
            return None
        if ctx_id is not None and ctx_id == self._last_ctx_id:
            return self._last_out
        await self._refresh_from_state()
        v = await self._step()
        self._last_ctx_id = ctx_id
        self._last_out = v
        return v

    async def _refresh_from_state(self) -> None:
        ev = await self.get_state_value("lovenseEvent")
        event = _unwrap_json_value(ev)
        if isinstance(event, dict):
            await self._apply_event(event)

    async def _apply_event(self, event: dict[str, Any]) -> None:
        event_id = _parse_event_id(event)
        if event_id and event_id == self._last_event_id:
            return

        summary = event.get("summary")
        if isinstance(summary, dict) and str(summary.get("type") or "") == "stop":
            self._last_event_id = event_id
            self._pattern_t0_s = None
            self._pattern_end_s = None
            self._strengths = []
            return

        cmd = _parse_vibration_cmd(event)
        if cmd is None:
            return
        self._last_event_id = cmd.event_id or event_id
        now_s = time.monotonic()
        self._pattern_t0_s = now_s
        self._step_ms = float(max(1.0, cmd.step_ms))
        self._strengths = list(cmd.strengths)
        if cmd.time_sec is not None:
            self._pattern_end_s = now_s + float(cmd.time_sec)
        else:
            self._pattern_end_s = None

    async def _step(self) -> float:
        if not self._strengths or self._pattern_t0_s is None:
            return 0.0
        now_s = time.monotonic()
        if self._pattern_end_s is not None and now_s >= float(self._pattern_end_s):
            self._pattern_t0_s = None
            self._pattern_end_s = None
            self._strengths = []
            return 0.0

        dt_ms = (now_s - float(self._pattern_t0_s)) * 1000.0
        idx = int(max(0.0, dt_ms) / float(self._step_ms))
        if not self._strengths:
            return 0.0
        raw = float(self._strengths[idx % len(self._strengths)])
        strength_max = _coerce_number(_unwrap_json_value(await self.get_state_value("strengthMax")))
        if strength_max is None:
            strength_max = _coerce_number(_unwrap_json_value(self._initial_state.get("strengthMax")))
        if strength_max is None or strength_max <= 0.0:
            strength_max = max(1.0, max(self._strengths))
        return _clamp01(raw / float(strength_max))


LovenseVibrationWaveRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=VIBRATION_OPERATOR_CLASS,
    version="0.0.1",
    label="Lovense Vibration Wave",
    description="Tick-driven 0..1 vibration waveform driven by Lovense Pattern events (basic step sequencer).",
    tags=["lovense", "signal", "waveform", "vibration"],
    execInPorts=["exec"],
    execOutPorts=["exec"],
    dataOutPorts=[F8DataPortSpec(name="out", description="Vibration strength (0..1).", valueSchema=number_schema())],
    stateFields=[
        F8StateSpec(
            name="lovenseEvent",
            label="Lovense Event",
            description="State-edge input from Lovense Mock Server: the latest event dict.",
            valueSchema=any_schema(),
            access=F8StateAccess.wo,
            showOnNode=True,
        ),
        F8StateSpec(
            name="strengthMax",
            label="Strength Max",
            description="Normalize Pattern strength values by this maximum (<=0 uses max(strengths)).",
            valueSchema=number_schema(default=0.0, minimum=0.0, maximum=100.0),
            access=F8StateAccess.rw,
            showOnNode=False,
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

    def _thrust_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return LovenseThrustingWaveRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    def _vibe_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return LovenseVibrationWaveRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, THRUSTING_OPERATOR_CLASS, _thrust_factory, overwrite=True)
    reg.register(SERVICE_CLASS, VIBRATION_OPERATOR_CLASS, _vibe_factory, overwrite=True)

    reg.register_operator_spec(LovenseThrustingWaveRuntimeNode.SPEC, overwrite=True)
    reg.register_operator_spec(LovenseVibrationWaveRuntimeNode.SPEC, overwrite=True)
    return reg
