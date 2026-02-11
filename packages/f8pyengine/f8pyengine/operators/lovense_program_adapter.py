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
    number_schema,
    string_schema,
)
from f8pysdk.json_unwrap import unwrap_json_value as _unwrap_json_value
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS

OPERATOR_CLASS: Final[str] = "f8.lovense_program_adapter"

logger = logging.getLogger(__name__)


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


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _split_action(action: Any) -> list[str]:
    s = str(action or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _parse_action_int(action: Any, *, prefix: str) -> int | None:
    for part in _split_action(action):
        if not part.startswith(prefix):
            continue
        try:
            return int(part.split(prefix, 1)[1])
        except (IndexError, ValueError):
            return None
    return None


def _parse_action_thrusting_depth(action: Any) -> tuple[int | None, int | None]:
    thrusting = _parse_action_int(action, prefix="Thrusting:")
    depth = _parse_action_int(action, prefix="Depth:")
    return thrusting, depth


@dataclass(frozen=True)
class _ProgramPayload:
    ts_ms: int
    time_sec: float | None
    hz: float
    loop_running_sec: float | None
    loop_pause_sec: float | None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tsMs": int(self.ts_ms), "hz": float(self.hz)}
        if self.time_sec is not None:
            out["timeSec"] = float(self.time_sec)
        else:
            out["timeSec"] = 0.0
        if self.loop_running_sec is not None:
            out["loopRunningSec"] = float(self.loop_running_sec)
        if self.loop_pause_sec is not None:
            out["loopPauseSec"] = float(self.loop_pause_sec)
        return out


class LovenseProgramAdapterRuntimeNode(OperatorNode):
    """
    Adapter that converts Lovense Mock Server `event` dicts into:

    - `program`: ProgramWave-compatible dict (tsMs/timeSec/hz/loopRunningSec/loopPauseSec)
    - `amplitude`: normalized 0..1 amplitude (e.g. depth or vibration strength)

    Intended usage:
      lovense_mock_server.event (state edge) -> this.lovenseEvent
      this.program (data) -> program_wave.program (data)
      this.amplitude (data) -> shaper/amplitude pipeline
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[p.name for p in (node.dataOutPorts or [])],
            state_fields=[s.name for s in (node.stateFields or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._last_event: dict[str, Any] | None = None
        self._dirty = True
        self._cache: dict[str, Any] = {
            "kind": "",
            "program": self._done_program_dict(),
            "amplitude": 0.0,
            "sequence": None,
        }
        self._last_published: dict[str, Any] = {}
        self._last_error: str | None = None

    @staticmethod
    def _done_program_dict() -> dict[str, Any]:
        now_ms = int(time.time() * 1000.0)
        # Ensure elapsedSec > timeSec so ProgramWave reports done=True and active=False.
        return {"tsMs": int(max(1, now_ms - 1000)), "timeSec": 0.001, "hz": 0.0}

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        _ = ts_ms
        name = str(field or "")
        if name == "lovenseEvent":
            event = _unwrap_json_value(value)
            if isinstance(event, dict):
                self._last_event = event
                self._dirty = True
                await self._recompute_and_publish()
            return
        # Mapping parameters changed; recompute against the last seen event.
        if name in (
            "minHz",
            "maxHz",
            "thrustingMax",
            "depthMax",
            "speedGamma",
            "vibrateHz",
            "vibrateMax",
        ):
            self._dirty = True
            await self._recompute_and_publish()

    def _log_error_once(self, msg: str, *, exc: BaseException | None = None) -> None:
        s = str(msg or "")
        if not s or s == self._last_error:
            return
        self._last_error = s
        if exc is None:
            logger.error("[%s:lovense_program_adapter] %s", self.node_id, s)
        else:
            logger.exception("[%s:lovense_program_adapter] %s", self.node_id, s, exc_info=exc)

    async def _recompute_and_publish(self) -> None:
        if not self._dirty:
            return
        await self._refresh_from_state()
        await self._compute_cache()
        self._dirty = False
        await self._publish_outputs()

    async def _refresh_from_state(self) -> None:
        if self._last_event is not None:
            return
        ev = await self.get_state_value("lovenseEvent")
        event = _unwrap_json_value(ev)
        if isinstance(event, dict):
            self._last_event = event

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

    async def _publish_outputs(self) -> None:
        program = self._cache.get("program")
        amplitude = self._cache.get("amplitude")
        kind = self._cache.get("kind")
        sequence = self._cache.get("sequence")

        next_values = {"program": program, "amplitude": amplitude, "kind": kind, "sequence": sequence}
        if next_values == self._last_published:
            return
        self._last_published = dict(next_values)

        try:
            await self.set_state("program", program)
            await self.set_state("amplitude", amplitude)
            await self.set_state("kind", kind)
            await self.set_state("sequence", sequence)
        except Exception as exc:
            # Safety boundary: state publishing should not crash the graph.
            self._log_error_once(f"failed to publish output states: {type(exc).__name__}: {exc}", exc=exc)

    @staticmethod
    def _parse_step_ms(rule: Any, *, default_ms: float) -> float:
        # Example: "V:1;F:v,r,p,t,f,s,d,o;S:200#"
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
        except (TypeError, ValueError):
            return float(default_ms)
        if ms <= 0:
            return float(default_ms)
        return float(ms)

    @staticmethod
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
            frag = part.strip()
            if not frag:
                continue
            n = _coerce_number(frag)
            if n is None:
                continue
            out.append(float(n))
        return out

    async def _compute_cache(self) -> None:
        event = self._last_event
        if event is None:
            self._cache = {"kind": "", "program": self._done_program_dict(), "amplitude": 0.0, "sequence": None}
            return

        ts_ms_v = _coerce_number(event.get("tsMs"))
        ts_ms = int(ts_ms_v) if ts_ms_v is not None and ts_ms_v > 0 else int(time.time() * 1000.0)

        # Prefer canonical summary if present, but do not require it.
        summary = event.get("summary")
        command = event.get("command")
        params = event.get("params")

        kind = ""
        if isinstance(summary, dict):
            kind = str(summary.get("type") or "")
        if not kind and isinstance(command, dict):
            kind = str(command.get("kind") or "")

        action = None
        if isinstance(summary, dict) and "action" in summary:
            action = summary.get("action")
        if action is None and isinstance(params, dict) and "action" in params:
            action = params.get("action")

        # Unified scheduling parameters.
        time_sec_raw = None
        loop_running_raw = None
        loop_pause_raw = None
        if isinstance(summary, dict):
            time_sec_raw = summary.get("timeSec")
            loop_running_raw = summary.get("loopRunningSec")
            loop_pause_raw = summary.get("loopPauseSec")
        if isinstance(params, dict):
            if time_sec_raw is None:
                time_sec_raw = params.get("timeSec")
            if loop_running_raw is None:
                loop_running_raw = params.get("loopRunningSec")
            if loop_pause_raw is None:
                loop_pause_raw = params.get("loopPauseSec")

        time_sec_raw_n = _coerce_number(time_sec_raw)
        time_sec: float | None = None
        if time_sec_raw_n is not None and time_sec_raw_n > 0.0:
            time_sec = float(time_sec_raw_n)

        loop_running_raw_n = _coerce_number(loop_running_raw)
        loop_pause_raw_n = _coerce_number(loop_pause_raw)
        loop_running = float(loop_running_raw_n) if loop_running_raw_n is not None and loop_running_raw_n > 0.0 else None
        loop_pause = float(loop_pause_raw_n) if loop_pause_raw_n is not None and loop_pause_raw_n > 0.0 else None

        if kind == "stop":
            self._cache = {"kind": kind, "program": self._done_program_dict(), "amplitude": 0.0, "sequence": None}
            return

        if kind == "solace_thrusting":
            thrusting = None
            depth = None
            if isinstance(summary, dict):
                thrusting = _coerce_number(summary.get("thrusting"))
                depth = _coerce_number(summary.get("depth"))
            if thrusting is None or depth is None:
                t_i, d_i = _parse_action_thrusting_depth(action)
                thrusting = _coerce_number(t_i)
                depth = _coerce_number(d_i)
            if thrusting is None or depth is None:
                self._cache = {"kind": kind, "program": self._done_program_dict(), "amplitude": 0.0}
                return

            # Map thrusting/depth -> (hz, amplitude) using the same parameters as LovenseThrustingWave.
            thrusting_max = float(await self._get_state_number("thrustingMax", default=20.0))
            depth_max = float(await self._get_state_number("depthMax", default=20.0))
            min_hz = float(await self._get_state_number("minHz", default=0.0))
            max_hz = float(await self._get_state_number("maxHz", default=3.0))
            speed_gamma = float(await self._get_state_number("speedGamma", default=1.0))

            thrust_norm = 0.0 if thrusting_max <= 0.0 else max(0.0, min(1.0, float(thrusting) / thrusting_max))
            depth_norm = 0.0 if depth_max <= 0.0 else max(0.0, min(1.0, float(depth) / depth_max))
            thrust_norm = math.pow(thrust_norm, max(0.01, float(speed_gamma)))

            hz = float(min_hz + thrust_norm * (max_hz - min_hz))
            amp = float(_clamp01(depth_norm))

            program = _ProgramPayload(
                ts_ms=ts_ms,
                time_sec=time_sec,
                hz=hz,
                loop_running_sec=loop_running,
                loop_pause_sec=loop_pause,
            ).to_dict()
            self._cache = {"kind": kind, "program": program, "amplitude": amp, "sequence": None}
            return

        if kind == "all_vibrate":
            strength = None
            if isinstance(summary, dict):
                strength = _coerce_number(summary.get("all"))
            if strength is None:
                strength = _coerce_number(_parse_action_int(action, prefix="All:"))
            if strength is None:
                self._cache = {"kind": kind, "program": self._done_program_dict(), "amplitude": 0.0}
                return

            vibrate_hz = float(await self._get_state_number("vibrateHz", default=2.0))
            vibrate_max = float(await self._get_state_number("vibrateMax", default=20.0))

            amp = 0.0 if vibrate_max <= 0.0 else float(_clamp01(float(strength) / vibrate_max))
            program = _ProgramPayload(
                ts_ms=ts_ms,
                time_sec=time_sec,
                hz=max(0.0, vibrate_hz),
                loop_running_sec=loop_running,
                loop_pause_sec=loop_pause,
            ).to_dict()
            self._cache = {"kind": kind, "program": program, "amplitude": amp, "sequence": None}
            return

        if kind == "vibration_pattern":
            if not isinstance(params, dict):
                self._cache = {"kind": kind, "program": self._done_program_dict(), "amplitude": 0.0, "sequence": None}
                return

            strengths = self._parse_strengths(params.get("strength"))
            if not strengths:
                self._cache = {"kind": kind, "program": self._done_program_dict(), "amplitude": 0.0, "sequence": None}
                return

            step_ms = self._parse_step_ms(params.get("rule"), default_ms=150.0)

            strength_max = float(await self._get_state_number("patternStrengthMax", default=20.0))
            hz_min = float(await self._get_state_number("patternMinHz", default=0.0))
            hz_max = float(await self._get_state_number("patternMaxHz", default=5.0))
            if strength_max <= 0.0:
                strength_max = 20.0

            values_hz: list[float] = []
            for s in strengths:
                norm = 0.0 if strength_max <= 0.0 else max(0.0, min(1.0, float(s) / strength_max))
                hz = float(hz_min + norm * (hz_max - hz_min))
                values_hz.append(float(max(0.0, hz)))

            seq_dict: dict[str, Any] = {
                "tsMs": int(ts_ms),
                "stepMs": float(step_ms),
                "values": list(values_hz),
                "timeSec": float(time_sec) if time_sec is not None else 0.0,
            }

            self._cache = {
                "kind": kind,
                "program": self._done_program_dict(),
                "amplitude": 0.0,
                "sequence": seq_dict,
            }
            return

        # Unknown kinds: keep outputs stable but clearly report kind for debugging.
        self._cache = {"kind": kind, "program": self._done_program_dict(), "amplitude": 0.0, "sequence": None}


LovenseProgramAdapterRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Lovense Program Adapter",
    description="Convert Lovense Mock Server events into ProgramWave input + amplitude.",
    tags=["lovense", "program", "adapter", "wave", "phase"],
    dataInPorts=[],
    dataOutPorts=[],
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
            name="program",
            label="Program",
            description="Readonly output: ProgramWave-compatible dict (tsMs/timeSec/hz/loopRunningSec/loopPauseSec).",
            valueSchema=any_schema(),
            access=F8StateAccess.ro,
            showOnNode=True,
            required=False,
        ),
        F8StateSpec(
            name="amplitude",
            label="Amplitude",
            description="Readonly output: normalized amplitude (0..1).",
            valueSchema=number_schema(default=0.0, minimum=0.0, maximum=1.0),
            access=F8StateAccess.ro,
            showOnNode=True,
            required=False,
        ),
        F8StateSpec(
            name="kind",
            label="Kind",
            description="Readonly output: event kind (debug).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            showOnNode=False,
            required=False,
        ),
        F8StateSpec(
            name="sequence",
            label="Sequence",
            description="Readonly output: sequence dict for Sequence Player (typically pattern -> hz sequence).",
            valueSchema=any_schema(),
            access=F8StateAccess.ro,
            showOnNode=False,
            required=False,
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
            valueSchema=number_schema(default=3.0, minimum=0.0, maximum=100.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="thrustingMax",
            label="Thrusting Max",
            description="Normalize thrusting value by this maximum.",
            valueSchema=number_schema(default=20.0, minimum=1.0, maximum=100.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="depthMax",
            label="Depth Max",
            description="Normalize depth value by this maximum (amplitude).",
            valueSchema=number_schema(default=3.0, minimum=1.0, maximum=100.0),
            access=F8StateAccess.wo,
            showOnNode=False,
        ),
        F8StateSpec(
            name="speedGamma",
            label="Speed Gamma",
            description="Nonlinear curve for thrusting->speed (>=0.01).",
            valueSchema=number_schema(default=1.0, minimum=0.01, maximum=10.0),
            access=F8StateAccess.wo,
            showOnNode=False,
        ),
        F8StateSpec(
            name="vibrateHz",
            label="Vibrate Hz",
            description="Default program frequency for All: vibration events.",
            valueSchema=number_schema(default=2.0, minimum=0.0, maximum=100.0),
            access=F8StateAccess.wo,
            showOnNode=False,
        ),
        F8StateSpec(
            name="vibrateMax",
            label="Vibrate Max",
            description="Normalize All: vibration strength by this maximum.",
            valueSchema=number_schema(default=20.0, minimum=1.0, maximum=100.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="patternMinHz",
            label="Pattern Min Hz",
            description="Pattern strength=0 mapped Hz minimum.",
            valueSchema=number_schema(default=0.0, minimum=0.0, maximum=100.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="patternMaxHz",
            label="Pattern Max Hz",
            description="Pattern strength=patternStrengthMax mapped Hz maximum.",
            valueSchema=number_schema(default=5.0, minimum=0.0, maximum=100.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="patternStrengthMax",
            label="Pattern Strength Max",
            description="Normalize Pattern strength values by this maximum before mapping to Hz.",
            valueSchema=number_schema(default=20.0, minimum=1.0, maximum=100.0),
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

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
        return LovenseProgramAdapterRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(LovenseProgramAdapterRuntimeNode.SPEC, overwrite=True)
    return reg
