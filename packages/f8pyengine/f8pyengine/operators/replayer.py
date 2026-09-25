from __future__ import annotations

from f8pysdk.specs import exec_port_specs

from f8pysdk.codec import coerce_int, coerce_flag
import asyncio
import logging
import time
from typing import Any, Final

import msgspec

from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8SpecEditPolicy,
    F8StateAccess,
    F8StateSpec,
    boolean_schema,
    editable_collection_edit_policy,
    integer_schema,
    string_schema,
)
from f8pysdk._specs.builtin_fields import OPERATOR_ID_FIELD_NAME, SVC_ID_FIELD_NAME
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from ..recording import (
    RecordingReader,
    TIME_MODE_OFFSET_FROM_PLAY,
    TIME_MODE_RECORDED_EPOCH,
    TimelineCursor,
    build_timeline_state,
)

OPERATOR_CLASS: Final[str] = "f8.replayer"

logger = logging.getLogger(__name__)

_REPLAYER_LOAD_ERRORS = (OSError, RuntimeError, TypeError, ValueError, msgspec.MsgspecError)
_REPLAYER_PLAYBACK_ERRORS = (OSError, RuntimeError, TypeError, ValueError)
_REPLAYER_BUS_PUBLISH_ERRORS = (OSError, RuntimeError, TypeError, ValueError)

_CONTROL_STATE_NAMES = {
    "path",
    "loop",
    "timeMode",
    "playing",
    "durationMs",
    "loaded",
    SVC_ID_FIELD_NAME,
    OPERATOR_ID_FIELD_NAME,
}
_POSITION_PORT = "positionMs"


class ReplayerRuntimeNode(OperatorNode):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[str(p.name) for p in (node.dataInPorts or [])],
            data_out_ports=[str(p.name) for p in (node.dataOutPorts or [])],
            state_fields=[str(s.name) for s in (node.stateFields or [])],
            exec_in_ports=[str(p) for p in (node.execInPorts or [])],
            exec_out_ports=[str(p) for p in (node.execOutPorts or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._path = str(self._initial_state.get("path") or "").strip()
        self._loop_enabled = coerce_flag(self._initial_state.get("loop"), default=False)
        self._time_mode = _coerce_time_mode(self._initial_state.get("timeMode"), default=TIME_MODE_OFFSET_FROM_PLAY)
        self._playing = coerce_flag(self._initial_state.get("playing"), default=False)
        self._duration_ms = coerce_int(self._initial_state.get("durationMs"), default=0)
        self._loaded = False
        self._events: list[Any] = []
        self._first_event_ts_ms = 0
        self._task: asyncio.Task[object] | None = None
        self._stop = asyncio.Event()
        self._emit_seq: int = 0
        self._self_state_writes: dict[str, Any] = {}
        self._user_state_names = tuple(
            name for name in self.state_fields if str(name).strip() and str(name) not in _CONTROL_STATE_NAMES
        )

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._startup_load(), name=f"replayer:startup:{self.node_id}")

    async def close(self) -> None:
        await self._stop_loop()

    async def on_exec(self, exec_id: str | int, in_port: str | None = None) -> list[str]:
        name = str(in_port or "").strip()
        if name == "play":
            await self._set_playing(True, emit_started_event=False)
            return ["started"] if self._playing else []
        if name == "pause":
            await self._set_playing(False, emit_started_event=False)
            return ["stopped"]
        if name == "stop":
            await self._stop_playback(reset_position=True)
            return ["stopped"]
        return []

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name in self._self_state_writes and self._self_state_writes.get(name) == value:
            self._self_state_writes.pop(name, None)
            return
        if name == "path":
            self._path = str(value or "").strip()
            await self._load_recording()
            return
        if name == "loop":
            self._loop_enabled = coerce_flag(value, default=self._loop_enabled)
            return
        if name == "timeMode":
            self._time_mode = _coerce_time_mode(value, default=self._time_mode)
            return
        if name == "playing":
            next_playing = coerce_flag(value, default=self._playing)
            await self._set_playing(next_playing, emit_started_event=next_playing)
            return

    async def validate_state(self, field: str, value: Any, *, ts_ms: int, meta: dict[str, Any]) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "path":
            return str(value or "").strip()
        if name == "loop":
            return coerce_flag(value, default=False)
        if name == "timeMode":
            return _coerce_time_mode(value, default=TIME_MODE_OFFSET_FROM_PLAY)
        if name == "playing":
            return coerce_flag(value, default=False)
        return value

    async def _set_playing(self, playing: bool, *, emit_started_event: bool) -> None:
        next_playing = bool(playing)
        if next_playing == self._playing and ((self._task is not None) == next_playing):
            return
        self._playing = next_playing
        await self._safe_set_state("playing", bool(self._playing))
        if not self._playing:
            await self._stop_loop()
            return
        await self._load_recording()
        if not self._loaded:
            self._playing = False
            await self._safe_set_state("playing", False)
            return
        loop_started = self._start_loop()
        if loop_started and emit_started_event:
            await self._safe_emit_exec("started", exec_id=self._next_emit_exec_id())

    async def _startup_load(self) -> None:
        if self._path:
            await self._load_recording()
        if self._playing:
            await self._set_playing(True, emit_started_event=True)

    def _start_loop(self) -> bool:
        task = self._task
        if task is not None and not task.done():
            return False
        self._stop.clear()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.exception("[%s:replayer] no running event loop", self.node_id)
            return False
        self._task = loop.create_task(self._run_playback(), name=f"replayer:{self.node_id}")
        return True

    async def _stop_loop(self) -> None:
        self._stop.set()
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _stop_playback(self, *, reset_position: bool) -> None:
        self._playing = False
        await self._safe_set_state("playing", False)
        await self._stop_loop()
        if reset_position:
            await self._safe_emit(_POSITION_PORT, 0)

    async def _load_recording(self) -> None:
        path = str(self._path or "").strip()
        if not path:
            self._loaded = False
            self._events = []
            self._first_event_ts_ms = 0
            self._duration_ms = 0
            await self._safe_set_state("loaded", False)
            await self._safe_set_state("durationMs", 0)
            return
        try:
            reader = RecordingReader(path)
            info = reader.read_info()
            header = info.header
            filtered_events: list[Any] = []
            first_event_ts_ms = int(header.created_ts_ms)
            seen_first = False
            for event in reader.iter_events():
                if event.type == "header":
                    continue
                if not seen_first:
                    first_event_ts_ms = _event_ts_ms(event)
                    seen_first = True
                filtered_events.append(event)
            self._events = filtered_events
            self._first_event_ts_ms = int(first_event_ts_ms)
            self._duration_ms = int(info.duration_ms)
            self._loaded = True
            await self._safe_set_state("loaded", True)
            await self._safe_set_state("durationMs", int(self._duration_ms))
            await self.clear_error()
        except _REPLAYER_LOAD_ERRORS as exc:
            self._loaded = False
            self._events = []
            self._first_event_ts_ms = 0
            self._duration_ms = 0
            logger.exception("[%s:replayer] failed to load recording", self.node_id, exc_info=exc)
            await self._safe_set_state("loaded", False)
            await self._safe_set_state("durationMs", 0)
            await self.report_error(
                "REPLAYER_LOAD_ERROR",
                str(exc),
                severity="error",
                fingerprint=f"replayer-load:{exc}",
            )

    async def _run_playback(self) -> None:
        try:
            while self._playing and not self._stop.is_set():
                await self._play_once()
                if not self._playing or self._stop.is_set():
                    break
                if not self._loop_enabled:
                    self._playing = False
                    await self._safe_set_state("playing", False)
                    await self._safe_emit_exec("done", exec_id=self._next_emit_exec_id())
                    break
                await self._safe_emit_exec("looped", exec_id=self._next_emit_exec_id())
        except asyncio.CancelledError:
            raise
        except _REPLAYER_PLAYBACK_ERRORS as exc:
            logger.exception("[%s:replayer] playback failed", self.node_id, exc_info=exc)
            self._playing = False
            await self._safe_set_state("playing", False)
            await self.report_error(
                "REPLAYER_PLAYBACK_ERROR",
                str(exc),
                severity="error",
                fingerprint=f"replayer-playback:{exc}",
            )

    async def _play_once(self) -> None:
        if not self._events:
            await self._safe_emit(_POSITION_PORT, 0)
            return
        now_monotonic_s = time.monotonic()
        now_wall_ts_ms = int(time.time() * 1000.0)
        timeline_state = build_timeline_state(
            mode=self._time_mode,
            start_monotonic_s=now_monotonic_s,
            start_wall_ts_ms=now_wall_ts_ms,
            first_event_ts_ms=self._first_event_ts_ms,
        )
        cursor = TimelineCursor(state=timeline_state)
        for event in self._events:
            if not self._playing or self._stop.is_set():
                return
            due_s = cursor.event_due_monotonic_s(event)
            await self._sleep_until(due_s, cursor)
            await self._dispatch_event(event)
        await self._safe_emit(_POSITION_PORT, int(self._duration_ms))

    async def _sleep_until(self, due_monotonic_s: float, cursor: TimelineCursor) -> None:
        while self._playing and not self._stop.is_set():
            now_monotonic_s = time.monotonic()
            position_ms = min(int(self._duration_ms), cursor.current_position_ms(now_monotonic_s=now_monotonic_s))
            await self._safe_emit(_POSITION_PORT, position_ms)
            remaining_s = float(due_monotonic_s - now_monotonic_s)
            if remaining_s <= 0.0:
                return
            sleep_s = min(0.01, remaining_s)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=sleep_s)
            except asyncio.TimeoutError:
                continue

    async def _dispatch_event(self, event: Any) -> None:
        if event.type == "data_sample":
            exec_id = self._next_emit_exec_id()
            allowed_ports = {
                str(name) for name in self.data_out_ports if str(name).strip() and str(name) != _POSITION_PORT
            }
            for port, value in dict(event.data).items():
                if port in allowed_ports:
                    await self._safe_emit(str(port), value, ctx_id=exec_id)
            await self._safe_emit_exec("sample", exec_id=exec_id)
            return
        if event.type == "state_change":
            if str(event.field) in self._user_state_names:
                await self._safe_set_state(str(event.field), event.value)

    def _next_emit_exec_id(self) -> int:
        self._emit_seq += 1
        return int(self._emit_seq)

    async def _safe_emit(self, port: str, value: Any, *, ctx_id: str | int | None = None) -> None:
        try:
            await self.emit(str(port), value, ctx_id=ctx_id)
        except _REPLAYER_BUS_PUBLISH_ERRORS as exc:
            logger.exception("[%s:replayer] failed to emit port: %s", self.node_id, port, exc_info=exc)

    async def _safe_set_state(self, field: str, value: Any) -> None:
        try:
            self._self_state_writes[str(field)] = value
            await self.set_state(str(field), value)
        except _REPLAYER_BUS_PUBLISH_ERRORS as exc:
            self._self_state_writes.pop(str(field), None)
            logger.exception("[%s:replayer] failed to publish state: %s", self.node_id, field, exc_info=exc)

    async def _safe_emit_exec(self, port: str, *, exec_id: str | int) -> None:
        try:
            await self.emit_exec(str(port), exec_id=exec_id)
        except _REPLAYER_BUS_PUBLISH_ERRORS as exc:
            logger.exception("[%s:replayer] failed to emit exec port: %s", self.node_id, port, exc_info=exc)


def _event_ts_ms(event: Any) -> int:
    if event.type == "data_sample":
        return int(event.tick_ts_ms)
    return int(event.state_ts_ms)


def _coerce_time_mode(value: Any, *, default: str) -> str:
    text = str(value or "").strip().lower()
    if text in (TIME_MODE_RECORDED_EPOCH, TIME_MODE_OFFSET_FROM_PLAY):
        return text
    return str(default)


ReplayerRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.debug",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Replayer",
    description="Playback recorded data and sparse state changes for debugging.",
    tags=["record", "replay", "debug", "playback"],
    execInPorts=exec_port_specs(["play", "pause", "stop"]),
    execOutPorts=exec_port_specs(["sample", "started", "stopped", "looped", "done"]),
    dataInPorts=[],
    dataOutPorts=[
        F8DataPortSpec(
            name=_POSITION_PORT,
            description="Current playback position in milliseconds.",
            valueSchema=integer_schema(default=0, minimum=0),
            definitionProtected=False,
        ),
    ],
    editPolicy=F8SpecEditPolicy(
        stateFields=editable_collection_edit_policy(),
        dataOutPorts=editable_collection_edit_policy(),
    ),
    stateFields=[
        F8StateSpec(
            name="path",
            label="Path",
            description="Recording file path.",
            valueSchema=string_schema(),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="loop",
            label="Loop",
            description="Loop when playback reaches the end.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="timeMode",
            label="Time Mode",
            description="Playback time mapping mode.",
            valueSchema=string_schema(
                default=TIME_MODE_OFFSET_FROM_PLAY, enum=[TIME_MODE_RECORDED_EPOCH, TIME_MODE_OFFSET_FROM_PLAY]
            ),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="playing",
            label="Playing",
            description="Whether playback is currently running.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="durationMs",
            label="Duration",
            description="Readonly recording duration in milliseconds.",
            valueSchema=integer_schema(default=0, minimum=0),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="loaded",
            label="Loaded",
            description="Readonly flag indicating whether the recording is loaded.",
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.ro,
            valueRequired=True,
            showOnNode=True,
        ),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(ReplayerRuntimeNode.SPEC, ReplayerRuntimeNode, overwrite=True)
    return registry
