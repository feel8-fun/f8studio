from __future__ import annotations

from f8pysdk.specs import exec_port_specs

import time
from dataclasses import dataclass
from typing import Any, Final

from f8pysdk.codec import coerce_flag, coerce_int, parse_number
from f8pysdk.specs import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    boolean_schema,
    complex_object_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.f8_naming import ensure_token
from f8pysdk.nodes import OperatorNode
from f8pysdk.registry import Registry

from ..constants import SERVICE_CLASS
from ._ports import exec_out_ports

OPERATOR_CLASS: Final[str] = "f8.playback_sync"

def _playback_input_schema():
    return complex_object_schema(
        properties={
            "position": number_schema(),
            "duration": number_schema(),
            "playing": boolean_schema(),
            "videoId": string_schema(),
        }
    )

@dataclass(frozen=True)
class _PlaybackSample:
    video_id: str | None
    position_s: float
    duration_s: float | None
    playing: bool

@dataclass(frozen=True)
class _EstimateSnapshot:
    video_id: str | None
    position_s: float | None
    raw_position_s: float | None
    duration_s: float | None
    playing: bool
    age_ms: int
    stale: bool

def _parse_playback(value: Any, *, default_playing: bool) -> _PlaybackSample | None:
    if not isinstance(value, dict):
        return None
    position_s = parse_number(value.get("position"))
    if position_s is None:
        return None
    duration_s = parse_number(value.get("duration"))
    if duration_s is not None and duration_s < 0.0:
        duration_s = None
    playing = coerce_flag(value.get("playing"), default=default_playing)

    raw_video_id = value.get("videoId")
    video_id: str | None = None
    if isinstance(raw_video_id, str):
        stripped = raw_video_id.strip()
        if stripped:
            video_id = stripped

    return _PlaybackSample(
        video_id=video_id,
        position_s=float(position_s),
        duration_s=duration_s,
        playing=bool(playing),
    )

class PlaybackSyncRuntimeNode(OperatorNode):
    """
    Extrapolates playback position from sparse IMPlayer playback state updates.

    Input payload (dataIn `playback`) is expected to include:
    - position (seconds)
    - duration (optional seconds)
    - playing (optional bool)
    - videoId (optional string)
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

        self._max_extrapolate_ms = coerce_int(
            self._initial_state.get("maxExtrapolateMs"), default=3000, minimum=0, maximum=600_000
        )
        self._playback_rate = self._coerce_playback_rate(self._initial_state.get("playbackRate"), default=1.0)
        self._clamp_to_duration = coerce_flag(self._initial_state.get("clampToDuration"), default=True)

        self._anchor_video_id: str | None = None
        self._anchor_position_s: float | None = None
        self._anchor_time_s: float | None = None
        self._anchor_duration_s: float | None = None
        self._anchor_playing = False
        self._last_update_time_s: float | None = None
        self._last_signature: tuple[str | None, float, float | None, bool] | None = None

        self._last_ctx_id: str | int | None = None
        self._last_snapshot: _EstimateSnapshot | None = None

    async def on_exec(self, _exec_id: str | int, _in_port: str | None = None) -> list[str]:
        return list(self._exec_out_ports)

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if str(port) != "playback":
            return
        observed_s = time.monotonic()
        del ts_ms
        self._update_anchor(value, observed_s=observed_s)
        self._last_ctx_id = None
        self._last_snapshot = None

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del ts_ms
        name = str(field or "").strip()
        if name == "maxExtrapolateMs":
            self._max_extrapolate_ms = coerce_int(value, default=self._max_extrapolate_ms, minimum=0, maximum=600_000)
            return
        if name == "playbackRate":
            self._playback_rate = self._coerce_playback_rate(value, default=self._playback_rate)
            return
        if name == "clampToDuration":
            self._clamp_to_duration = coerce_flag(value, default=self._clamp_to_duration)
            return

    async def validate_state(
        self, field: str, value: Any, *, ts_ms: int | None = None, meta: dict[str, Any] | None = None
    ) -> Any:
        del ts_ms, meta
        name = str(field or "").strip()
        if name == "maxExtrapolateMs":
            return coerce_int(value, default=3000, minimum=0, maximum=600_000)
        if name == "playbackRate":
            rate = parse_number(value)
            if rate is None:
                raise ValueError("playbackRate must be a number")
            if rate < 0.0 or rate > 16.0:
                raise ValueError("playbackRate must be in [0, 16]")
            return float(rate)
        if name == "clampToDuration":
            return coerce_flag(value, default=True)
        return value

    async def compute_output(self, port: str, ctx_id: str | int | None = None) -> Any:
        name = str(port or "")
        if name not in ("position", "rawPosition", "duration", "playing", "videoId", "ageMs", "stale"):
            return None

        if ctx_id is not None and ctx_id == self._last_ctx_id and self._last_snapshot is not None:
            return self._select_output(self._last_snapshot, port=name)

        observed_s = time.monotonic()
        payload = await self.pull("playback", ctx_id=ctx_id)
        self._update_anchor(payload, observed_s=observed_s)
        snapshot = self._build_snapshot(now_s=observed_s)

        self._last_ctx_id = ctx_id
        self._last_snapshot = snapshot
        return self._select_output(snapshot, port=name)

    @staticmethod
    def _coerce_playback_rate(value: Any, *, default: float) -> float:
        rate = parse_number(value)
        if rate is None:
            return float(default)
        return float(max(0.0, min(16.0, rate)))

    def _update_anchor(self, payload: Any, *, observed_s: float) -> None:
        sample = _parse_playback(payload, default_playing=self._anchor_playing)
        if sample is None:
            return

        signature = (sample.video_id, sample.position_s, sample.duration_s, sample.playing)
        if signature == self._last_signature:
            return

        self._anchor_video_id = sample.video_id
        self._anchor_position_s = float(sample.position_s)
        self._anchor_time_s = float(observed_s)
        self._anchor_duration_s = sample.duration_s
        self._anchor_playing = bool(sample.playing)
        self._last_update_time_s = float(observed_s)
        self._last_signature = signature

    def _build_snapshot(self, *, now_s: float) -> _EstimateSnapshot:
        if self._anchor_position_s is None:
            return _EstimateSnapshot(
                video_id=self._anchor_video_id,
                position_s=None,
                raw_position_s=None,
                duration_s=self._anchor_duration_s,
                playing=self._anchor_playing,
                age_ms=0,
                stale=False,
            )

        age_ms = 0
        if self._last_update_time_s is not None:
            age_ms = max(0, int((now_s - self._last_update_time_s) * 1000.0))

        delta_s = 0.0
        if self._anchor_playing and self._anchor_time_s is not None:
            elapsed_s = max(0.0, float(now_s - self._anchor_time_s))
            max_extra_s = float(self._max_extrapolate_ms) / 1000.0
            if self._max_extrapolate_ms <= 0:
                delta_s = elapsed_s * float(self._playback_rate)
            else:
                delta_s = min(elapsed_s, max_extra_s) * float(self._playback_rate)

        position_s = float(self._anchor_position_s + delta_s)
        if position_s < 0.0:
            position_s = 0.0
        if self._clamp_to_duration and self._anchor_duration_s is not None:
            position_s = min(position_s, float(self._anchor_duration_s))

        stale = self._max_extrapolate_ms > 0 and age_ms > int(self._max_extrapolate_ms)
        return _EstimateSnapshot(
            video_id=self._anchor_video_id,
            position_s=float(position_s),
            raw_position_s=float(self._anchor_position_s),
            duration_s=self._anchor_duration_s,
            playing=self._anchor_playing,
            age_ms=age_ms,
            stale=bool(stale),
        )

    @staticmethod
    def _select_output(snapshot: _EstimateSnapshot, *, port: str) -> Any:
        if port == "position":
            return snapshot.position_s
        if port == "rawPosition":
            return snapshot.raw_position_s
        if port == "duration":
            return snapshot.duration_s
        if port == "playing":
            return snapshot.playing
        if port == "videoId":
            return snapshot.video_id
        if port == "ageMs":
            return snapshot.age_ms
        if port == "stale":
            return snapshot.stale
        return None

PlaybackSyncRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=f"{SERVICE_CLASS}.motion",
    operatorClass=OPERATOR_CLASS,
    version="0.0.1",
    label="Playback Sync",
    description="Extrapolates IMPlayer playback position between sparse playback state updates.",
    tags=["playback", "estimate", "timing", "media", "sync"],
    execInPorts=exec_port_specs([]),
    execOutPorts=exec_port_specs([]),
    dataInPorts=[
        F8DataPortSpec(
            name="playback",
            description="Playback payload from f8.implayer/playback (position/duration/playing/videoId).",
            valueSchema=_playback_input_schema(),
            definitionProtected=False,
        ),
    ],
    dataOutPorts=[
        F8DataPortSpec(name="position", description="Estimated playback position (seconds).", valueSchema=number_schema()),
        F8DataPortSpec(
            name="rawPosition", description="Latest raw position from playback payload (seconds).", valueSchema=number_schema(),
            showOnNode=False,
        ),
        F8DataPortSpec(name="duration", description="Latest duration (seconds).", valueSchema=number_schema(), showOnNode=False),
        F8DataPortSpec(name="playing", description="Latest playing flag.", valueSchema=boolean_schema(), showOnNode=False),
        F8DataPortSpec(name="videoId", description="Latest video id.", valueSchema=string_schema(), showOnNode=False),
        F8DataPortSpec(name="ageMs", description="Age of latest playback sample in milliseconds.", valueSchema=integer_schema(), showOnNode=False),
        F8DataPortSpec(name="stale", description="True if sample age exceeds max extrapolation window.", valueSchema=boolean_schema(), showOnNode=False),
    ],
    stateFields=[
        F8StateSpec(
            name="maxExtrapolateMs",
            label="Max Extrapolate (ms)",
            description="Limit extrapolation horizon to avoid drift when playback state is stale (0 = unlimited).",
            valueSchema=integer_schema(default=1000, minimum=0, maximum=600_000),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="playbackRate",
            label="Playback Rate",
            description="Rate multiplier used for extrapolation when playing.",
            valueSchema=number_schema(default=1.0, minimum=0.0, maximum=16.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="clampToDuration",
            label="Clamp To Duration",
            description="Clamp estimated position to latest duration when available.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
    ],
)

def register_operator(registry: Registry) -> Registry:
    registry.register_operator(PlaybackSyncRuntimeNode.SPEC, PlaybackSyncRuntimeNode, overwrite=True)
    return registry
