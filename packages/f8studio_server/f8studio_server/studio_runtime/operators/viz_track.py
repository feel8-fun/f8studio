from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import msgspec

from f8pysdk.codec import coerce_flag, coerce_float, coerce_int
from f8pysdk.f8_naming import ensure_token
from f8pysdk.registry import Registry
from f8pysdk.specs import (
    F8DataPortSpec,
    F8DataTypeSchema,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8StateAccess,
    F8StateSpec,
    array_schema,
    boolean_schema,
    complex_object_schema,
    integer_schema,
    number_schema,
    string_schema,
    video_frame_port,
)

from ..identifiers import SERVICE_CLASS
from ._viz_base import StudioVizRuntimeNodeBase, viz_sampling_state_fields
from .categories import PALETTE_CATEGORY_VIZ

OPERATOR_CLASS = "f8.viz.track"
RENDERER_CLASS = "viz_track"

logger = logging.getLogger(__name__)


class TrackPayloadInput(msgspec.Struct, rename="camel"):
    schema_version: str = ""
    ts_ms: int | None = None
    width: int | None = None
    height: int | None = None
    skeleton_protocol: str = ""
    tracks: list[object] = msgspec.field(default_factory=list)
    detections: list[object] = msgspec.field(default_factory=list)
    vectors: list[object] = msgspec.field(default_factory=list)


class TrackItemInput(msgspec.Struct, rename="camel"):
    id: int | None = None
    bbox: list[object] | None = None
    keypoints: list[object] | None = None
    kind: str = ""
    source: str = ""
    skeleton_protocol: str = ""


class KeypointInput(msgspec.Struct):
    x: float
    y: float
    score: float | None = None


class FlowVectorInput(msgspec.Struct):
    x: float
    y: float
    dx: float
    dy: float
    mag: float


@dataclass(frozen=True)
class TrackSample:
    ts_ms: int
    bbox: tuple[float, float, float, float] | None
    keypoints: list[dict[str, float | None]] | None
    kind: str
    skeleton_protocol: str | None


def _track_schema() -> F8DataTypeSchema:
    item = complex_object_schema(
        properties={
            "id": integer_schema(),
            "bbox": array_schema(items=number_schema()),
            "keypoints": array_schema(
                items=complex_object_schema(
                    properties={"x": number_schema(), "y": number_schema(), "score": number_schema()}
                )
            ),
            "kind": string_schema(),
            "skeletonProtocol": string_schema(),
        }
    )
    return complex_object_schema(
        properties={
            "schemaVersion": string_schema(),
            "tsMs": integer_schema(),
            "width": integer_schema(),
            "height": integer_schema(),
            "skeletonProtocol": string_schema(),
            "tracks": array_schema(items=item),
            "detections": array_schema(items=item),
        }
    )


class VizTrackRuntimeNode(StudioVizRuntimeNodeBase):
    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[port.name for port in (node.dataInPorts or [])],
            data_out_ports=[],
            state_fields=[field.name for field in (node.stateFields or [])],
            initial_state=initial_state,
        )
        state = self._initial_state
        self._throttle_ms = coerce_int(state.get("throttleMs"), default=50, minimum=0, maximum=60_000)
        self._history_ms = coerce_int(state.get("historyMs"), default=500, minimum=0, maximum=60_000)
        self._history_frames = coerce_int(state.get("historyFrames"), default=10, minimum=1, maximum=200)
        self._flow_arrow_scale = coerce_float(state.get("flowArrowScale"), default=1.0, minimum=0.1, maximum=20.0)
        self._flow_arrow_min_mag = coerce_float(state.get("flowArrowMinMag"), default=0.0, minimum=0.0, maximum=100.0)
        self._flow_arrow_max_count = coerce_int(state.get("flowArrowMaxCount"), default=2_000, minimum=100, maximum=20_000)
        self._show_dense_flow = coerce_flag(state.get("showDenseFlow"), default=True)
        self._show_sparse_flow = coerce_flag(state.get("showSparseFlow"), default=True)
        dense_mode = str(state.get("denseFlowMode") or "hsv").lower()
        self._dense_flow_mode = dense_mode if dense_mode in {"hsv", "arrows"} else "hsv"
        self._width = 0
        self._height = 0
        self._tracks: dict[int, deque[TrackSample]] = {}
        self._flow_payload: dict[str, object] | None = None
        self._dirty = False
        self._last_refresh_ms = 0
        self._refresh_task: asyncio.Task[object] | None = None
        self._warned_inputs: set[str] = set()

    async def close(self) -> None:
        task = self._refresh_task
        self._refresh_task = None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.presentation.emit(self.node_id, "viz.track.detach", {}, ts_ms=int(time.time() * 1_000))

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        if field == "throttleMs":
            self._throttle_ms = coerce_int(value, default=self._throttle_ms, minimum=0, maximum=60_000)
        elif field == "historyMs":
            self._history_ms = coerce_int(value, default=self._history_ms, minimum=0, maximum=60_000)
        elif field == "historyFrames":
            self._history_frames = coerce_int(value, default=self._history_frames, minimum=1, maximum=200)
        elif field == "flowArrowScale":
            self._flow_arrow_scale = coerce_float(value, default=self._flow_arrow_scale, minimum=0.1, maximum=20.0)
        elif field == "flowArrowMinMag":
            self._flow_arrow_min_mag = coerce_float(value, default=self._flow_arrow_min_mag, minimum=0.0, maximum=100.0)
        elif field == "flowArrowMaxCount":
            self._flow_arrow_max_count = coerce_int(value, default=self._flow_arrow_max_count, minimum=100, maximum=20_000)
        elif field == "showDenseFlow":
            self._show_dense_flow = coerce_flag(value, default=self._show_dense_flow)
        elif field == "showSparseFlow":
            self._show_sparse_flow = coerce_flag(value, default=self._show_sparse_flow)
        elif field == "denseFlowMode":
            normalized = str(value).strip().lower()
            self._dense_flow_mode = normalized if normalized in {"hsv", "arrows"} else "hsv"
        else:
            return
        self._dirty = True
        await self._schedule_refresh(int(ts_ms) if ts_ms is not None else int(time.time() * 1_000))

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if port not in {"detections", "inputData", "input"}:
            return
        try:
            payload = msgspec.convert(value, type=TrackPayloadInput, strict=False)
        except (msgspec.ValidationError, TypeError) as exc:
            signature = f"{port}:{type(value).__name__}:{exc}"
            if signature not in self._warned_inputs:
                self._warned_inputs.add(signature)
                logger.warning("track visualization ignored invalid payload node_id=%s port=%s: %s", self.node_id, port, exc)
            return
        now_ms = int(ts_ms) if ts_ms is not None else payload.ts_ms or int(time.time() * 1_000)
        if payload.width is not None:
            self._width = payload.width
        if payload.height is not None:
            self._height = payload.height
        if payload.schema_version == "f8visionFlowField/1":
            self._ingest_flow(payload, now_ms)
        else:
            self._ingest_tracks(payload, now_ms)
        self._dirty = True
        self._prune(now_ms)
        await self._schedule_refresh(now_ms)

    def _ingest_flow(self, payload: TrackPayloadInput, now_ms: int) -> None:
        vectors: list[dict[str, float]] = []
        for raw_vector in payload.vectors:
            try:
                vector = msgspec.convert(raw_vector, type=FlowVectorInput, strict=False)
            except (msgspec.ValidationError, TypeError):
                continue
            if vector.mag < self._flow_arrow_min_mag:
                continue
            vectors.append({"x": vector.x, "y": vector.y, "dx": vector.dx, "dy": vector.dy, "mag": vector.mag})
            if len(vectors) >= self._flow_arrow_max_count:
                break
        self._flow_payload = {
            "schemaVersion": "f8visionFlowField/1",
            "tsMs": payload.ts_ms or now_ms,
            "width": self._width,
            "height": self._height,
            "vectors": vectors,
        }

    def _ingest_tracks(self, payload: TrackPayloadInput, now_ms: int) -> None:
        raw_items = payload.tracks or payload.detections
        detections = not bool(payload.tracks)
        for index, raw_item in enumerate(raw_items, start=1):
            try:
                item = msgspec.convert(raw_item, type=TrackItemInput, strict=False)
            except (msgspec.ValidationError, TypeError):
                continue
            track_id = item.id if item.id is not None else index
            bbox = self._parse_bbox(item.bbox)
            keypoints = self._parse_keypoints(item.keypoints)
            protocol = item.skeleton_protocol or payload.skeleton_protocol
            sample = TrackSample(
                ts_ms=now_ms,
                bbox=bbox,
                keypoints=keypoints,
                kind="det" if detections else (item.kind or item.source or "track"),
                skeleton_protocol=protocol or None,
            )
            self._tracks.setdefault(track_id, deque()).append(sample)

    @staticmethod
    def _parse_bbox(value: list[object] | None) -> tuple[float, float, float, float] | None:
        if value is None or len(value) != 4:
            return None
        try:
            return (
                msgspec.convert(value[0], type=float, strict=False),
                msgspec.convert(value[1], type=float, strict=False),
                msgspec.convert(value[2], type=float, strict=False),
                msgspec.convert(value[3], type=float, strict=False),
            )
        except (msgspec.ValidationError, TypeError):
            return None

    @staticmethod
    def _parse_keypoints(value: list[object] | None) -> list[dict[str, float | None]] | None:
        if value is None:
            return None
        output: list[dict[str, float | None]] = []
        for raw_keypoint in value:
            try:
                keypoint = msgspec.convert(raw_keypoint, type=KeypointInput, strict=False)
            except (msgspec.ValidationError, TypeError):
                continue
            encoded: dict[str, float | None] = {"x": keypoint.x, "y": keypoint.y}
            if keypoint.score is not None:
                encoded["score"] = keypoint.score
            output.append(encoded)
        return output

    async def _schedule_refresh(self, now_ms: int) -> None:
        target_ms = self._last_refresh_ms + self._throttle_ms
        if self._throttle_ms <= 0 or self._last_refresh_ms <= 0 or now_ms >= target_ms:
            await self._flush(now_ms)
            return
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        self._refresh_task = asyncio.create_task(self._flush_after(target_ms - now_ms), name=f"web-studio:track:{self.node_id}")

    async def _flush_after(self, delay_ms: int) -> None:
        await asyncio.sleep(max(0, delay_ms) / 1_000)
        await self._flush(int(time.time() * 1_000))

    async def _flush(self, now_ms: int) -> None:
        self._prune(now_ms)
        tracks: list[dict[str, object]] = []
        for track_id, history in sorted(self._tracks.items()):
            samples: list[dict[str, object]] = []
            for sample in history:
                encoded: dict[str, object] = {"tsMs": sample.ts_ms, "kind": sample.kind}
                if sample.bbox is not None:
                    encoded["bbox"] = list(sample.bbox)
                if sample.keypoints is not None:
                    encoded["keypoints"] = sample.keypoints
                if sample.skeleton_protocol is not None:
                    encoded["skeletonProtocol"] = sample.skeleton_protocol
                samples.append(encoded)
            tracks.append({"id": track_id, "history": samples})
        self.presentation.emit(
            self.node_id,
            "viz.track.set",
            {
                "width": self._width,
                "height": self._height,
                "historyMs": self._history_ms,
                "historyFrames": self._history_frames,
                "throttleMs": self._throttle_ms,
                "tracks": tracks,
                "flow": self._flow_payload,
                "flowArrowScale": self._flow_arrow_scale,
                "flowArrowMinMag": self._flow_arrow_min_mag,
                "flowArrowMaxCount": self._flow_arrow_max_count,
                "showDenseFlow": self._show_dense_flow,
                "showSparseFlow": self._show_sparse_flow,
                "denseFlowMode": self._dense_flow_mode,
                "flowStreamKey": str(self.input_zenoh_key("flow") or ""),
                "videoStreamKey": str(self.input_zenoh_key("video") or ""),
                "nowMs": now_ms,
            },
            ts_ms=now_ms,
        )
        self._last_refresh_ms = now_ms
        self._dirty = False

    def _prune(self, now_ms: int) -> None:
        cutoff = now_ms - self._history_ms if self._history_ms > 0 else None
        for track_id in list(self._tracks):
            history = self._tracks[track_id]
            while cutoff is not None and history and history[0].ts_ms < cutoff:
                history.popleft()
            while len(history) > self._history_frames:
                history.popleft()
            if not history:
                del self._tracks[track_id]


VizTrackRuntimeNode.SPEC = F8OperatorSpec(
    schemaVersion=F8OperatorSchemaVersion.f8operator_1,
    serviceClass=SERVICE_CLASS,
    paletteCategory=PALETTE_CATEGORY_VIZ,
    operatorClass=OPERATOR_CLASS,
    version="0.1.0",
    label="Track Viz",
    description="Visualize tracking, pose, and optical-flow data in Web Studio.",
    tags=["viz", "tracking", "pose", "web"],
    dataInPorts=[
        F8DataPortSpec(name="detections", description="Tracking or detection payload.", valueSchema=_track_schema()),
        video_frame_port(name="video", description="Optional background video stream.", required=False),
        video_frame_port(name="flow", description="Optional dense optical-flow stream.", required=False),
    ],
    dataOutPorts=[],
    rendererClass=RENDERER_CLASS,
    stateFields=[
        F8StateSpec(name="throttleMs", valueSchema=integer_schema(default=50, minimum=0, maximum=60_000), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="historyMs", valueSchema=integer_schema(default=500, minimum=0, maximum=60_000), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="historyFrames", valueSchema=integer_schema(default=10, minimum=1, maximum=200), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="flowArrowScale", valueSchema=number_schema(default=1.0, minimum=0.1, maximum=20.0), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="flowArrowMinMag", valueSchema=number_schema(default=0.0, minimum=0.0, maximum=100.0), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="flowArrowMaxCount", valueSchema=integer_schema(default=2_000, minimum=100, maximum=20_000), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="showDenseFlow", valueSchema=boolean_schema(default=True), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="showSparseFlow", valueSchema=boolean_schema(default=True), access=F8StateAccess.rw, required=True),
        F8StateSpec(name="denseFlowMode", valueSchema=string_schema(default="hsv", enum=["hsv", "arrows"]), access=F8StateAccess.rw, required=True),
        *viz_sampling_state_fields(),
    ],
)


def register_operator(registry: Registry) -> Registry:
    registry.register_operator(VizTrackRuntimeNode.SPEC, VizTrackRuntimeNode, overwrite=True)
    return registry


__all__ = ["VizTrackRuntimeNode", "register_operator"]
