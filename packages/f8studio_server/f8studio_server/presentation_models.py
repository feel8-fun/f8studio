"""Typed payloads emitted to browser presentation and streaming consumers."""

from __future__ import annotations

from typing import Annotated
import msgspec
from f8pysdk.specs import F8JsonValue


class StreamHello(msgspec.Struct, frozen=True, kw_only=True, rename="camel", tag="stream.hello", tag_field="type"):
    server_epoch: str
    sequence: int
    resumed: bool


class LiveSnapshot(msgspec.Struct, frozen=True, kw_only=True, tag="live.snapshot", tag_field="type"):
    values: dict[str, F8JsonValue]


class LivePatch(msgspec.Struct, frozen=True, kw_only=True, tag="live.patch", tag_field="type"):
    set: dict[str, F8JsonValue]
    delete: list[str]


class SkeletonNode(msgspec.Struct, frozen=True, kw_only=True):
    index: int
    name: str
    pos: Annotated[list[float], msgspec.Meta(min_length=3, max_length=3)]
    rot: Annotated[list[float], msgspec.Meta(min_length=4, max_length=4)] | None


class SkeletonPerson(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    name: str
    bbox: list[float] | None
    skeleton_protocol: str
    skeleton_edges: list[Annotated[list[int], msgspec.Meta(min_length=2, max_length=2)]] | None
    nodes: list[SkeletonNode]


class SkeletonRenderFlags(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    show_person_boxes: bool | msgspec.UnsetType = msgspec.UNSET
    show_person_names: bool | msgspec.UnsetType = msgspec.UNSET
    show_bone_points: bool | msgspec.UnsetType = msgspec.UNSET
    show_skeleton_lines: bool | msgspec.UnsetType = msgspec.UNSET
    show_bone_axes: bool | msgspec.UnsetType = msgspec.UNSET
    show_bone_names: bool | msgspec.UnsetType = msgspec.UNSET
    auto_zoom_on_new_people: bool | msgspec.UnsetType = msgspec.UNSET
    marker_scale: float | msgspec.UnsetType = msgspec.UNSET


class SkeletonLimits(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    max_people: int
    max_bones_per_person: int


class SkeletonPerformanceHints(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    total_nodes: int | msgspec.UnsetType = msgspec.UNSET
    large_skeleton_mode: bool | msgspec.UnsetType = msgspec.UNSET
    suppress_bone_axes: bool | msgspec.UnsetType = msgspec.UNSET
    suppress_bone_names: bool | msgspec.UnsetType = msgspec.UNSET
    suppress_axis_tree: bool | msgspec.UnsetType = msgspec.UNSET
    suppress_person_boxes: bool | msgspec.UnsetType = msgspec.UNSET
    max_visible_bone_labels: int | None | msgspec.UnsetType = msgspec.UNSET
    recommended_fps_cap: int | msgspec.UnsetType = msgspec.UNSET


class SkeletonScene(msgspec.Struct, frozen=True, kw_only=True, rename="camel"):
    ts_ms: int
    world_up: str
    people: list[SkeletonPerson]
    ui_fps_cap: int | msgspec.UnsetType = msgspec.UNSET
    render_flags: SkeletonRenderFlags | msgspec.UnsetType = msgspec.UNSET
    limits: SkeletonLimits | msgspec.UnsetType = msgspec.UNSET
    performance_hints: SkeletonPerformanceHints | msgspec.UnsetType = msgspec.UNSET


class TrackHistorySample(msgspec.Struct, frozen=True, kw_only=True):
    tsMs: int
    kind: str
    bbox: list[float] | msgspec.UnsetType = msgspec.UNSET
    keypoints: list[dict[str, float | None]] | msgspec.UnsetType = msgspec.UNSET
    skeletonProtocol: str | msgspec.UnsetType = msgspec.UNSET


class TrackHistory(msgspec.Struct, frozen=True, kw_only=True):
    id: int
    history: list[TrackHistorySample]


class TrackFlow(msgspec.Struct, frozen=True, kw_only=True):
    schemaVersion: str
    tsMs: int
    width: int
    height: int
    vectors: list[dict[str, float]]


class TextUpdate(msgspec.Struct, frozen=True, kw_only=True):
    value: F8JsonValue


class WaveScene(msgspec.Struct, frozen=True, kw_only=True):
    series: dict[str, list[tuple[int, float]]]
    colors: dict[str, list[int]]
    windowMs: int
    nowMs: int
    showLegend: bool
    minVal: float | None
    maxVal: float | None


class VideoConfig(msgspec.Struct, frozen=True, kw_only=True):
    videoStreamKey: str
    throttleMs: int
    flowStreamKey: str
    flowDisplayMode: str
    flowMagScale: float
    flowStride: int
    scaleMode: str
    scalarStreamKey: str
    scalarDisplayMode: str
    scalarColormap: str
    scalarRangeMode: str
    scalarMin: float
    scalarMax: float
    scalarAutoPercentileLo: float
    scalarAutoPercentileHi: float
    scalarInvert: bool
    scalarNanMode: str


class AudioConfig(msgspec.Struct, frozen=True, kw_only=True):
    audioStreamKey: str
    throttleMs: int
    historyMs: int
    channel: int


class TCodeSnapshot(msgspec.Struct, frozen=True, kw_only=True):
    model: str
    line: str
    channels: dict[str, int]
    resetVersion: int


class TrackScene(msgspec.Struct, frozen=True, kw_only=True):
    width: int
    height: int
    historyMs: int
    historyFrames: int
    throttleMs: int
    tracks: list[TrackHistory]
    flow: TrackFlow | None
    flowArrowScale: float
    flowArrowMinMag: float
    flowArrowMaxCount: int
    showDenseFlow: bool
    showSparseFlow: bool
    denseFlowMode: str
    flowStreamKey: str
    videoStreamKey: str
    nowMs: int
