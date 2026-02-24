from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

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
    string_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from ..ui_bus import emit_ui_command
from ._viz_base import StudioVizRuntimeNodeBase, viz_sampling_state_fields


OPERATOR_CLASS = "f8.viz.track"
RENDERER_CLASS = "viz_track"


@dataclass
class _Sample:
    ts_ms: int
    bbox: tuple[float, float, float, float] | None = None  # x1,y1,x2,y2
    keypoints: list[dict[str, Any]] | None = None
    kind: str = "track"  # "track" | "match" | other
    skeleton_protocol: str | None = None


class VizTrackRuntimeNode(StudioVizRuntimeNodeBase):
    """
    Studio-side node that visualizes tracking results.

    Expected input payloads:
    - Multi-target (from f8.detecttracker `detections`):
      { "tsMs": int, "width": int, "height": int, "tracks": [ {id, bbox, keypoints?}, ... ] }
    - Multi-target (from f8.dl.detector / f8.dl.humandetector / f8.cvkit.templatematch):
      { "schemaVersion": "f8visionDetections/1", "tsMs": int, "width": int, "height": int, "detections": [ {bbox, keypoints?}, ... ] }
    - Single-target (from f8.cvkit.tracking `tracking`):
      { "tsMs": int, "width": int, "height": int, "bbox": [x1,y1,x2,y2] | null }

    The runtime node maintains a short history per track id, and emits a UI command
    that the render node draws (boxes, pose, and fading motion trails).
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[p.name for p in (node.dataInPorts or [])],
            data_out_ports=[],
            state_fields=[s.name for s in (node.stateFields or [])],
            initial_state=initial_state,
        )
        self._config_loaded = False

        self._width: int | None = None
        self._height: int | None = None

        self._history_ms: int = 500
        self._history_frames: int = 10
        self._throttle_ms: int = 50
        self._video_shm_name: str = ""
        self._flow_arrow_scale: float = 1.0
        self._flow_arrow_min_mag: float = 0.0
        self._flow_arrow_max_count: int = 2000

        self._tracks: dict[int, deque[_Sample]] = {}
        self._flow_payload: dict[str, Any] | None = None
        self._dirty: bool = False
        self._last_refresh_ms: int | None = None
        self._refresh_task: asyncio.Task[object] | None = None
        self._scheduled_refresh_ms: int | None = None

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._emit_initial_scene(), name=f"pystudio:trackviz:init:{self.node_id}")

    async def _emit_initial_scene(self) -> None:
        await self._ensure_config_loaded()
        self._dirty = True
        await self._schedule_refresh(now_ms=int(time.time() * 1000))

    async def close(self) -> None:
        try:
            t = self._refresh_task
            self._refresh_task = None
            self._scheduled_refresh_ms = None
            if t is not None:
                t.cancel()
        except (AttributeError, RuntimeError, TypeError):
            pass
        try:
            if t is not None:
                await asyncio.gather(t, return_exceptions=True)
        except (RuntimeError, TypeError):
            pass
        emit_ui_command(self.node_id, "viz.track.detach", {}, ts_ms=int(time.time() * 1000))

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        f = str(field or "").strip()
        if f == "throttleMs":
            self._throttle_ms = await self._get_int_state("throttleMs", default=50, minimum=0, maximum=60000)
            self._dirty = True
        elif f == "historyMs":
            self._history_ms = await self._get_int_state("historyMs", default=500, minimum=0, maximum=60000)
            self._dirty = True
        elif f == "historyFrames":
            self._history_frames = await self._get_int_state("historyFrames", default=10, minimum=1, maximum=200)
            self._dirty = True
        elif f == "videoShmName":
            self._video_shm_name = await self._get_str_state("videoShmName", default="")
            self._dirty = True
        elif f == "flowArrowScale":
            self._flow_arrow_scale = await self._get_float_state("flowArrowScale", default=1.0, minimum=0.1, maximum=20.0)
            self._dirty = True
        elif f == "flowArrowMinMag":
            self._flow_arrow_min_mag = await self._get_float_state("flowArrowMinMag", default=0.0, minimum=0.0, maximum=100.0)
            self._dirty = True
        elif f == "flowArrowMaxCount":
            self._flow_arrow_max_count = await self._get_int_state("flowArrowMaxCount", default=2000, minimum=100, maximum=20000)
            self._dirty = True
        else:
            return
        now = int(ts_ms) if ts_ms is not None else int(time.time() * 1000)
        await self._schedule_refresh(now_ms=now)

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        if str(port or "") not in ("detections", "inputData", "input"):
            return
        await self._ensure_config_loaded()

        payload = value if isinstance(value, dict) else {}
        now = int(ts_ms) if ts_ms is not None else int(payload.get("tsMs") or time.time() * 1000)
        schema_version = ""
        try:
            schema_version = str(payload.get("schemaVersion") or "").strip()
        except (AttributeError, TypeError, ValueError):
            schema_version = ""
        payload_skeleton_protocol = ""
        try:
            payload_skeleton_protocol = str(payload.get("skeletonProtocol") or "").strip()
        except (AttributeError, TypeError, ValueError):
            payload_skeleton_protocol = ""

        try:
            w = payload.get("width")
            h = payload.get("height")
            if w is not None and h is not None:
                self._width = int(w)
                self._height = int(h)
        except (AttributeError, TypeError, ValueError):
            pass

        if schema_version == "f8visionFlowField/1":
            vectors_in = payload.get("vectors")
            vectors_out: list[dict[str, float]] = []
            if isinstance(vectors_in, list):
                for item in vectors_in:
                    if not isinstance(item, dict):
                        continue
                    try:
                        x = float(item.get("x"))
                        y = float(item.get("y"))
                        dx = float(item.get("dx"))
                        dy = float(item.get("dy"))
                        mag = float(item.get("mag"))
                    except (TypeError, ValueError):
                        continue
                    if mag < float(self._flow_arrow_min_mag):
                        continue
                    vectors_out.append({"x": x, "y": y, "dx": dx, "dy": dy, "mag": mag})
                    if len(vectors_out) >= int(self._flow_arrow_max_count):
                        break
            self._flow_payload = {
                "schemaVersion": "f8visionFlowField/1",
                "tsMs": int(payload.get("tsMs") or now),
                "width": int(self._width or 0),
                "height": int(self._height or 0),
                "vectors": vectors_out,
            }
            self._dirty = True
            await self._schedule_refresh(now_ms=now)
            return

        tracks_any = payload.get("tracks")
        tracks: list[dict[str, Any]] = [t for t in tracks_any if isinstance(t, dict)] if isinstance(tracks_any, list) else []

        # New schema support: f8visionDetections/1
        if not tracks:
            dets_any = payload.get("detections")
            if isinstance(dets_any, list):
                det_tracks: list[dict[str, Any]] = []
                for i, det in enumerate(dets_any, start=1):
                    if not isinstance(det, dict):
                        continue
                    det_id = i
                    try:
                        if det.get("id") is not None:
                            det_id = int(det.get("id"))
                    except Exception:
                        det_id = i
                    bbox = None
                    try:
                        bb = det.get("bbox")
                        if isinstance(bb, (list, tuple)) and len(bb) == 4 and all(v is not None for v in bb):
                            x1, y1, x2, y2 = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                            bbox = (x1, y1, x2, y2)
                    except Exception:
                        bbox = None
                    kps = None
                    try:
                        kp = det.get("keypoints")
                        if isinstance(kp, list):
                            kps = [x for x in kp if isinstance(x, dict)]
                    except Exception:
                        kps = None
                    det_tracks.append(
                        {
                            "id": int(det_id),
                            "bbox": list(bbox) if bbox is not None else None,
                            "keypoints": kps,
                            "kind": "det",
                            "skeletonProtocol": str(det.get("skeletonProtocol") or payload_skeleton_protocol or "").strip(),
                        }
                    )
                tracks = det_tracks

        # Single-target compatibility: accept a top-level bbox and treat it as track id 1.
        if not tracks:
            bb0 = payload.get("bbox")
            bbox0 = None
            kind0 = "track"
            try:
                if isinstance(bb0, (list, tuple)) and len(bb0) == 4:
                    x1, y1, x2, y2 = (float(bb0[0]), float(bb0[1]), float(bb0[2]), float(bb0[3]))
                    bbox0 = (x1, y1, x2, y2)
            except Exception:
                bbox0 = None
            if bbox0 is None:
                # Fallback for template tracker when status="lost": visualize best match bbox (debug-friendly).
                try:
                    m = payload.get("match") if isinstance(payload.get("match"), dict) else {}
                    mb = (m or {}).get("bbox")
                    if isinstance(mb, (list, tuple)) and len(mb) == 4 and all(v is not None for v in mb):
                        x1, y1, x2, y2 = (float(mb[0]), float(mb[1]), float(mb[2]), float(mb[3]))
                        bbox0 = (x1, y1, x2, y2)
                        kind0 = "match"
                except Exception:
                    bbox0 = None
            kps0 = None
            try:
                kp0 = payload.get("keypoints")
                if isinstance(kp0, list):
                    kps0 = [x for x in kp0 if isinstance(x, dict)]
            except Exception:
                kps0 = None
            tracks = [
                {
                    "id": 1,
                    "bbox": list(bbox0) if bbox0 is not None else None,
                    "keypoints": kps0,
                    "kind": kind0,
                    "skeletonProtocol": payload_skeleton_protocol,
                }
            ]

        for t in tracks:
            if not isinstance(t, dict):
                continue
            if not isinstance(t, dict):
                continue
            try:
                tid = int(t.get("id"))
            except (TypeError, ValueError):
                continue
            bbox = None
            try:
                bb = t.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4 and all(v is not None for v in bb):
                    x1, y1, x2, y2 = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
                    bbox = (x1, y1, x2, y2)
            except Exception:
                bbox = None
            kps = None
            try:
                kp = t.get("keypoints")
                if isinstance(kp, list):
                    kps = [x for x in kp if isinstance(x, dict)]
            except Exception:
                kps = None
            kind = "track"
            try:
                kind = str(t.get("kind") or t.get("source") or "track")
            except Exception:
                kind = "track"
            skeleton_protocol = ""
            try:
                skeleton_protocol = str(t.get("skeletonProtocol") or payload_skeleton_protocol or "").strip()
            except Exception:
                skeleton_protocol = ""

            q = self._tracks.get(tid)
            if q is None:
                q = deque()
                self._tracks[tid] = q
            q.append(
                _Sample(
                    ts_ms=now,
                    bbox=bbox,
                    keypoints=kps,
                    kind=kind,
                    skeleton_protocol=(skeleton_protocol if skeleton_protocol else None),
                )
            )

        self._dirty = True
        self._prune(now_ms=now)
        await self._schedule_refresh(now_ms=now)

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        self._throttle_ms = await self._get_int_state("throttleMs", default=50, minimum=0, maximum=60000)
        self._history_ms = await self._get_int_state("historyMs", default=500, minimum=0, maximum=60000)
        self._history_frames = await self._get_int_state("historyFrames", default=10, minimum=1, maximum=200)
        self._video_shm_name = await self._get_str_state("videoShmName", default="")
        self._flow_arrow_scale = await self._get_float_state("flowArrowScale", default=1.0, minimum=0.1, maximum=20.0)
        self._flow_arrow_min_mag = await self._get_float_state("flowArrowMinMag", default=0.0, minimum=0.0, maximum=100.0)
        self._flow_arrow_max_count = await self._get_int_state("flowArrowMaxCount", default=2000, minimum=100, maximum=20000)
        self._config_loaded = True

    async def _schedule_refresh(self, *, now_ms: int) -> None:
        throttle_ms = max(0, int(self._throttle_ms))
        last_refresh = int(self._last_refresh_ms or 0)
        if throttle_ms <= 0 or last_refresh <= 0:
            await self._flush(now_ms=now_ms)
            return

        target_ms = last_refresh + throttle_ms
        if int(now_ms) >= int(target_ms):
            await self._flush(now_ms=now_ms)
            return

        if self._refresh_task is not None and not self._refresh_task.done():
            return

        delay_ms = max(0, int(target_ms) - int(now_ms))
        self._scheduled_refresh_ms = int(target_ms)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._refresh_task = loop.create_task(
            self._flush_after(delay_ms),
            name=f"pystudio:trackviz:flush:{self.node_id}",
        )

    async def _flush_after(self, delay_ms: int) -> None:
        try:
            await asyncio.sleep(float(max(0, int(delay_ms))) / 1000.0)
        except (RuntimeError, TypeError, ValueError):
            return
        await self._flush(now_ms=int(time.time() * 1000))

    async def _flush(self, *, now_ms: int) -> None:
        self._scheduled_refresh_ms = None
        self._prune(now_ms=int(now_ms))

        changed = bool(self._dirty)
        if not changed and not self._tracks and self._flow_payload is None:
            self._last_refresh_ms = int(now_ms)
            return

        out_tracks: list[dict[str, Any]] = []
        for tid, q in sorted(self._tracks.items(), key=lambda kv: kv[0]):
            hist: list[dict[str, Any]] = []
            for s in q:
                item: dict[str, Any] = {"tsMs": int(s.ts_ms)}
                if s.bbox is not None:
                    item["bbox"] = [float(x) for x in s.bbox]
                if s.keypoints is not None:
                    item["keypoints"] = s.keypoints
                if s.kind:
                    item["kind"] = str(s.kind)
                if s.skeleton_protocol:
                    item["skeletonProtocol"] = str(s.skeleton_protocol)
                hist.append(item)
            out_tracks.append({"id": int(tid), "history": hist})

        emit_ui_command(
            self.node_id,
            "viz.track.set",
            {
                "width": int(self._width or 0),
                "height": int(self._height or 0),
                "historyMs": int(self._history_ms),
                "historyFrames": int(self._history_frames),
                "throttleMs": int(self._throttle_ms),
                "tracks": out_tracks,
                "flow": self._flow_payload if self._flow_payload is not None else None,
                "flowArrowScale": float(self._flow_arrow_scale),
                "flowArrowMinMag": float(self._flow_arrow_min_mag),
                "nowMs": int(now_ms),
                "videoShmName": str(self._video_shm_name or "").strip(),
            },
            ts_ms=int(now_ms),
        )

        self._last_refresh_ms = int(now_ms)
        self._dirty = False

    def _prune(self, *, now_ms: int) -> None:
        window_ms = max(0, int(self._history_ms))
        frame_limit = max(1, int(self._history_frames))
        cutoff = int(now_ms) - int(window_ms) if window_ms > 0 else None

        drop: list[int] = []
        for tid, q in self._tracks.items():
            if cutoff is not None:
                while q and int(q[0].ts_ms) < cutoff:
                    q.popleft()
            while len(q) > frame_limit:
                q.popleft()
            if not q:
                drop.append(tid)
        for tid in drop:
            self._tracks.pop(tid, None)

    async def _get_int_state(self, name: str, *, default: int, minimum: int, maximum: int) -> int:
        v: Any = None
        try:
            v = await self.get_state_value(name)
        except Exception:
            v = None
        if v is None:
            v = self._initial_state.get(name)
        try:
            out = int(v) if v is not None else int(default)
        except Exception:
            out = int(default)
        if out < minimum:
            out = minimum
        if out > maximum:
            out = maximum
        return out

    async def _get_str_state(self, name: str, *, default: str) -> str:
        v: Any = None
        try:
            v = await self.get_state_value(name)
        except Exception:
            v = None
        if v is None:
            v = self._initial_state.get(name)
        try:
            return str(v) if v is not None else str(default)
        except Exception:
            return str(default)

    async def _get_float_state(self, name: str, *, default: float, minimum: float, maximum: float) -> float:
        v: Any = None
        try:
            v = await self.get_state_value(name)
        except Exception:
            v = None
        if v is None:
            v = self._initial_state.get(name)
        try:
            out = float(v) if v is not None else float(default)
        except Exception:
            out = float(default)
        if out < minimum:
            out = minimum
        if out > maximum:
            out = maximum
        return out

def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return VizTrackRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)

    reg.register_operator_spec(
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=SERVICE_CLASS,
            operatorClass=OPERATOR_CLASS,
            version="0.0.1",
            label="TrackViz",
            description="Visualize tracking boxes/poses from a data port (history + fading trail).",
            tags=["viz", "tracking", "pose", "ui"],
            dataInPorts=[
                F8DataPortSpec(
                    name="detections",
                    description="Tracking/detection payload (f8.detecttracker or f8visionDetections/1).",
                    valueSchema=any_schema(),
                ),
            ],
            dataOutPorts=[],
            rendererClass=RENDERER_CLASS,
            stateFields=[
                F8StateSpec(
                    name="uiUpdate",
                    label="UI Update",
                    description="Pause/resume embedded TrackViz updates in the editor.",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="throttleMs",
                    label="Refresh (ms)",
                    description="UI refresh interval in milliseconds.",
                    valueSchema=integer_schema(default=50, minimum=0, maximum=60000),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="historyMs",
                    label="History (ms)",
                    description="Keep history within this time window.",
                    valueSchema=integer_schema(default=500, minimum=0, maximum=60000),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="historyFrames",
                    label="History (frames)",
                    description="Also keep up to this many recent samples per track.",
                    valueSchema=integer_schema(default=10, minimum=1, maximum=200),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="videoShmName",
                    label="Video SHM Name",
                    description="Optional BGRA Video SHM mapping name used as TrackViz background.",
                    valueSchema=string_schema(default=""),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="flowArrowScale",
                    label="Flow Arrow Scale",
                    description="Scale factor applied to optical-flow arrow vectors.",
                    valueSchema=number_schema(default=1.0, minimum=0.1, maximum=20.0),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="flowArrowMinMag",
                    label="Flow Min Magnitude",
                    description="Hide optical-flow arrows whose magnitude is below this value.",
                    valueSchema=number_schema(default=0.0, minimum=0.0, maximum=100.0),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="flowArrowMaxCount",
                    label="Flow Max Arrows",
                    description="Maximum number of optical-flow arrows rendered per frame.",
                    valueSchema=integer_schema(default=2000, minimum=100, maximum=20000),
                    access=F8StateAccess.rw,
                    showOnNode=False,
                ),
                *viz_sampling_state_fields(show_on_node=False),
            ],
        ),
        overwrite=True,
    )
    return reg
