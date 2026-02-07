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
    integer_schema,
)
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import OperatorNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..constants import SERVICE_CLASS
from ..ui_bus import emit_ui_command


OPERATOR_CLASS = "f8.trackviz"
RENDERER_CLASS = "pystudio_trackviz"


@dataclass
class _Sample:
    ts_ms: int
    bbox: tuple[float, float, float, float] | None = None  # x1,y1,x2,y2
    keypoints: list[dict[str, Any]] | None = None
    kind: str = "track"  # "track" | "match" | other


class PyStudioTrackVizRuntimeNode(OperatorNode):
    """
    Studio-side node that visualizes tracking results.

    Expected input payloads:
    - Multi-target (from f8.detecttracker `detections`):
      { "tsMs": int, "width": int, "height": int, "tracks": [ {id, bbox, keypoints?}, ... ] }
    - Single-target (from f8.templatetracker `tracking`):
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
        )
        self._initial_state = dict(initial_state or {})
        self._config_loaded = False

        self._width: int | None = None
        self._height: int | None = None

        self._history_ms: int = 500
        self._history_frames: int = 10
        self._throttle_ms: int = 50

        self._tracks: dict[int, deque[_Sample]] = {}
        self._dirty: bool = False
        self._last_refresh_ms: int | None = None
        self._refresh_task: asyncio.Task[object] | None = None
        self._scheduled_refresh_ms: int | None = None

    async def close(self) -> None:
        try:
            t = self._refresh_task
            self._refresh_task = None
            self._scheduled_refresh_ms = None
            if t is not None:
                t.cancel()
        except Exception:
            pass
        try:
            if t is not None:
                await asyncio.gather(t, return_exceptions=True)
        except Exception:
            pass

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        f = str(field or "").strip()
        if f == "throttleMs":
            self._throttle_ms = await self._get_int_state("throttleMs", default=50, minimum=0, maximum=60000)
        elif f == "historyMs":
            self._history_ms = await self._get_int_state("historyMs", default=500, minimum=0, maximum=60000)
        elif f == "historyFrames":
            self._history_frames = await self._get_int_state("historyFrames", default=10, minimum=1, maximum=200)
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

        try:
            w = payload.get("width")
            h = payload.get("height")
            if w is not None and h is not None:
                self._width = int(w)
                self._height = int(h)
        except Exception:
            pass

        tracks_any = payload.get("tracks")
        tracks: list[dict[str, Any]] = [t for t in tracks_any if isinstance(t, dict)] if isinstance(tracks_any, list) else []

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
            tracks = [{"id": 1, "bbox": list(bbox0) if bbox0 is not None else None, "keypoints": kps0, "kind": kind0}]

        for t in tracks:
            if not isinstance(t, dict):
                continue
            if not isinstance(t, dict):
                continue
            try:
                tid = int(t.get("id"))
            except Exception:
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

            q = self._tracks.get(tid)
            if q is None:
                q = deque()
                self._tracks[tid] = q
            q.append(_Sample(ts_ms=now, bbox=bbox, keypoints=kps, kind=kind))

        self._dirty = True
        self._prune(now_ms=now)
        await self._schedule_refresh(now_ms=now)

    async def _ensure_config_loaded(self) -> None:
        if self._config_loaded:
            return
        self._throttle_ms = await self._get_int_state("throttleMs", default=50, minimum=0, maximum=60000)
        self._history_ms = await self._get_int_state("historyMs", default=500, minimum=0, maximum=60000)
        self._history_frames = await self._get_int_state("historyFrames", default=10, minimum=1, maximum=200)
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
        except Exception:
            return
        self._refresh_task = loop.create_task(
            self._flush_after(delay_ms),
            name=f"pystudio:trackviz:flush:{self.node_id}",
        )

    async def _flush_after(self, delay_ms: int) -> None:
        try:
            await asyncio.sleep(float(max(0, int(delay_ms))) / 1000.0)
        except Exception:
            return
        await self._flush(now_ms=int(time.time() * 1000))

    async def _flush(self, *, now_ms: int) -> None:
        self._scheduled_refresh_ms = None
        self._prune(now_ms=int(now_ms))

        changed = bool(self._dirty)
        if not changed and not self._tracks:
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
                hist.append(item)
            out_tracks.append({"id": int(tid), "history": hist})

        emit_ui_command(
            self.node_id,
            "trackviz.set",
            {
                "width": int(self._width or 0),
                "height": int(self._height or 0),
                "historyMs": int(self._history_ms),
                "historyFrames": int(self._history_frames),
                "tracks": out_tracks,
                "nowMs": int(now_ms),
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


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return PyStudioTrackVizRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

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
                    description="Tracking payload (e.g. from f8.detecttracker.detections).",
                    valueSchema=any_schema(),
                ),
            ],
            dataOutPorts=[],
            rendererClass=RENDERER_CLASS,
            stateFields=[
                F8StateSpec(
                    name="throttleMs",
                    label="Refresh (ms)",
                    description="UI refresh interval in milliseconds.",
                    valueSchema=integer_schema(default=50, minimum=0, maximum=60000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="historyMs",
                    label="History (ms)",
                    description="Keep history within this time window.",
                    valueSchema=integer_schema(default=500, minimum=0, maximum=60000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="historyFrames",
                    label="History (frames)",
                    description="Also keep up to this many recent samples per track.",
                    valueSchema=integer_schema(default=10, minimum=1, maximum=200),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
            ],
        ),
        overwrite=True,
    )
    return reg
