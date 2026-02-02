from __future__ import annotations

import asyncio
import base64
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import cv2  # type: ignore
import numpy as np  # type: ignore

from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import ServiceNode
from f8pysdk.shm.video import VideoShmReader, default_video_shm_name

from .constants import SERVICE_CLASS


def _coerce_int(v: Any, *, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        out = int(v)
    except Exception:
        out = int(default)
    if minimum is not None and out < minimum:
        out = int(minimum)
    if maximum is not None and out > maximum:
        out = int(maximum)
    return out


def _coerce_float(v: Any, *, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        out = float(v)
    except Exception:
        out = float(default)
    if minimum is not None and out < minimum:
        out = float(minimum)
    if maximum is not None and out > maximum:
        out = float(maximum)
    return out


def _coerce_str(v: Any, *, default: str = "") -> str:
    try:
        s = str(v) if v is not None else ""
    except Exception:
        s = ""
    s = s.strip()
    return s if s else default


def _coerce_bool(v: Any, *, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = _coerce_str(v).lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return bool(default)


class _RollingWindow:
    def __init__(self, *, window_ms: int) -> None:
        self.window_ms = int(window_ms)
        self._q: deque[tuple[int, float]] = deque()
        self._sum = 0.0

    def set_window(self, window_ms: int) -> None:
        self.window_ms = int(max(0, window_ms))
        self._q.clear()
        self._sum = 0.0

    def push(self, ts_ms: int, v: float) -> None:
        self._q.append((int(ts_ms), float(v)))
        self._sum += float(v)
        self.prune(ts_ms)

    def prune(self, now_ms: int) -> None:
        win = int(self.window_ms)
        if win <= 0:
            self._q.clear()
            self._sum = 0.0
            return
        cutoff = int(now_ms) - win
        while self._q and int(self._q[0][0]) < cutoff:
            _, v = self._q.popleft()
            self._sum -= float(v)

    def mean(self, now_ms: int) -> float | None:
        self.prune(now_ms)
        n = len(self._q)
        if n <= 0:
            return None
        return float(self._sum) / float(n)

    def count(self, now_ms: int) -> int:
        self.prune(now_ms)
        return int(len(self._q))


class _Telemetry:
    def __init__(self) -> None:
        self.interval_ms = 1000
        self.window_ms = 2000
        self._last_emit_ms = 0

        self._frames = _RollingWindow(window_ms=self.window_ms)
        self._dup_skipped = _RollingWindow(window_ms=self.window_ms)
        self._matches = _RollingWindow(window_ms=self.window_ms)
        self._track_ok = _RollingWindow(window_ms=self.window_ms)

        self._t_total = _RollingWindow(window_ms=self.window_ms)
        self._t_shm_wait_ms = _RollingWindow(window_ms=self.window_ms)
        self._t_shm_read_ms = _RollingWindow(window_ms=self.window_ms)
        self._t_pre_ms = _RollingWindow(window_ms=self.window_ms)
        self._t_match_ms = _RollingWindow(window_ms=self.window_ms)
        self._t_track_ms = _RollingWindow(window_ms=self.window_ms)
        self._t_emit_ms = _RollingWindow(window_ms=self.window_ms)

        self._track_ok_last = 0
        self._match_last = 0

    def set_config(self, *, interval_ms: int, window_ms: int) -> None:
        interval_ms_i = max(0, int(interval_ms))
        window_ms_i = max(100, int(window_ms))
        # Avoid clearing the rolling window unless config actually changes.
        if int(interval_ms_i) == int(self.interval_ms) and int(window_ms_i) == int(self.window_ms):
            return
        self.interval_ms = int(interval_ms_i)
        self.window_ms = int(window_ms_i)
        for w in (
            self._frames,
            self._dup_skipped,
            self._matches,
            self._track_ok,
            self._t_total,
            self._t_shm_wait_ms,
            self._t_shm_read_ms,
            self._t_pre_ms,
            self._t_match_ms,
            self._t_track_ms,
            self._t_emit_ms,
        ):
            w.set_window(self.window_ms)

    def observe(
        self,
        *,
        ts_ms: int,
        dup_skipped: int,
        did_match: bool,
        track_ok: bool,
        total_ms: float,
        shm_wait_ms: float,
        shm_read_ms: float,
        pre_ms: float,
        match_ms: float,
        track_ms: float,
        emit_ms: float,
    ) -> None:
        self._frames.push(ts_ms, 1.0)
        self._dup_skipped.push(ts_ms, float(dup_skipped))
        self._matches.push(ts_ms, 1.0 if did_match else 0.0)
        self._track_ok.push(ts_ms, 1.0 if track_ok else 0.0)

        self._t_total.push(ts_ms, float(total_ms))
        self._t_shm_wait_ms.push(ts_ms, float(shm_wait_ms))
        self._t_shm_read_ms.push(ts_ms, float(shm_read_ms))
        self._t_pre_ms.push(ts_ms, float(pre_ms))
        self._t_match_ms.push(ts_ms, float(match_ms))
        self._t_track_ms.push(ts_ms, float(track_ms))
        self._t_emit_ms.push(ts_ms, float(emit_ms))

        self._track_ok_last = 1 if track_ok else 0
        self._match_last = 1 if did_match else 0

    def should_emit(self, now_ms: int) -> bool:
        if self.interval_ms <= 0:
            return False
        return (int(now_ms) - int(self._last_emit_ms)) >= int(self.interval_ms)

    def mark_emitted(self, now_ms: int) -> None:
        self._last_emit_ms = int(now_ms)

    def summary(
        self,
        *,
        now_ms: int,
        node_id: str,
        tracker_kind: str,
        match_method: str,
        shm_name: str,
        shm_has_event: bool,
        shm_wait_timeout_ms: int,
        frame_id_last_seen: int,
        frame_id_last_processed: int | None,
    ) -> dict[str, Any]:
        frames = int(self._frames.count(now_ms))
        fps = float(frames) * 1000.0 / float(max(1, int(self.window_ms)))
        dup_avg = self._dup_skipped.mean(now_ms)
        match_frac = self._matches.mean(now_ms)
        track_frac = self._track_ok.mean(now_ms)

        return {
            "schemaVersion": "f8telemetry/1",
            "tsMs": int(now_ms),
            "nodeId": str(node_id),
            "serviceClass": SERVICE_CLASS,
            "windowMs": int(self.window_ms),
            "source": {"shmName": str(shm_name), "hasEvent": bool(shm_has_event), "waitTimeoutMs": int(shm_wait_timeout_ms)},
            "frameId": {
                "lastSeen": int(frame_id_last_seen),
                "lastProcessed": int(frame_id_last_processed) if frame_id_last_processed is not None else None,
                "dupSkippedAvg": float(dup_avg) if dup_avg is not None else 0.0,
            },
            "counts": {"frames": int(frames), "trackOkLast": int(self._track_ok_last), "matchLast": int(self._match_last)},
            "rates": {
                "fps": float(fps),
                "matchFraction": float(match_frac) if match_frac is not None else 0.0,
                "trackOkFraction": float(track_frac) if track_frac is not None else 0.0,
            },
            "timingsMsAvg": {
                "total": self._t_total.mean(now_ms),
                "shmWait": self._t_shm_wait_ms.mean(now_ms),
                "shmRead": self._t_shm_read_ms.mean(now_ms),
                "preprocess": self._t_pre_ms.mean(now_ms),
                "match": self._t_match_ms.mean(now_ms),
                "trackUpdate": self._t_track_ms.mean(now_ms),
                "emit": self._t_emit_ms.mean(now_ms),
            },
            "runtime": {"trackerKind": str(tracker_kind), "matchMethod": str(match_method)},
        }


@dataclass
class _Capture:
    frame_id: int
    ts_ms: int
    bgr: np.ndarray


def _create_tracker(kind: str) -> Any | None:
    k = str(kind or "").lower()
    if k in ("none", ""):
        return None
    try:
        if k == "csrt":
            fn = getattr(cv2, "TrackerCSRT_create", None) or getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None)
            return fn() if callable(fn) else None
        if k == "kcf":
            fn = getattr(cv2, "TrackerKCF_create", None) or getattr(getattr(cv2, "legacy", None), "TrackerKCF_create", None)
            return fn() if callable(fn) else None
        if k == "mosse":
            fn = getattr(cv2, "TrackerMOSSE_create", None) or getattr(getattr(cv2, "legacy", None), "TrackerMOSSE_create", None)
            return fn() if callable(fn) else None
    except Exception:
        return None
    return None


def _as_b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _from_b64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=False)


def _encode_image_b64(
    bgr: np.ndarray,
    *,
    fmt: str,
    quality: int,
    max_bytes: int,
    max_width: int,
    max_height: int,
) -> tuple[str, dict[str, Any]]:
    img = bgr

    def _resize_if_needed(x: np.ndarray) -> np.ndarray:
        hh, ww = int(x.shape[0]), int(x.shape[1])
        scale = 1.0
        if max_width > 0:
            scale = min(scale, float(max_width) / float(max(1, ww)))
        if max_height > 0:
            scale = min(scale, float(max_height) / float(max(1, hh)))
        if scale >= 1.0:
            return x
        nw = max(1, int(round(ww * scale)))
        nh = max(1, int(round(hh * scale)))
        return cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)

    img = _resize_if_needed(img)

    fmt_l = str(fmt or "jpg").lower()
    ext = ".jpg" if fmt_l == "jpg" else ".png"
    q = int(max(1, min(100, int(quality))))

    last_b64 = ""
    last_raw_len = 0
    for _ in range(16):
        params: list[int] = []
        if ext == ".jpg":
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(q)]
        ok, buf = cv2.imencode(ext, img, params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        raw = bytes(buf)
        last_raw_len = len(raw)
        b64 = _as_b64(raw)
        last_b64 = b64
        if len(b64) <= int(max_bytes):
            meta = {"format": fmt_l, "width": int(img.shape[1]), "height": int(img.shape[0]), "bytes": int(len(raw)), "b64Bytes": int(len(b64))}
            return b64, meta

        if ext == ".jpg" and q > 30:
            q = max(30, int(q * 0.85))
            continue

        hh, ww = int(img.shape[0]), int(img.shape[1])
        if ww <= 64 or hh <= 64:
            break
        img = cv2.resize(img, (max(64, int(ww * 0.85)), max(64, int(hh * 0.85))), interpolation=cv2.INTER_AREA)

    raise ValueError(f"encoded image exceeds maxBytes={max_bytes} (b64 len={len(last_b64)} raw={last_raw_len})")


def _decode_png_b64_to_bgr(png_b64: str) -> np.ndarray | None:
    s = str(png_b64 or "").strip()
    if not s:
        return None
    try:
        raw = _from_b64(s)
        buf = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return np.ascontiguousarray(img)
    except Exception:
        return None


def _xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    x = int(min(x1, x2))
    y = int(min(y1, y2))
    w = int(abs(x2 - x1))
    h = int(abs(y2 - y1))
    return x, y, w, h


def _clip_xywh(x: int, y: int, w: int, h: int, *, width: int, height: int) -> tuple[int, int, int, int]:
    x = max(0, min(int(x), max(0, width - 1)))
    y = max(0, min(int(y), max(0, height - 1)))
    w = max(1, min(int(w), max(1, width - x)))
    h = max(1, min(int(h), max(1, height - y)))
    return x, y, w, h


def _xywh_to_xyxy(x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
    return int(x), int(y), int(x + w), int(y + h)


def _iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return float(inter) / float(max(1, ua))


def _match_method_code(name: str) -> int:
    n = str(name or "").strip().upper()
    mapping = {
        "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
        "TM_CCOEFF": cv2.TM_CCOEFF,
        "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
        "TM_CCORR": cv2.TM_CCORR,
        "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
        "TM_SQDIFF": cv2.TM_SQDIFF,
    }
    return int(mapping.get(n, cv2.TM_CCOEFF_NORMED))


class TemplateTrackerServiceNode(ServiceNode):
    def __init__(self, *, node_id: str, node: Any, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=["tracking", "telemetry"],
            state_fields=[s.name for s in (getattr(node, "stateFields", None) or [])],
        )
        self._initial_state = dict(initial_state or {})
        self._task: asyncio.Task[object] | None = None

        self._active = True
        self._source_service_id: str = ""
        self._shm_name: str = ""
        self._tracker_kind: str = "csrt"
        self._match_method: str = "TM_CCOEFF_NORMED"
        self._match_threshold: float = 0.75
        self._search_margin_px: int = 200
        self._reacquire_interval_ms: int = 500

        self._telemetry = _Telemetry()
        self._last_error: str = ""

        self._shm: VideoShmReader | None = None
        self._shm_open_name: str = ""

        self._template_bgr: np.ndarray | None = None
        self._template_gray: np.ndarray | None = None
        self._template_b64: str = ""

        self._tracker: Any | None = None
        self._bbox_xywh: tuple[int, int, int, int] | None = None
        self._last_match_ts_ms: int = 0
        self._force_match = False

        self._last_processed_frame_id: int | None = None
        self._dup_skipped_since_last_processed: int = 0
        self._last_capture: _Capture | None = None

    def attach(self, bus: Any) -> None:
        super().attach(bus)
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._loop(), name=f"template_tracker:loop:{self.node_id}")
        except Exception:
            pass

    async def close(self) -> None:
        t = self._task
        self._task = None
        if t is not None:
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
        self._close_shm()

    async def on_lifecycle(self, active: bool, meta: dict[str, Any]) -> None:
        del meta
        self._active = bool(active)

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        del value, ts_ms
        name = str(field or "").strip()

        if name == "active":
            self._active = _coerce_bool(await self.get_state("active"), default=self._active)
        elif name == "sourceServiceId":
            self._source_service_id = _coerce_str(await self.get_state("sourceServiceId"), default=self._source_service_id)
            await self._maybe_reopen_shm()
        elif name == "shmName":
            self._shm_name = _coerce_str(await self.get_state("shmName"), default=self._shm_name)
            await self._maybe_reopen_shm()
        elif name == "trackerKind":
            v = _coerce_str(await self.get_state("trackerKind"), default=self._tracker_kind).lower()
            self._tracker_kind = v if v in ("none", "csrt", "kcf", "mosse") else "csrt"
            self._reset_tracker()
        elif name == "matchMethod":
            self._match_method = _coerce_str(await self.get_state("matchMethod"), default=self._match_method)
        elif name == "matchThreshold":
            self._match_threshold = _coerce_float(await self.get_state("matchThreshold"), default=self._match_threshold, minimum=0.0, maximum=1.0)
        elif name == "searchMarginPx":
            self._search_margin_px = _coerce_int(await self.get_state("searchMarginPx"), default=self._search_margin_px, minimum=0, maximum=100000)
        elif name == "reacquireIntervalMs":
            self._reacquire_interval_ms = _coerce_int(await self.get_state("reacquireIntervalMs"), default=self._reacquire_interval_ms, minimum=0, maximum=60000)
        elif name == "telemetryIntervalMs" or name == "telemetryWindowMs":
            self._telemetry.set_config(
                interval_ms=_coerce_int(
                    await self.get_state("telemetryIntervalMs"),
                    default=int(self._initial_state.get("telemetryIntervalMs") or 1000),
                    minimum=0,
                    maximum=60000,
                ),
                window_ms=_coerce_int(
                    await self.get_state("telemetryWindowMs"),
                    default=int(self._initial_state.get("telemetryWindowMs") or 2000),
                    minimum=100,
                    maximum=60000,
                ),
            )
        elif name == "templatePngB64":
            await self._load_template_from_state()

    async def on_command(self, name: str, args: dict[str, Any] | None = None, *, meta: dict[str, Any] | None = None) -> Any:
        del meta
        call = str(name or "").strip()
        a = dict(args or {})
        if call == "captureFrame":
            return await self._cmd_capture_frame(a)
        if call == "setTemplateFromCaptureRoi":
            return await self._cmd_set_template_from_capture_roi(a)
        if call == "matchNow":
            self._force_match = True
            return {"ok": True}
        if call == "clearTemplate":
            await self._clear_template()
            return {"ok": True}
        raise ValueError(f"unknown call: {call}")

    async def _ensure_config_loaded(self) -> None:
        self._active = _coerce_bool(await self.get_state("active"), default=bool(self._initial_state.get("active", True)))
        self._tracker_kind = _coerce_str(await self.get_state("trackerKind"), default=str(self._initial_state.get("trackerKind") or "csrt")).lower()
        self._match_method = _coerce_str(await self.get_state("matchMethod"), default=str(self._initial_state.get("matchMethod") or "TM_CCOEFF_NORMED"))
        self._match_threshold = _coerce_float(
            await self.get_state("matchThreshold"),
            default=float(self._initial_state.get("matchThreshold") or 0.75),
            minimum=0.0,
            maximum=1.0,
        )
        self._search_margin_px = _coerce_int(await self.get_state("searchMarginPx"), default=int(self._initial_state.get("searchMarginPx") or 200), minimum=0)
        self._reacquire_interval_ms = _coerce_int(
            await self.get_state("reacquireIntervalMs"),
            default=int(self._initial_state.get("reacquireIntervalMs") or 500),
            minimum=0,
            maximum=60000,
        )

        self._source_service_id = _coerce_str(await self.get_state("sourceServiceId"), default=str(self._initial_state.get("sourceServiceId") or ""))
        self._shm_name = _coerce_str(await self.get_state("shmName"), default=str(self._initial_state.get("shmName") or ""))

        self._telemetry.set_config(
            interval_ms=_coerce_int(await self.get_state("telemetryIntervalMs"), default=int(self._initial_state.get("telemetryIntervalMs") or 1000), minimum=0),
            window_ms=_coerce_int(await self.get_state("telemetryWindowMs"), default=int(self._initial_state.get("telemetryWindowMs") or 2000), minimum=100),
        )
        await self._load_template_from_state()

    async def _load_template_from_state(self) -> None:
        b64 = _coerce_str(await self.get_state("templatePngB64"), default=str(self._initial_state.get("templatePngB64") or ""))
        if b64 == self._template_b64:
            return
        self._template_b64 = b64
        self._template_bgr = _decode_png_b64_to_bgr(b64)
        if self._template_bgr is None:
            self._template_gray = None
            self._reset_tracker()
            return
        self._template_gray = cv2.cvtColor(self._template_bgr, cv2.COLOR_BGR2GRAY)
        self._reset_tracker()

    def _resolve_shm_name(self) -> str:
        shm = str(self._shm_name or "").strip()
        if shm:
            return shm
        sid = str(self._source_service_id or "").strip()
        if sid:
            return default_video_shm_name(sid)
        return ""

    async def _maybe_reopen_shm(self) -> None:
        want = self._resolve_shm_name()
        if want == self._shm_open_name:
            return
        self._close_shm()

    def _close_shm(self) -> None:
        if self._shm:
            try:
                self._shm.close()
            except Exception:
                pass
        self._shm = None
        self._shm_open_name = ""

    def _open_shm(self, shm_name: str) -> bool:
        self._close_shm()
        try:
            shm = VideoShmReader(shm_name)
            shm.open(use_event=True)
            self._shm = shm
            self._shm_open_name = shm_name
            return True
        except Exception:
            self._close_shm()
            return False

    def _reset_tracker(self) -> None:
        self._tracker = None
        self._bbox_xywh = None

    async def _clear_template(self) -> None:
        self._template_b64 = ""
        self._template_bgr = None
        self._template_gray = None
        self._reset_tracker()
        try:
            await self.set_state("templatePngB64", "")
        except Exception:
            pass

    def _read_shm_bgr(self) -> tuple[int, int, int, int, np.ndarray] | None:
        assert self._shm is not None
        header, payload = self._shm.read_latest_bgra()
        if header is None or payload is None:
            return None
        frame_id = int(header.frame_id)
        ts_ms = int(header.ts_ms)
        width, height, pitch = int(header.width), int(header.height), int(header.pitch)
        if width <= 0 or height <= 0 or pitch <= 0:
            return None
        frame_bytes = int(pitch) * int(height)
        if len(payload) < frame_bytes:
            return None
        buf = np.frombuffer(payload, dtype=np.uint8)
        rows = buf.reshape((height, pitch))
        bgra = rows[:, : width * 4].reshape((height, width, 4))
        bgr = np.ascontiguousarray(bgra[:, :, 0:3])
        return frame_id, ts_ms, width, height, bgr

    async def _cmd_capture_frame(self, args: dict[str, Any]) -> dict[str, Any]:
        await self._ensure_config_loaded()
        shm_name = self._resolve_shm_name()
        if not shm_name:
            raise ValueError("missing shmName/sourceServiceId")
        if self._shm is None:
            if not self._open_shm(shm_name):
                raise RuntimeError(f"failed to open shm: {shm_name}")

        fmt = _coerce_str(args.get("format"), default="jpg").lower()
        quality = _coerce_int(args.get("quality"), default=85, minimum=1, maximum=100)
        max_bytes = _coerce_int(args.get("maxBytes"), default=900000, minimum=10000, maximum=5000000)
        max_w = _coerce_int(args.get("maxWidth"), default=1280, minimum=0, maximum=10000)
        max_h = _coerce_int(args.get("maxHeight"), default=720, minimum=0, maximum=10000)

        got = self._read_shm_bgr()
        if got is None:
            raise RuntimeError("no frame available")
        frame_id, ts_ms, width, height, bgr = got
        self._last_capture = _Capture(frame_id=frame_id, ts_ms=ts_ms, bgr=bgr)

        b64, meta = _encode_image_b64(bgr, fmt=fmt, quality=quality, max_bytes=max_bytes, max_width=max_w, max_height=max_h)
        return {
            "frameId": int(frame_id),
            "tsMs": int(ts_ms),
            "source": {"width": int(width), "height": int(height), "shmName": str(self._shm_open_name or shm_name)},
            "image": {"b64": str(b64), **meta},
        }

    async def _cmd_set_template_from_capture_roi(self, args: dict[str, Any]) -> dict[str, Any]:
        cap = self._last_capture
        if cap is None:
            raise ValueError("no capture available; call captureFrame first")
        cap_id = _coerce_int(args.get("captureFrameId"), default=-1)
        if int(cap_id) != int(cap.frame_id):
            raise ValueError("captureFrameId mismatch; call captureFrame again")
        x1 = _coerce_int(args.get("x1"), default=0, minimum=0)
        y1 = _coerce_int(args.get("y1"), default=0, minimum=0)
        x2 = _coerce_int(args.get("x2"), default=0, minimum=0)
        y2 = _coerce_int(args.get("y2"), default=0, minimum=0)
        x, y, w, h = _xyxy_to_xywh(x1, y1, x2, y2)
        hh, ww = int(cap.bgr.shape[0]), int(cap.bgr.shape[1])
        x, y, w, h = _clip_xywh(x, y, w, h, width=ww, height=hh)
        crop = np.ascontiguousarray(cap.bgr[y : y + h, x : x + w])

        b64, meta = _encode_image_b64(crop, fmt="png", quality=100, max_bytes=900000, max_width=0, max_height=0)
        await self.set_state("templatePngB64", b64)
        self._template_b64 = b64
        self._template_bgr = crop
        self._template_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        self._reset_tracker()
        return {"ok": True, "template": meta}

    def _match_template(
        self, frame_gray: np.ndarray
    ) -> tuple[tuple[int, int, int, int] | None, float, tuple[int, int, int, int] | None]:
        tpl = self._template_gray
        if tpl is None:
            return None, 0.0, None
        th, tw = int(tpl.shape[0]), int(tpl.shape[1])
        fh, fw = int(frame_gray.shape[0]), int(frame_gray.shape[1])
        if tw <= 1 or th <= 1 or tw > fw or th > fh:
            return None, 0.0, None

        roi = None
        if self._bbox_xywh is not None and self._search_margin_px > 0:
            x, y, w, h = self._bbox_xywh
            m = int(self._search_margin_px)
            rx = max(0, x - m)
            ry = max(0, y - m)
            rx2 = min(fw, x + w + m)
            ry2 = min(fh, y + h + m)
            if (rx2 - rx) >= tw and (ry2 - ry) >= th:
                roi = (rx, ry, rx2 - rx, ry2 - ry)

        if roi is not None:
            rx, ry, rw, rh = roi
            search = frame_gray[ry : ry + rh, rx : rx + rw]
            off_x, off_y = rx, ry
        else:
            search = frame_gray
            off_x, off_y = 0, 0

        method = _match_method_code(self._match_method)
        res = cv2.matchTemplate(search, tpl, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            score = float(1.0 - min_val)
            loc = min_loc
        else:
            score = float(max_val)
            loc = max_loc

        x = int(loc[0] + off_x)
        y = int(loc[1] + off_y)
        bbox = (x, y, tw, th)
        return bbox, score, roi

    def _ensure_tracker_for_bbox(self, bgr: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> bool:
        self._tracker = _create_tracker(self._tracker_kind)
        if self._tracker is None:
            return False
        x, y, w, h = bbox_xywh
        # OpenCV 4.13 tracker.init is strict about integer box types.
        try:
            res = self._tracker.init(bgr, (int(x), int(y), int(w), int(h)))
            ok = True if res is None else bool(res)
        except Exception as exc:
            raise RuntimeError(f"tracker.init failed kind={self._tracker_kind} bbox_xywh={bbox_xywh}: {exc}") from exc
        if ok:
            self._bbox_xywh = bbox_xywh
        return ok

    def _update_tracker(self, bgr: np.ndarray) -> tuple[bool, tuple[int, int, int, int] | None]:
        if self._tracker is None or self._bbox_xywh is None:
            return False, None
        ok, box = self._tracker.update(bgr)
        if not ok:
            return False, None
        x, y, w, h = [int(round(v)) for v in box]
        x, y, w, h = _clip_xywh(x, y, w, h, width=int(bgr.shape[1]), height=int(bgr.shape[0]))
        self._bbox_xywh = (x, y, w, h)
        return True, self._bbox_xywh

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(0)
                await self._ensure_config_loaded()

                if not self._active:
                    await asyncio.sleep(0.05)
                    continue

                shm_name = self._resolve_shm_name()
                if not shm_name:
                    await asyncio.sleep(0.05)
                    continue
                if self._shm is None:
                    if not self._open_shm(shm_name):
                        await asyncio.sleep(0.1)
                        continue

                assert self._shm is not None
                t0 = time.perf_counter()
                self._shm.wait_new_frame(timeout_ms=10)
                t_wait = time.perf_counter()
                got = self._read_shm_bgr()
                t_read = time.perf_counter()
                if got is None:
                    continue
                frame_id, ts_ms, width, height, bgr = got

                if self._last_processed_frame_id is not None and int(frame_id) == int(self._last_processed_frame_id):
                    self._dup_skipped_since_last_processed += 1
                    continue
                dup_skipped = int(self._dup_skipped_since_last_processed)
                self._dup_skipped_since_last_processed = 0
                self._last_processed_frame_id = int(frame_id)

                frame_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                t_pre = time.perf_counter()

                did_match = False
                track_ok = False
                status = "idle"
                match_score = 0.0
                match_roi = None
                match_bbox = None
                match_ms = 0.0
                track_ms = 0.0

                if self._template_gray is None:
                    self._reset_tracker()
                    status = "no_template"
                else:
                    do_match = bool(self._force_match)
                    self._force_match = False
                    if self._bbox_xywh is None or self._tracker is None:
                        do_match = True
                    if self._reacquire_interval_ms > 0 and (ts_ms - self._last_match_ts_ms) >= int(self._reacquire_interval_ms):
                        do_match = True

                    if do_match:
                        t_match0 = time.perf_counter()
                        match_bbox, match_score, match_roi = self._match_template(frame_gray)
                        t_match1 = time.perf_counter()
                        match_ms = (t_match1 - t_match0) * 1000.0
                        did_match = True
                        self._last_match_ts_ms = int(ts_ms)
                        if match_bbox is not None and float(match_score) >= float(self._match_threshold):
                            if self._bbox_xywh is None or _iou_xywh(self._bbox_xywh, match_bbox) < 0.9:
                                self._reset_tracker()
                                self._ensure_tracker_for_bbox(bgr, match_bbox)
                        else:
                            if self._bbox_xywh is None:
                                self._reset_tracker()

                    t_track0 = time.perf_counter()
                    if self._tracker is not None and self._bbox_xywh is not None:
                        track_ok, _bbox_xywh = self._update_tracker(bgr)
                        if not track_ok:
                            self._reset_tracker()
                    t_track1 = time.perf_counter()
                    track_ms = (t_track1 - t_track0) * 1000.0

                    if self._bbox_xywh is not None and track_ok:
                        status = "tracking"
                    elif match_bbox is not None and float(match_score) >= float(self._match_threshold):
                        status = "matched"
                    else:
                        status = "lost"

                out_bbox_xyxy = None
                if self._bbox_xywh is not None:
                    x, y, w, h = self._bbox_xywh
                    out_bbox_xyxy = list(_xywh_to_xyxy(x, y, w, h))

                payload_out: dict[str, Any] = {
                    "frameId": int(frame_id),
                    "tsMs": int(ts_ms),
                    "width": int(width),
                    "height": int(height),
                    "status": str(status),
                    "bbox": out_bbox_xyxy,
                    "match": {
                        "score": float(match_score),
                        "threshold": float(self._match_threshold),
                        "method": str(self._match_method),
                        "roi": (list(match_roi) if match_roi is not None else None),
                        "bbox": (list(_xywh_to_xyxy(*match_bbox)) if match_bbox is not None else None),
                    },
                    "tracker": {"kind": str(self._tracker_kind), "ok": bool(track_ok)},
                }

                t_emit0 = time.perf_counter()
                await self.emit("tracking", payload_out, ts_ms=int(ts_ms))
                t_emit1 = time.perf_counter()

                t_end = time.perf_counter()
                self._telemetry.observe(
                    ts_ms=int(ts_ms),
                    dup_skipped=dup_skipped,
                    did_match=did_match,
                    track_ok=track_ok,
                    total_ms=(t_end - t0) * 1000.0,
                    shm_wait_ms=(t_wait - t0) * 1000.0,
                    shm_read_ms=(t_read - t_wait) * 1000.0,
                    pre_ms=(t_pre - t_read) * 1000.0,
                    match_ms=float(match_ms),
                    track_ms=float(track_ms),
                    emit_ms=(t_emit1 - t_emit0) * 1000.0,
                )

                if self._telemetry.should_emit(int(ts_ms)):
                    shm_has_event = False
                    try:
                        shm_has_event = bool(getattr(self._shm, "_event", None) is not None)
                    except Exception:
                        shm_has_event = False
                    tel = self._telemetry.summary(
                        now_ms=int(ts_ms),
                        node_id=self.node_id,
                        tracker_kind=self._tracker_kind,
                        match_method=self._match_method,
                        shm_name=str(self._shm_open_name or shm_name),
                        shm_has_event=shm_has_event,
                        shm_wait_timeout_ms=10,
                        frame_id_last_seen=int(frame_id),
                        frame_id_last_processed=self._last_processed_frame_id,
                    )
                    await self.emit("telemetry", tel, ts_ms=int(ts_ms))
                    self._telemetry.mark_emitted(int(ts_ms))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = str(exc)
                try:
                    await self.set_state("lastError", self._last_error)
                except Exception:
                    pass
                await asyncio.sleep(0.1)
