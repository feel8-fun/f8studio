from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .onnx_detectors import Detection, PoseKeypoint
from .vision_utils import clamp_xyxy, iou_xyxy, xywh_to_xyxy, xyxy_to_xywh


TrackerKind = Literal["none", "csrt", "kcf", "mosse"]


def _create_tracker(kind: TrackerKind) -> Any | None:
    if kind == "none":
        return None
    import cv2  # type: ignore

    def _try(fn_name: str) -> Any | None:
        fn = getattr(cv2, fn_name, None)
        if callable(fn):
            return fn()
        legacy = getattr(cv2, "legacy", None)
        fn2 = getattr(legacy, fn_name, None) if legacy is not None else None
        if callable(fn2):
            return fn2()
        return None

    if kind == "csrt":
        trk = _try("TrackerCSRT_create")
    elif kind == "kcf":
        trk = _try("TrackerKCF_create")
    elif kind == "mosse":
        trk = _try("TrackerMOSSE_create")
    else:
        trk = None
    return trk


@dataclass
class Track:
    track_id: int
    cls: str
    xyxy: tuple[int, int, int, int]
    conf: float
    age: int = 0
    mismatch: int = 0
    tracker_kind: TrackerKind = "none"
    _tracker: Any | None = field(default=None, repr=False)
    keypoints: list[PoseKeypoint] | None = None
    obb: list[tuple[float, float]] | None = None

    def init_tracker(self, frame_bgr: Any) -> None:
        trk = _create_tracker(self.tracker_kind)
        self._tracker = trk
        if trk is None:
            return
        trk.init(frame_bgr, xyxy_to_xywh(self.xyxy))

    def update_tracker(self, frame_bgr: Any, *, frame_size: tuple[int, int]) -> bool:
        if self._tracker is None:
            return False
        ok, box = self._tracker.update(frame_bgr)
        if not ok:
            return False
        try:
            x, y, w, h = box
        except Exception:
            return False
        self.xyxy = clamp_xyxy(xywh_to_xyxy(float(x), float(y), float(w), float(h)), size=frame_size)
        return True


def associate_and_update_tracks(
    *,
    tracks: list[Track],
    dets: list[Detection],
    frame_bgr: Any,
    frame_size: tuple[int, int],
    tracker_kind: TrackerKind,
    iou_match: float,
    mismatch_iou: float,
    mismatch_patience: int,
    max_age: int,
    max_targets: int,
    reinit_on_detect: bool,
    next_id: int,
) -> tuple[list[Track], int]:
    """
    Detection-step association:
    - match detections to existing tracks by IoU + class
    - optionally reinit the cv tracker on match
    - create new tracks up to max_targets
    """
    candidates = [d for d in dets]
    used: set[int] = set()

    for trk in tracks:
        best_i = -1
        best_iou = 0.0
        for i_det, d in enumerate(candidates):
            if i_det in used:
                continue
            if d.cls != trk.cls:
                continue
            v = iou_xyxy(trk.xyxy, d.xyxy)
            if v > best_iou:
                best_iou = v
                best_i = i_det

        if best_i >= 0 and best_iou >= float(iou_match):
            used.add(best_i)
            d = candidates[best_i]
            trk.xyxy = clamp_xyxy(d.xyxy, size=frame_size)
            trk.conf = float(d.conf)
            trk.keypoints = d.keypoints
            trk.obb = d.obb
            trk.age = 0
            trk.mismatch = trk.mismatch + 1 if best_iou < float(mismatch_iou) else 0
            trk.tracker_kind = tracker_kind
            if reinit_on_detect:
                trk.init_tracker(frame_bgr)
        else:
            trk.mismatch += 1
            trk.age += 1

    # Drop inconsistent/old tracks.
    tracks = [t for t in tracks if t.age <= int(max_age) and t.mismatch < int(mismatch_patience)]

    # Acquire new tracks.
    for i_det, d in enumerate(candidates):
        if len(tracks) >= int(max_targets):
            break
        if i_det in used:
            continue
        t = Track(
            track_id=next_id,
            cls=d.cls,
            xyxy=clamp_xyxy(d.xyxy, size=frame_size),
            conf=float(d.conf),
            tracker_kind=tracker_kind,
            keypoints=d.keypoints,
            obb=d.obb,
        )
        next_id += 1
        t.init_tracker(frame_bgr)
        tracks.append(t)

    return tracks, next_id


def update_tracks_with_cv(
    *,
    tracks: list[Track],
    frame_bgr: Any,
    frame_size: tuple[int, int],
    mismatch_patience: int,
    max_age: int,
) -> list[Track]:
    """
    Non-detection step: update tracks using their per-track cv tracker (if available).
    """
    out: list[Track] = []
    for trk in tracks:
        # If no CV tracker is available (e.g. trackerKind=none or OpenCV build missing trackers),
        # keep the last detection for a short time instead of immediately dropping to empty.
        if trk._tracker is None:
            trk.age += 1
            if trk.age <= int(max_age):
                out.append(trk)
            continue
        ok = trk.update_tracker(frame_bgr, frame_size=frame_size)
        if ok:
            trk.age = 0
        else:
            trk.age += 1
            trk.mismatch += 1
        if trk.age <= int(max_age) and trk.mismatch < int(mismatch_patience):
            out.append(trk)
    return out
