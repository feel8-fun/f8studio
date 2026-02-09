from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class Det:
    cls: str
    conf: float
    xyxy: tuple[int, int, int, int]


@dataclass
class Track:
    track_id: int
    cls: str
    xyxy: tuple[int, int, int, int]
    conf: float
    age: int = 0
    mismatch: int = 0
    _tracker: Any | None = None


@dataclass(frozen=True)
class FrameOut:
    frame_idx: int
    time_sec: float | None
    tracks: list[dict[str, Any]]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = a_area + b_area - inter
    return float(inter) / float(denom) if denom > 0 else 0.0


def _center_dist2(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0
    dx = acx - bcx
    dy = acy - bcy
    return dx * dx + dy * dy


def _area(a: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = a
    return max(0, x2 - x1) * max(0, y2 - y1)


def _load_model_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise SystemExit("Missing dependency 'pyyaml'.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid config yaml: {path}")
    return data


def _xyxy_to_xywh(xyxy: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    # OpenCV Tracker APIs in opencv-contrib-python expect integer bounding boxes.
    return int(x1), int(y1), int(max(1, x2 - x1)), int(max(1, y2 - y1))


def _xywh_to_xyxy(x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    return x1, y1, x2, y2


def _clamp_xyxy(xyxy: tuple[int, int, int, int], *, size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = size
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _create_tracker_api(cv2: Any, factory_func: str) -> Any:
    """Try to create tracker from standard API, fallback to legacy API."""
    # Try standard API first
    if hasattr(cv2, factory_func):
        try:
            factory = getattr(cv2, factory_func)
            return factory()
        except (AttributeError, TypeError) as exc:
            pass  # Fall through to legacy API
    
    # Try legacy API
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, factory_func):
        try:
            factory = getattr(cv2.legacy, factory_func)
            return factory()
        except (AttributeError, TypeError) as exc:
            pass
    
    return None


def _create_csrt_tracker() -> Any:
    import cv2  # type: ignore

    trk = _create_tracker_api(cv2, "TrackerCSRT_create")
    if trk is not None:
        return trk
    raise SystemExit(
        "OpenCV CSRT tracker not available. Install opencv-contrib-python (or opencv-contrib-python-headless)."
    )


def _create_tracker(kind: Literal["csrt", "kcf", "mosse"]) -> Any:
    import cv2  # type: ignore

    if kind == "csrt":
        trk = _create_tracker_api(cv2, "TrackerCSRT_create")
    elif kind == "kcf":
        trk = _create_tracker_api(cv2, "TrackerKCF_create")
    elif kind == "mosse":
        trk = _create_tracker_api(cv2, "TrackerMOSSE_create")
    else:
        raise SystemExit(f"Unsupported tracker: {kind}")

    if trk is None:
        raise SystemExit(
            f"OpenCV tracker '{kind}' not available. Install opencv-contrib-python (or opencv-contrib-python-headless)."
        )
    return trk


def _open_video(path: Path) -> tuple[Any, dict[str, Any]]:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    return cap, {"fps": fps, "frame_count": frame_count, "size": (w, h)}


def _create_writer(output: Path, *, fps: float, size: tuple[int, int]) -> Any:
    import cv2  # type: ignore

    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if output.suffix.lower() == ".mp4" else "XVID"))
    writer = cv2.VideoWriter(str(output), fourcc, float(fps), (int(size[0]), int(size[1])))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open video writer: {output}")
    return writer


def _nms_xyxy(boxes: Any, scores: Any, *, iou_thr: float) -> list[int]:
    import numpy as np  # type: ignore

    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32, copy=False)
    scores = scores.astype(np.float32, copy=False)
    order = scores.argsort()[::-1]

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])

        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        denom = areas[i] + areas[rest] - inter
        iou = np.where(denom > 0, inter / denom, 0.0)
        order = rest[iou <= float(iou_thr)]

    return keep


def _letterbox_bgr(
    img_bgr: Any,
    *,
    new_shape: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[Any, float, tuple[int, int]]:
    import cv2  # type: ignore

    h0, w0 = img_bgr.shape[:2]
    new_w, new_h = int(new_shape[0]), int(new_shape[1])
    r = min(new_w / float(w0), new_h / float(h0)) if w0 and h0 else 1.0
    if r <= 0:
        r = 1.0
    w1 = int(round(w0 * r))
    h1 = int(round(h0 * r))
    img = cv2.resize(img_bgr, (w1, h1), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - w1
    pad_h = new_h - h1
    left = int(pad_w // 2)
    right = int(pad_w - left)
    top = int(pad_h // 2)
    bottom = int(pad_h - top)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, float(r), (left, top)


class _Detector:
    def detect(
        self,
        frame_bgr: Any,
        *,
        conf: float,
        iou: float,
        imgsz: int | None,
        device: str | None,
    ) -> list[Det]:
        raise NotImplementedError


class _UltralyticsDetector(_Detector):
    def __init__(self, model_path: Path, *, class_names_fallback: list[str] | None) -> None:
        try:
            # Prevent Ultralytics from auto-installing pip packages at runtime (which can
            # accidentally install CPU onnxruntime and disable CUDAExecutionProvider).
            os.environ.setdefault("YOLO_AUTOINSTALL", "false")
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise SystemExit("Missing dependency 'ultralytics'.") from exc

        try:
            self._model = YOLO(str(model_path), task="detect")
        except TypeError:
            self._model = YOLO(str(model_path))
        self._names = _get_class_names(self._model, class_names_fallback)

    def detect(
        self,
        frame_bgr: Any,
        *,
        conf: float,
        iou: float,
        imgsz: int | None,
        device: str | None,
    ) -> list[Det]:
        kwargs: dict[str, Any] = {"conf": conf, "iou": iou, "verbose": False}
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        if device:
            kwargs["device"] = device
        result = self._model(frame_bgr, **kwargs)[0]

        try:
            boxes = result.boxes
        except Exception:
            boxes = None
        if boxes is None:
            return []

        import numpy as np  # type: ignore

        xyxy = boxes.xyxy.cpu().numpy().astype(np.int32)
        confs = boxes.conf.cpu().numpy().astype(float)
        clss = boxes.cls.cpu().numpy().astype(int)

        out: list[Det] = []
        for (x1, y1, x2, y2), c, cls_idx in zip(xyxy, confs, clss, strict=False):
            cls_name = self._names.get(int(cls_idx), str(int(cls_idx)))
            out.append(Det(cls=cls_name, conf=float(c), xyxy=(int(x1), int(y1), int(x2), int(y2))))
        return out


class _OnnxYoloDetector(_Detector):
    def __init__(
        self,
        model_path: Path,
        *,
        input_width: int,
        input_height: int,
        class_names_fallback: list[str] | None,
        ort_provider: Literal["auto", "cuda", "cpu"] = "auto",
    ) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            raise SystemExit("Missing dependency 'onnxruntime-gpu' (or 'onnxruntime').") from exc

        available = ort.get_available_providers()
        if ort_provider == "cpu" or "CUDAExecutionProvider" not in available:
            providers = ["CPUExecutionProvider"]
        elif ort_provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])

        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._providers = self._session.get_providers()
        self._input_w = int(input_width)
        self._input_h = int(input_height)
        self._names = {int(i): str(v) for i, v in enumerate(class_names_fallback or [])}
        self._nc = len(self._names) if self._names else None

        print(f"ONNX Runtime providers: {available} (active: {self._providers})")

    def detect(
        self,
        frame_bgr: Any,
        *,
        conf: float,
        iou: float,
        imgsz: int | None,
        device: str | None,
    ) -> list[Det]:
        del device  # not used for ONNX Runtime
        del imgsz  # input size is defined by the exported ONNX model / yaml config

        import cv2  # type: ignore
        import numpy as np  # type: ignore

        lb, r, (pad_x, pad_y) = _letterbox_bgr(frame_bgr, new_shape=(self._input_w, self._input_h))
        img_rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3xHxW

        out0 = self._session.run(None, {self._input_name: x})[0]
        pred = np.asarray(out0)
        if pred.ndim != 3:
            return []

        # Accept both (1, C, N) and (1, N, C)
        if pred.shape[1] < pred.shape[2]:
            pred = np.transpose(pred, (0, 2, 1))
        pred = pred[0]  # (N, C)

        c = int(pred.shape[1])
        nc = self._nc
        if nc is None:
            # Try to infer: (x,y,w,h)+(cls...) or (x,y,w,h,obj)+(cls...)
            nc = max(1, c - 4)
        has_obj = (c == 5 + nc)

        if has_obj:
            xywh = pred[:, 0:4]
            obj = pred[:, 4]
            cls_scores = pred[:, 5 : 5 + nc]
            cls_idx = cls_scores.argmax(axis=1)
            cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_idx]
            scores = obj * cls_conf
        else:
            xywh = pred[:, 0:4]
            cls_scores = pred[:, 4 : 4 + nc]
            cls_idx = cls_scores.argmax(axis=1)
            scores = cls_scores[np.arange(cls_scores.shape[0]), cls_idx]

        keep0 = scores >= float(conf)
        if not np.any(keep0):
            return []

        xywh = xywh[keep0]
        scores = scores[keep0]
        cls_idx = cls_idx[keep0]

        # xywh (center) -> xyxy in letterboxed input coords
        x_c = xywh[:, 0]
        y_c = xywh[:, 1]
        w = xywh[:, 2]
        h = xywh[:, 3]
        x1 = x_c - w / 2.0
        y1 = y_c - h / 2.0
        x2 = x_c + w / 2.0
        y2 = y_c + h / 2.0
        boxes_lb = np.stack([x1, y1, x2, y2], axis=1)

        keep_idx = _nms_xyxy(boxes_lb, scores, iou_thr=float(iou))
        if not keep_idx:
            return []

        boxes_lb = boxes_lb[keep_idx]
        scores = scores[keep_idx]
        cls_idx = cls_idx[keep_idx]

        # Map boxes back to original frame coords
        boxes = boxes_lb.copy()
        boxes[:, [0, 2]] -= float(pad_x)
        boxes[:, [1, 3]] -= float(pad_y)
        boxes /= float(r) if r > 0 else 1.0

        h0, w0 = frame_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, float(w0))
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, float(h0))

        out: list[Det] = []
        for (x1f, y1f, x2f, y2f), sc, ci in zip(boxes, scores, cls_idx, strict=False):
            x1i, y1i, x2i, y2i = int(round(float(x1f))), int(round(float(y1f))), int(round(float(x2f))), int(round(float(y2f)))
            if x2i <= x1i or y2i <= y1i:
                continue
            name = self._names.get(int(ci), str(int(ci)))
            out.append(Det(cls=name, conf=float(sc), xyxy=(x1i, y1i, x2i, y2i)))
        return out


def _load_detector(
    model_path: Path,
    *,
    config: dict[str, Any] | None,
    class_names_fallback: list[str] | None,
    ort_provider: Literal["auto", "cuda", "cpu"] = "auto",
) -> _Detector:
    if model_path.suffix.lower() == ".onnx":
        input_w = int(config.get("input_width", 0)) if config else 0
        input_h = int(config.get("input_height", 0)) if config else 0
        if input_w <= 0 or input_h <= 0:
            raise SystemExit("ONNX model requires 'input_width' and 'input_height' in --config yaml.")
        return _OnnxYoloDetector(
            model_path,
            input_width=input_w,
            input_height=input_h,
            class_names_fallback=class_names_fallback,
            ort_provider=ort_provider,
        )
    return _UltralyticsDetector(model_path, class_names_fallback=class_names_fallback)


def _get_class_names(model: Any, fallback: list[str] | None) -> dict[int, str]:
    try:
        names = model.names
    except Exception:
        names = None
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {int(i): str(v) for i, v in enumerate(names)}
    if fallback:
        return {int(i): str(v) for i, v in enumerate(fallback)}
    return {}


def _detect_frame(
    detector: _Detector,
    frame_bgr: Any,
    *,
    conf: float,
    iou: float,
    imgsz: int | None,
    device: str | None,
    class_names_fallback: list[str] | None,
) -> list[Det]:
    del class_names_fallback  # handled inside detector
    return detector.detect(frame_bgr, conf=conf, iou=iou, imgsz=imgsz, device=device)


def _select_candidates(
    dets: list[Det],
    *,
    target_classes: set[str] | None,
    min_conf: float,
) -> list[Det]:
    filtered = [d for d in dets if d.conf >= min_conf]
    if target_classes:
        filtered = [d for d in filtered if d.cls in target_classes]
    return filtered


def _rank_for_selection(
    det: Det,
    *,
    rule: Literal["conf", "area", "center"],
    frame_size: tuple[int, int],
) -> float:
    if rule == "conf":
        return det.conf
    if rule == "area":
        return float(_area(det.xyxy))
    w, h = frame_size
    cx = w / 2.0
    cy = h / 2.0
    center_box = (int(cx), int(cy), int(cx), int(cy))
    dist = _center_dist2(det.xyxy, center_box)
    return -(dist) + det.conf * 1e6


def _associate_tracks(
    tracks: list[Track],
    dets: list[Det],
    *,
    iou_match: float,
    max_age: int,
) -> tuple[list[Track], list[Det]]:
    remaining = dets[:]
    for trk in tracks:
        best_iou = 0.0
        best_idx = -1
        for i, d in enumerate(remaining):
            if d.cls != trk.cls:
                continue
            v = _iou(trk.xyxy, d.xyxy)
            if v > best_iou:
                best_iou = v
                best_idx = i
        if best_idx >= 0 and best_iou >= iou_match:
            d = remaining.pop(best_idx)
            trk.xyxy = d.xyxy
            trk.conf = d.conf
            trk.age = 0
        else:
            trk.age += 1

    tracks = [t for t in tracks if t.age <= max_age]
    return tracks, remaining


def _acquire_new_tracks(
    tracks: list[Track],
    dets: list[Det],
    *,
    max_targets: int,
    select_rule: Literal["conf", "area", "center"],
    frame_size: tuple[int, int],
    next_id: int,
) -> tuple[list[Track], int]:
    if len(tracks) >= max_targets:
        return tracks, next_id
    ranked = sorted(
        dets,
        key=lambda d: _rank_for_selection(d, rule=select_rule, frame_size=frame_size),
        reverse=True,
    )
    for d in ranked:
        if len(tracks) >= max_targets:
            break
        tracks.append(Track(track_id=next_id, cls=d.cls, xyxy=d.xyxy, conf=d.conf, age=0))
        next_id += 1
    return tracks, next_id


def run_video(args: argparse.Namespace) -> int:

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    config: dict[str, Any] | None = None
    class_names_fallback: list[str] | None = None
    if args.config:
        config = _load_model_config(Path(args.config))
        model_path = Path(config.get("model_path", ""))
        if not model_path.is_absolute():
            model_path = Path(args.config).resolve().parent / model_path
        class_names_fallback = config.get("classes") if isinstance(config.get("classes"), list) else None
        conf = float(args.conf) if args.conf is not None else float(config.get("conf_threshold", 0.25))
        iou = float(args.iou) if args.iou is not None else float(config.get("iou_threshold", 0.45))
    else:
        model_path = Path(args.model)
        conf = float(args.conf) if args.conf is not None else 0.25
        iou = float(args.iou) if args.iou is not None else 0.45

    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    cap, meta = _open_video(video_path)
    fps = float(meta["fps"])
    src_size = meta["size"]
    src_w, src_h = src_size

    import cv2  # type: ignore

    max_side = int(args.max_side) if args.max_side is not None else 0
    if max_side > 0 and src_w > 0 and src_h > 0 and max(src_w, src_h) > max_side:
        scale = float(max_side) / float(max(src_w, src_h))
    else:
        scale = 1.0

    proc_w = int(round(src_w * scale)) if src_w else 0
    proc_h = int(round(src_h * scale)) if src_h else 0
    if proc_w <= 0 or proc_h <= 0:
        # fallback to first frame
        ok0, frame0 = cap.read()
        if not ok0 or frame0 is None:
            raise SystemExit("Failed to read first frame.")
        src_h, src_w = frame0.shape[:2]
        if max_side > 0 and max(src_w, src_h) > max_side:
            scale = float(max_side) / float(max(src_w, src_h))
        else:
            scale = 1.0
        proc_w = int(round(src_w * scale))
        proc_h = int(round(src_h * scale))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_size = (proc_w, proc_h)

    jsonl_file = None
    if args.jsonl:
        jsonl_path = Path(args.jsonl)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_path.open("w", encoding="utf-8", newline="\n")

    target_classes = set(args.target_classes) if args.target_classes else None
    detect_stride = max(1, int(args.detect_stride))
    acquire_hz = float(args.acquire_hz)
    track_detect_hz = float(args.track_detect_hz)
    mismatch_iou = float(args.mismatch_iou)
    mismatch_patience = max(1, int(args.mismatch_patience))
    reinit_on_match = bool(args.reinit_on_match)

    detector = _load_detector(
        model_path,
        config=config,
        class_names_fallback=class_names_fallback,
        ort_provider=args.ort_provider,
    )
    tracks: list[Track] = []
    next_id = 1

    start = time.time()
    frame_idx = 0
    last_detect_time: float | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if scale != 1.0:
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)

            time_sec = None
            try:
                time_sec = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            except Exception:
                pass
            if time_sec is None:
                time_sec = frame_idx / fps if fps > 0 else None

            # Update trackers every frame (when active)
            for trk in tracks:
                if trk._tracker is None:
                    continue
                ok_t, bbox = trk._tracker.update(frame)
                if ok_t:
                    trk.xyxy = _clamp_xyxy(_xywh_to_xyxy(*bbox), size=frame_size)
                    trk.age += 1
                else:
                    trk.age += 1
                    trk.mismatch += 1

            dets: list[Det] = []
            # Determine detection rate based on state:
            # - no tracks: acquire_hz
            # - tracking: track_detect_hz
            do_detect = False
            if frame_idx % detect_stride == 0:
                interval = (1.0 / acquire_hz) if not tracks else (1.0 / track_detect_hz)
                if last_detect_time is None or time_sec is None:
                    do_detect = True
                else:
                    do_detect = (time_sec - last_detect_time) >= max(0.0, interval)

            if do_detect:
                if time_sec is not None:
                    last_detect_time = time_sec
                dets = _detect_frame(
                    detector,
                    frame,
                    conf=conf,
                    iou=iou,
                    imgsz=args.imgsz,
                    device=args.device,
                    class_names_fallback=class_names_fallback,
                )

                candidates = _select_candidates(dets, target_classes=target_classes, min_conf=conf)

                # Validate / correct existing tracks with detections
                used: set[int] = set()
                for trk in tracks:
                    best_i = -1
                    best_iou = 0.0
                    for i_det, d in enumerate(candidates):
                        if i_det in used:
                            continue
                        if d.cls != trk.cls:
                            continue
                        v = _iou(trk.xyxy, d.xyxy)
                        if v > best_iou:
                            best_iou = v
                            best_i = i_det

                    if best_i >= 0 and best_iou >= float(args.iou_match):
                        used.add(best_i)
                        d = candidates[best_i]
                        if best_iou < mismatch_iou:
                            trk.mismatch += 1
                        else:
                            trk.mismatch = 0

                        if reinit_on_match:
                            trk.xyxy = _clamp_xyxy(d.xyxy, size=frame_size)
                            trk.conf = d.conf
                            trk._tracker = _create_tracker(args.tracker)
                            trk._tracker.init(frame, _xyxy_to_xywh(trk.xyxy))
                    else:
                        trk.mismatch += 1

                # Drop tracks that are consistently inconsistent or too old
                max_age = int(args.max_age)
                tracks = [t for t in tracks if t.age <= max_age and t.mismatch < mismatch_patience]

                # Acquire new tracks if needed
                remaining = [d for i_det, d in enumerate(candidates) if i_det not in used]
                tracks, next_id = _acquire_new_tracks(
                    tracks,
                    remaining,
                    max_targets=int(args.max_targets),
                    select_rule=args.select,
                    frame_size=frame_size,
                    next_id=next_id,
                )
                # Initialize trackers for newly acquired tracks
                for trk in tracks:
                    if trk._tracker is None:
                        trk.xyxy = _clamp_xyxy(trk.xyxy, size=frame_size)
                        trk._tracker = _create_tracker(args.tracker)
                        trk._tracker.init(frame, _xyxy_to_xywh(trk.xyxy))
            else:
                # No detection this frame: trackers already updated above
                tracks = [t for t in tracks if t.age <= int(args.max_age) and t.mismatch < mismatch_patience]

            annotated = frame.copy()
            for trk in tracks:
                x1, y1, x2, y2 = trk.xyxy
                color = (0, 255, 255) if trk.mismatch == 0 else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    f"id={trk.track_id} {trk.cls} {trk.conf:.2f} age={trk.age} mis={trk.mismatch}",
                    (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            if jsonl_file is not None:
                jsonl_file.write(
                    json.dumps(
                        asdict(
                            FrameOut(
                                frame_idx=frame_idx,
                                time_sec=time_sec,
                                tracks=[asdict(t) for t in tracks],
                            )
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            cv2.imshow("f8-track", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            frame_idx += 1
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if jsonl_file is not None:
            jsonl_file.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    elapsed = max(1e-6, time.time() - start)
    fps_eff = frame_idx / elapsed
    print(f"Done. frames={frame_idx} elapsed={elapsed:.1f}s ({fps_eff:.1f} FPS)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="f8-track")
    sub = parser.add_subparsers(dest="command", required=True)

    video = sub.add_parser("video", help="Video detection + rule-based tracking.")
    video.add_argument("video", help="Input video path.")
    video.add_argument("--jsonl", default=None, help="Optional JSONL output with per-frame tracks.")

    model_group = video.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", default=None, help="Ultralytics model path (.pt/.onnx).")
    model_group.add_argument("--config", default=None, help="Model yaml (X-AnyLabeling style).")

    video.add_argument("--device", default=None, help="Ultralytics device for .pt models, e.g. 'cpu', '0' (default: auto).")
    video.add_argument(
        "--ort-provider",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="ONNX Runtime provider for .onnx models (default: auto).",
    )
    video.add_argument("--imgsz", type=int, default=None, help="Ultralytics inference size for .pt models (optional).")

    video.add_argument("--conf", type=float, default=None, help="Confidence threshold (default: from yaml or 0.25).")
    video.add_argument("--iou", type=float, default=None, help="IoU threshold (default: from yaml or 0.45).")

    video.add_argument(
        "--tracker",
        choices=("csrt", "kcf", "mosse"),
        default="csrt",
        help="OpenCV tracker used between detections (default: csrt).",
    )
    video.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Downscale frames so max(H, W) <= this value for faster inference (default: 1024; 0 disables).",
    )
    video.add_argument("--detect-stride", type=int, default=1, help="Run detector every N frames (default: 1).")
    video.add_argument("--acquire-hz", type=float, default=10.0, help="Detection rate when acquiring targets (default: 10).")
    video.add_argument(
        "--track-detect-hz",
        type=float,
        default=1.0,
        help="Detection rate when tracking (default: 1).",
    )
    video.add_argument("--max-targets", type=int, default=2, help="Max targets to keep tracking (default: 2).")
    video.add_argument(
        "--target-class",
        action="append",
        dest="target_classes",
        help="Only track detections of this class (repeatable).",
    )
    video.add_argument(
        "--select",
        choices=("conf", "area", "center"),
        default="conf",
        help="Rule to select new targets when acquiring (default: conf).",
    )
    video.add_argument(
        "--iou-match",
        type=float,
        default=0.3,
        help="IoU threshold to match a detection to an existing track (default: 0.3).",
    )
    video.add_argument(
        "--mismatch-iou",
        type=float,
        default=0.2,
        help="If matched but IoU < this threshold, count as mismatch (default: 0.2).",
    )
    video.add_argument(
        "--mismatch-patience",
        type=int,
        default=3,
        help="Drop and reacquire a track after N mismatches (default: 3).",
    )
    video.add_argument(
        "--reinit-on-match",
        action="store_true",
        help="When matched with a detection, re-initialize CSRT from detection bbox (default: off).",
    )
    video.add_argument(
        "--max-age",
        type=int,
        default=200,
        help="How many frames a track can survive without match before dropped (default: 200).",
    )
    # Always show preview; press 'q' to quit.
    video.set_defaults(func=run_video)
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = os.sys.argv[1:]

    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.func(ns))
