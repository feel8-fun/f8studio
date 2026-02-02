from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .model_config import ModelSpec, ModelTask
from .vision_utils import LetterboxResult, letterbox_bgr, nms_xyxy


@dataclass(frozen=True)
class PoseKeypoint:
    x: float
    y: float
    score: float | None = None


@dataclass(frozen=True)
class Detection:
    cls: str
    conf: float
    xyxy: tuple[int, int, int, int]
    keypoints: list[PoseKeypoint] | None = None
    obb: list[tuple[float, float]] | None = None
    angle: float | None = None


def _choose_ort_providers(*, prefer: Literal["auto", "cuda", "cpu"]) -> list[str]:
    import onnxruntime as ort  # type: ignore

    available = list(ort.get_available_providers())
    if prefer == "cpu":
        return ["CPUExecutionProvider"]
    if prefer == "cuda":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    # auto
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class OnnxYoloDetector:
    def __init__(
        self,
        spec: ModelSpec,
        *,
        ort_provider: Literal["auto", "cuda", "cpu"] = "auto",
    ) -> None:
        import onnxruntime as ort  # type: ignore

        self.spec = spec
        providers = _choose_ort_providers(prefer=ort_provider)
        self._session = ort.InferenceSession(str(spec.onnx_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self.active_providers = list(self._session.get_providers())

    def detect(self, frame_bgr: Any) -> tuple[list[Detection], dict[str, Any]]:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        spec = self.spec
        lb: LetterboxResult = letterbox_bgr(frame_bgr, new_shape=(spec.input_width, spec.input_height))
        img_rgb = cv2.cvtColor(lb.image_bgr, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3xHxW

        out0 = self._session.run(None, {self._input_name: x})[0]
        pred = np.asarray(out0)
        if pred.ndim != 3:
            return [], {"reason": "unexpected_pred_ndim", "pred_shape": list(pred.shape)}

        # Accept both (1, C, N) and (1, N, C)
        if pred.shape[1] < pred.shape[2]:
            pred = np.transpose(pred, (0, 2, 1))
        pred = pred[0]  # (N, C)

        if spec.task == "yolo_pose":
            return self._decode_pose(pred, lb=lb, frame_bgr=frame_bgr), {"pred_shape": list(pred.shape)}
        if spec.task == "yolo_obb":
            return self._decode_obb(pred, lb=lb, frame_bgr=frame_bgr), {"pred_shape": list(pred.shape)}
        return self._decode_det(pred, lb=lb, frame_bgr=frame_bgr), {"pred_shape": list(pred.shape)}

    def _decode_det(self, pred: Any, *, lb: LetterboxResult, frame_bgr: Any) -> list[Detection]:
        import numpy as np  # type: ignore

        spec = self.spec
        c = int(pred.shape[1])
        names = {int(i): str(v) for i, v in enumerate(spec.classes or [])}
        nc = len(names) if names else max(1, c - 4)
        has_obj = c == 5 + nc

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

        keep0 = scores >= float(spec.conf_threshold)
        if not np.any(keep0):
            return []

        xywh = xywh[keep0]
        scores = scores[keep0]
        cls_idx = cls_idx[keep0]

        boxes = self._xywh_to_xyxy_lb(xywh)
        keep_idx = nms_xyxy(boxes, scores, iou_thr=float(spec.iou_threshold))
        if not keep_idx:
            return []

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        cls_idx = cls_idx[keep_idx]

        boxes_img = self._map_boxes_to_frame(boxes, lb=lb, frame_bgr=frame_bgr)
        out: list[Detection] = []
        for (x1f, y1f, x2f, y2f), sc, ci in zip(boxes_img, scores, cls_idx, strict=False):
            x1i, y1i, x2i, y2i = int(round(float(x1f))), int(round(float(y1f))), int(round(float(x2f))), int(round(float(y2f)))
            if x2i <= x1i or y2i <= y1i:
                continue
            out.append(Detection(cls=names.get(int(ci), str(int(ci))), conf=float(sc), xyxy=(x1i, y1i, x2i, y2i)))
        return out

    def _decode_pose(self, pred: Any, *, lb: LetterboxResult, frame_bgr: Any) -> list[Detection]:
        import numpy as np  # type: ignore

        spec = self.spec
        names = {int(i): str(v) for i, v in enumerate(spec.classes or [])}
        nc = len(names) if names else 1
        dims = max(1, int(spec.keypoint_dims or 3))

        c = int(pred.shape[1])
        # Prefer explicit keypoint list length; otherwise infer from channel count.
        kpt_count = len(spec.keypoints) if spec.keypoints else None

        # NMS-style export layout:
        #   [x1, y1, x2, y2, score, cls] + [kpts...]
        # Common for "end-to-end" (NMS-in-graph) exports, e.g. output shape (1, 300, 57).
        if kpt_count is not None and c == 6 + int(kpt_count) * dims:
            cls_col = pred[:, 5]
            cls_int_like = False
            try:
                cls_int_like = bool(np.mean(np.abs(cls_col - np.round(cls_col)) < 1e-3) > 0.95)
            except Exception:
                cls_int_like = False

            if nc == 1 or cls_int_like:
                boxes = pred[:, 0:4].astype(np.float32, copy=False)
                scores = pred[:, 4].astype(np.float32, copy=False)
                cls_idx = np.round(cls_col).astype(np.int64, copy=False)
                cls_idx = np.clip(cls_idx, 0, max(0, nc - 1))
                kpts = pred[:, 6 : 6 + int(kpt_count) * dims].astype(np.float32, copy=False)

                # Some exports emit normalized coords (0..1). Detect and scale to model input.
                w_in = float(lb.image_bgr.shape[1])
                h_in = float(lb.image_bgr.shape[0])
                if float(np.max(boxes)) <= 1.5:
                    boxes = boxes.copy()
                    boxes[:, [0, 2]] *= w_in
                    boxes[:, [1, 3]] *= h_in
                if kpts.size and float(np.max(kpts[:, 0::dims])) <= 1.5 and float(np.max(kpts[:, 1::dims])) <= 1.5:
                    kpts = kpts.copy()
                    kpts[:, 0::dims] *= w_in
                    kpts[:, 1::dims] *= h_in

                keep0 = scores >= float(spec.conf_threshold)
                if not np.any(keep0):
                    return []
                boxes = boxes[keep0]
                scores = scores[keep0]
                cls_idx = cls_idx[keep0]
                kpts = kpts[keep0]

                keep_idx = nms_xyxy(boxes, scores, iou_thr=float(spec.iou_threshold))
                if not keep_idx:
                    return []
                boxes = boxes[keep_idx]
                scores = scores[keep_idx]
                cls_idx = cls_idx[keep_idx]
                kpts = kpts[keep_idx]

                boxes_img = self._map_boxes_to_frame(boxes, lb=lb, frame_bgr=frame_bgr)
                out: list[Detection] = []
                h0, w0 = frame_bgr.shape[:2]
                for (x1f, y1f, x2f, y2f), sc, ci, kp_flat in zip(boxes_img, scores, cls_idx, kpts, strict=False):
                    x1i, y1i, x2i, y2i = (
                        int(round(float(x1f))),
                        int(round(float(y1f))),
                        int(round(float(x2f))),
                        int(round(float(y2f))),
                    )
                    if x2i <= x1i or y2i <= y1i:
                        continue
                    kp = np.asarray(kp_flat, dtype=np.float32).reshape((int(kpt_count), dims))
                    kps_out: list[PoseKeypoint] = []
                    for j in range(int(kpt_count)):
                        x = float(kp[j, 0])
                        y = float(kp[j, 1])
                        s = float(kp[j, 2]) if dims >= 3 else None
                        x = (x - float(lb.pad_x)) / float(lb.scale if lb.scale > 0 else 1.0)
                        y = (y - float(lb.pad_y)) / float(lb.scale if lb.scale > 0 else 1.0)
                        x = max(0.0, min(float(w0), x))
                        y = max(0.0, min(float(h0), y))
                        kps_out.append(PoseKeypoint(x=x, y=y, score=s))
                    out.append(
                        Detection(
                            cls=names.get(int(ci), str(int(ci))),
                            conf=float(sc),
                            xyxy=(x1i, y1i, x2i, y2i),
                            keypoints=kps_out,
                        )
                    )
                return out

        # Two common layouts:
        # - [xywh] + [cls...] + [kpts...]
        # - [xywh] + [obj] + [cls...] + [kpts...]
        has_obj = False
        kpt_off = 4 + nc
        if kpt_count is not None:
            if c == 5 + nc + kpt_count * dims:
                has_obj = True
                kpt_off = 5 + nc
            elif c == 4 + nc + kpt_count * dims:
                has_obj = False
                kpt_off = 4 + nc
        else:
            # Infer count by divisibility.
            if c > 5 + nc and (c - (5 + nc)) % dims == 0:
                has_obj = True
                kpt_off = 5 + nc
                kpt_count = (c - (5 + nc)) // dims
            elif c > 4 + nc and (c - (4 + nc)) % dims == 0:
                has_obj = False
                kpt_off = 4 + nc
                kpt_count = (c - (4 + nc)) // dims
            else:
                return []

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

        keep0 = scores >= float(spec.conf_threshold)
        if not np.any(keep0):
            return []

        xywh = xywh[keep0]
        scores = scores[keep0]
        cls_idx = cls_idx[keep0]
        kpts = pred[keep0, kpt_off : kpt_off + int(kpt_count) * dims]

        boxes = self._xywh_to_xyxy_lb(xywh)
        keep_idx = nms_xyxy(boxes, scores, iou_thr=float(spec.iou_threshold))
        if not keep_idx:
            return []

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        cls_idx = cls_idx[keep_idx]
        kpts = kpts[keep_idx]

        boxes_img = self._map_boxes_to_frame(boxes, lb=lb, frame_bgr=frame_bgr)
        out: list[Detection] = []

        h0, w0 = frame_bgr.shape[:2]
        for (x1f, y1f, x2f, y2f), sc, ci, kp_flat in zip(boxes_img, scores, cls_idx, kpts, strict=False):
            x1i, y1i, x2i, y2i = int(round(float(x1f))), int(round(float(y1f))), int(round(float(x2f))), int(round(float(y2f)))
            if x2i <= x1i or y2i <= y1i:
                continue

            kp = np.asarray(kp_flat, dtype=np.float32).reshape((int(kpt_count), dims))
            kps_out: list[PoseKeypoint] = []
            for j in range(int(kpt_count)):
                x = float(kp[j, 0])
                y = float(kp[j, 1])
                s = float(kp[j, 2]) if dims >= 3 else None
                # Map back to frame coords.
                x = (x - float(lb.pad_x)) / float(lb.scale if lb.scale > 0 else 1.0)
                y = (y - float(lb.pad_y)) / float(lb.scale if lb.scale > 0 else 1.0)
                x = max(0.0, min(float(w0), x))
                y = max(0.0, min(float(h0), y))
                kps_out.append(PoseKeypoint(x=x, y=y, score=s))

            out.append(
                Detection(
                    cls=names.get(int(ci), str(int(ci))),
                    conf=float(sc),
                    xyxy=(x1i, y1i, x2i, y2i),
                    keypoints=kps_out,
                )
            )
        return out

    def _decode_obb(self, pred: Any, *, lb: LetterboxResult, frame_bgr: Any) -> list[Detection]:
        import numpy as np  # type: ignore

        spec = self.spec
        names = {int(i): str(v) for i, v in enumerate(spec.classes or [])}
        nc = len(names) if names else 1
        c = int(pred.shape[1])

        # Common layouts:
        # - [x,y,w,h,angle] + [cls...]
        # - [x,y,w,h,angle] + [obj] + [cls...]
        has_obj = c == 6 + nc
        if not (c == 5 + nc or c == 6 + nc):
            return []

        xywha = pred[:, 0:5]
        if has_obj:
            obj = pred[:, 5]
            cls_scores = pred[:, 6 : 6 + nc]
            cls_idx = cls_scores.argmax(axis=1)
            cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_idx]
            scores = obj * cls_conf
        else:
            cls_scores = pred[:, 5 : 5 + nc]
            cls_idx = cls_scores.argmax(axis=1)
            scores = cls_scores[np.arange(cls_scores.shape[0]), cls_idx]

        keep0 = scores >= float(spec.conf_threshold)
        if not np.any(keep0):
            return []

        xywha = xywha[keep0]
        scores = scores[keep0]
        cls_idx = cls_idx[keep0]

        polys_lb = self._xywha_to_poly_lb(xywha)
        # Approximate NMS via axis-aligned bounds.
        boxes = np.stack(
            [
                polys_lb[:, :, 0].min(axis=1),
                polys_lb[:, :, 1].min(axis=1),
                polys_lb[:, :, 0].max(axis=1),
                polys_lb[:, :, 1].max(axis=1),
            ],
            axis=1,
        )
        keep_idx = nms_xyxy(boxes, scores, iou_thr=float(spec.iou_threshold))
        if not keep_idx:
            return []

        polys_lb = polys_lb[keep_idx]
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        cls_idx = cls_idx[keep_idx]

        boxes_img = self._map_boxes_to_frame(boxes, lb=lb, frame_bgr=frame_bgr)
        polys_img = self._map_polys_to_frame(polys_lb, lb=lb, frame_bgr=frame_bgr)

        out: list[Detection] = []
        for (x1f, y1f, x2f, y2f), poly, sc, ci in zip(boxes_img, polys_img, scores, cls_idx, strict=False):
            x1i, y1i, x2i, y2i = int(round(float(x1f))), int(round(float(y1f))), int(round(float(x2f))), int(round(float(y2f)))
            if x2i <= x1i or y2i <= y1i:
                continue
            out.append(
                Detection(
                    cls=names.get(int(ci), str(int(ci))),
                    conf=float(sc),
                    xyxy=(x1i, y1i, x2i, y2i),
                    obb=[(float(x), float(y)) for x, y in poly],
                )
            )
        return out

    @staticmethod
    def _xywh_to_xyxy_lb(xywh: Any) -> Any:
        import numpy as np  # type: ignore

        x_c = xywh[:, 0]
        y_c = xywh[:, 1]
        w = xywh[:, 2]
        h = xywh[:, 3]
        x1 = x_c - w / 2.0
        y1 = y_c - h / 2.0
        x2 = x_c + w / 2.0
        y2 = y_c + h / 2.0
        return np.stack([x1, y1, x2, y2], axis=1)

    def _map_boxes_to_frame(self, boxes_lb: Any, *, lb: LetterboxResult, frame_bgr: Any) -> Any:
        import numpy as np  # type: ignore

        boxes = boxes_lb.astype(np.float32, copy=True)
        boxes[:, [0, 2]] -= float(lb.pad_x)
        boxes[:, [1, 3]] -= float(lb.pad_y)
        boxes /= float(lb.scale) if lb.scale > 0 else 1.0

        h0, w0 = frame_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, float(w0))
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, float(h0))
        return boxes

    def _xywha_to_poly_lb(self, xywha: Any) -> Any:
        import numpy as np  # type: ignore

        angle_unit = "deg"
        try:
            ymeta = (self.spec.meta or {}).get("yolo")
            if isinstance(ymeta, dict):
                angle_unit = str(ymeta.get("angleUnit") or "deg").strip().lower()
        except Exception:
            angle_unit = "deg"

        polys: list[Any] = []
        for (cx, cy, w, h, ang) in xywha:
            a = float(ang)
            if angle_unit in ("rad", "radian", "radians"):
                a = a * 180.0 / math.pi
            try:
                import cv2  # type: ignore

                pts = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(a)))  # type: ignore[arg-type]
                polys.append(pts.astype(np.float32))
            except Exception:
                # Fallback: axis-aligned rect in letterbox coords.
                x1 = float(cx) - float(w) / 2.0
                y1 = float(cy) - float(h) / 2.0
                x2 = float(cx) + float(w) / 2.0
                y2 = float(cy) + float(h) / 2.0
                polys.append(np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
        return np.stack(polys, axis=0) if polys else np.zeros((0, 4, 2), dtype=np.float32)

    def _map_polys_to_frame(self, polys_lb: Any, *, lb: LetterboxResult, frame_bgr: Any) -> Any:
        import numpy as np  # type: ignore

        polys = polys_lb.astype(np.float32, copy=True)
        polys[:, :, 0] -= float(lb.pad_x)
        polys[:, :, 1] -= float(lb.pad_y)
        polys /= float(lb.scale) if lb.scale > 0 else 1.0

        h0, w0 = frame_bgr.shape[:2]
        polys[:, :, 0] = polys[:, :, 0].clip(0, float(w0))
        polys[:, :, 1] = polys[:, :, 1].clip(0, float(h0))
        return polys
