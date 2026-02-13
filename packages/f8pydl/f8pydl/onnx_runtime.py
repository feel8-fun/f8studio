from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from .model_config import ModelSpec
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


@dataclass(frozen=True)
class Classification:
    cls: str
    score: float


def _choose_ort_providers(*, prefer: Literal["auto", "cuda", "cpu"]) -> list[str]:
    import onnxruntime as ort  # type: ignore

    try:
        available = list(ort.get_available_providers())  # type: ignore[attr-defined]
    except Exception:
        available = []
    by_lower = {str(p).lower(): str(p) for p in available}
    cuda = by_lower.get("cudaexecutionprovider", "CUDAExecutionProvider")
    cpu = by_lower.get("cpuexecutionprovider", "CPUExecutionProvider")
    if prefer == "cpu":
        return [cpu]
    if prefer == "cuda":
        if "cudaexecutionprovider" in by_lower:
            return [cuda, cpu]
        return [cpu]
    if "cudaexecutionprovider" in by_lower:
        return [cuda, cpu]
    return [cpu]


class _OnnxSession:
    def __init__(self, model_path: str, *, ort_provider: Literal["auto", "cuda", "cpu"]) -> None:
        import onnxruntime as ort  # type: ignore

        self.provider_warning: str = ""
        providers = _choose_ort_providers(prefer=ort_provider)
        try:
            self._session = ort.InferenceSession(model_path, providers=providers)
        except Exception as exc:
            prefer = str(ort_provider or "auto").lower()
            if prefer in ("auto", "cuda"):
                try:
                    available = list(ort.get_available_providers())  # type: ignore[attr-defined]
                except Exception:
                    available = []
                self.provider_warning = (
                    f"Failed to init ORT providers={providers!r}; falling back to CPUExecutionProvider. "
                    f"availableProviders={available!r}; error={exc}"
                )
                self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            else:
                raise
        self.input_name = str(self._session.get_inputs()[0].name)
        self.active_providers = list(self._session.get_providers())
        self.input_meta = self._session.get_inputs()[0]

    def run(self, x: Any) -> Any:
        out = self._session.run(None, {self.input_name: x})
        if not out:
            raise RuntimeError("ONNX Runtime returned empty outputs.")
        return out[0]


class OnnxYoloDetectorRuntime:
    def __init__(self, spec: ModelSpec, *, ort_provider: Literal["auto", "cuda", "cpu"] = "auto") -> None:
        self.spec = spec
        self._session = _OnnxSession(str(spec.onnx_path), ort_provider=ort_provider)
        self.provider_warning = self._session.provider_warning
        self.active_providers = self._session.active_providers

    def infer(self, frame_bgr: Any) -> tuple[list[Detection], dict[str, Any]]:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        spec = self.spec
        lb: LetterboxResult = letterbox_bgr(frame_bgr, new_shape=(spec.input_width, spec.input_height))
        img_rgb = cv2.cvtColor(lb.image_bgr, cv2.COLOR_BGR2RGB)
        x = img_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # 1x3xHxW

        out0 = self._session.run(x)
        pred = np.asarray(out0)
        if pred.ndim != 3:
            return [], {"reason": "unexpected_pred_ndim", "pred_shape": list(pred.shape)}

        if pred.shape[1] < pred.shape[2]:
            pred = np.transpose(pred, (0, 2, 1))
        pred = pred[0]

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
            x1i = int(round(float(x1f)))
            y1i = int(round(float(y1f)))
            x2i = int(round(float(x2f)))
            y2i = int(round(float(y2f)))
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
        kpt_count = len(spec.keypoints) if spec.keypoints else None

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
                    x1i = int(round(float(x1f)))
                    y1i = int(round(float(y1f)))
                    x2i = int(round(float(x2f)))
                    y2i = int(round(float(y2f)))
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
            x1i = int(round(float(x1f)))
            y1i = int(round(float(y1f)))
            x2i = int(round(float(x2f)))
            y2i = int(round(float(y2f)))
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

    def _decode_obb(self, pred: Any, *, lb: LetterboxResult, frame_bgr: Any) -> list[Detection]:
        import numpy as np  # type: ignore

        spec = self.spec
        names = {int(i): str(v) for i, v in enumerate(spec.classes or [])}
        nc = len(names) if names else 1
        c = int(pred.shape[1])
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
            x1i = int(round(float(x1f)))
            y1i = int(round(float(y1f)))
            x2i = int(round(float(x2f)))
            y2i = int(round(float(y2f)))
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


class OnnxClassifierRuntime:
    def __init__(self, spec: ModelSpec, *, ort_provider: Literal["auto", "cuda", "cpu"] = "auto") -> None:
        self.spec = spec
        self._session = _OnnxSession(str(spec.onnx_path), ort_provider=ort_provider)
        self.provider_warning = self._session.provider_warning
        self.active_providers = self._session.active_providers

    def infer(self, frame_bgr: Any, *, top_k: int) -> tuple[list[Classification], dict[str, Any]]:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        x = self._prepare_input(frame_bgr, cv2=cv2, np=np)
        out0 = self._session.run(x)
        logits = np.asarray(out0)
        scores = self._flatten_scores(logits, np=np)
        probs = self._softmax(scores, np=np)
        top_n = max(1, min(int(top_k), int(probs.shape[0])))
        indices = np.argsort(probs)[::-1][:top_n]
        names = self.spec.classes

        out: list[Classification] = []
        for idx in indices:
            i = int(idx)
            cls_name = names[i] if i < len(names) else str(i)
            out.append(Classification(cls=cls_name, score=float(probs[i])))
        return out, {"pred_shape": list(logits.shape)}

    def _prepare_input(self, frame_bgr: Any, *, cv2: Any, np: Any) -> Any:
        spec = self.spec
        img = cv2.resize(frame_bgr, (int(spec.input_width), int(spec.input_height)), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_type = str(self._session.input_meta.type or "").lower()
        is_float_input = "float" in input_type

        input_shape = self._session.input_meta.shape
        is_nchw = True
        if isinstance(input_shape, list) and len(input_shape) == 4:
            ch_dim = input_shape[1]
            if isinstance(ch_dim, int) and ch_dim != 3:
                is_nchw = False
            if input_shape[3] == 3:
                is_nchw = False

        if is_float_input:
            x = img_rgb.astype(np.float32) / 255.0
        else:
            x = img_rgb.astype(np.uint8)

        if is_nchw:
            x = np.transpose(x, (2, 0, 1))[None, ...]
        else:
            x = x[None, ...]
        return x

    @staticmethod
    def _flatten_scores(logits: Any, *, np: Any) -> Any:
        if logits.ndim == 1:
            return logits.astype(np.float32, copy=False)
        if logits.ndim == 2:
            return logits[0].astype(np.float32, copy=False)
        if logits.ndim == 3:
            if logits.shape[0] == 1:
                return logits.reshape((-1,)).astype(np.float32, copy=False)
            raise ValueError(f"Unsupported classification output shape: {logits.shape}")
        raise ValueError(f"Unsupported classification output ndim: {logits.ndim}")

    @staticmethod
    def _softmax(scores: Any, *, np: Any) -> Any:
        shifted = scores - np.max(scores)
        ex = np.exp(shifted)
        denom = np.sum(ex)
        if float(denom) <= 0.0:
            return np.zeros_like(scores, dtype=np.float32)
        return ex / denom
