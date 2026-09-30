from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from .model_config import ModelSpec
from .vision_utils import LetterboxResult, letterbox_bgr, nms_xyxy

logger = logging.getLogger(__name__)

_ORT_CPU_PROVIDER = "CPUExecutionProvider"
_ORT_CUDA_PROVIDER = "CUDAExecutionProvider"
_ORT_PROVIDER_QUERY_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)
_CV2_IMPORT_ERRORS = (ImportError, OSError)
_NUMERIC_INFERENCE_ERRORS = (FloatingPointError, TypeError, ValueError)


class _OrtInferenceSessionFactory(Protocol):
    def __call__(self, model_path: str, *, providers: list[str]) -> Any: ...


class _OrtModule(Protocol):
    InferenceSession: _OrtInferenceSessionFactory

    def get_available_providers(self) -> list[str]: ...


class _Cv2Module(Protocol):
    error: type[Exception]

    def boxPoints(self, box: tuple[tuple[float, float], tuple[float, float], float]) -> Any: ...  # noqa: N802


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


@dataclass(frozen=True)
class TemporalDetectorInputSpec:
    layout: Literal["bcthw", "btchw"]
    clip_length: int
    channels: int
    input_height: int
    input_width: int


@dataclass(frozen=True)
class _OrtSessionInitResult:
    session: Any
    provider_warning: str


def _available_ort_providers(ort: _OrtModule) -> list[str]:
    try:
        return [str(provider) for provider in ort.get_available_providers()]
    except _ORT_PROVIDER_QUERY_ERRORS:
        logger.debug("failed to query ONNX Runtime providers", exc_info=True)
        return []


def _choose_ort_providers_from_available(
    available: list[str],
    *,
    prefer: Literal["auto", "cuda", "cpu"],
) -> list[str]:
    by_lower = {str(p).lower(): str(p) for p in available}
    cuda = by_lower.get("cudaexecutionprovider", _ORT_CUDA_PROVIDER)
    cpu = by_lower.get("cpuexecutionprovider", _ORT_CPU_PROVIDER)
    if prefer == "cpu":
        return [cpu]
    if prefer == "cuda":
        if "cudaexecutionprovider" in by_lower:
            return [cuda, cpu]
        return [cpu]
    if "cudaexecutionprovider" in by_lower:
        return [cuda, cpu]
    return [cpu]


def _choose_ort_providers(*, prefer: Literal["auto", "cuda", "cpu"]) -> list[str]:
    import onnxruntime as ort  # type: ignore

    return _choose_ort_providers_from_available(_available_ort_providers(cast(_OrtModule, ort)), prefer=prefer)


def _create_ort_session(
    ort: _OrtModule,
    model_path: str,
    *,
    ort_provider: Literal["auto", "cuda", "cpu"],
) -> _OrtSessionInitResult:
    providers = _choose_ort_providers_from_available(_available_ort_providers(cast(_OrtModule, ort)), prefer=ort_provider)
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        return _OrtSessionInitResult(session=session, provider_warning="")
    except Exception as exc:
        if ort_provider == "cpu":
            raise
        available = _available_ort_providers(cast(_OrtModule, ort))
        provider_warning = (
            f"Failed to init ORT providers={providers!r}; falling back to CPUExecutionProvider. "
            f"availableProviders={available!r}; error={exc}"
        )
        logger.warning(
            "failed to initialize ONNX Runtime session; falling back to CPU provider "
            "model_path=%s providers=%r available_providers=%r",
            model_path,
            providers,
            available,
            exc_info=True,
        )
        session = ort.InferenceSession(model_path, providers=[_ORT_CPU_PROVIDER])
        return _OrtSessionInitResult(session=session, provider_warning=provider_warning)


class _OnnxSession:
    def __init__(self, model_path: str, *, ort_provider: Literal["auto", "cuda", "cpu"]) -> None:
        import onnxruntime as ort  # type: ignore

        session_result = _create_ort_session(cast(_OrtModule, ort), model_path, ort_provider=ort_provider)
        self._session = session_result.session
        self.provider_warning = session_result.provider_warning
        inputs = list(self._session.get_inputs())
        if not inputs:
            raise ValueError("ONNX model must have at least 1 input")
        self.input_name = str(inputs[0].name)
        self.active_providers = list(self._session.get_providers())
        self.input_meta = inputs[0]

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
            except _NUMERIC_INFERENCE_ERRORS:
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
                detections: list[Detection] = []
                h0, w0 = frame_bgr.shape[:2]
                for (x1f, y1f, x2f, y2f), sc, ci, kp_flat in zip(boxes_img, scores, cls_idx, kpts, strict=False):
                    x1i = int(round(float(x1f)))
                    y1i = int(round(float(y1f)))
                    x2i = int(round(float(x2f)))
                    y2i = int(round(float(y2f)))
                    if x2i <= x1i or y2i <= y1i:
                        continue
                    kp = np.asarray(kp_flat, dtype=np.float32).reshape((int(kpt_count), dims))
                    pose_keypoints: list[PoseKeypoint] = []
                    for j in range(int(kpt_count)):
                        x = float(kp[j, 0])
                        y = float(kp[j, 1])
                        s = float(kp[j, 2]) if dims >= 3 else None
                        x = (x - float(lb.pad_x)) / float(lb.scale if lb.scale > 0 else 1.0)
                        y = (y - float(lb.pad_y)) / float(lb.scale if lb.scale > 0 else 1.0)
                        x = max(0.0, min(float(w0), x))
                        y = max(0.0, min(float(h0), y))
                        pose_keypoints.append(PoseKeypoint(x=x, y=y, score=s))
                    detections.append(
                        Detection(
                            cls=names.get(int(ci), str(int(ci))),
                            conf=float(sc),
                            xyxy=(x1i, y1i, x2i, y2i),
                            keypoints=pose_keypoints,
                        )
                    )
                return detections

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

        angle_unit = self._obb_angle_unit()
        cv2_module = self._optional_cv2_module()

        polys: list[Any] = []
        for (cx, cy, w, h, ang) in xywha:
            a = float(ang)
            if angle_unit in ("rad", "radian", "radians"):
                a = a * 180.0 / math.pi
            pts = self._cv2_box_points(cv2_module, cx=float(cx), cy=float(cy), w=float(w), h=float(h), angle_deg=float(a))
            if pts is not None:
                polys.append(pts.astype(np.float32))
                continue
            x1 = float(cx) - float(w) / 2.0
            y1 = float(cy) - float(h) / 2.0
            x2 = float(cx) + float(w) / 2.0
            y2 = float(cy) + float(h) / 2.0
            polys.append(np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32))
        return np.stack(polys, axis=0) if polys else np.zeros((0, 4, 2), dtype=np.float32)

    def _obb_angle_unit(self) -> str:
        meta = self.spec.meta
        if meta is None:
            return "deg"
        ymeta = meta.get("yolo")
        if not isinstance(ymeta, dict):
            return "deg"
        return str(ymeta.get("angleUnit") or "deg").strip().lower()

    @staticmethod
    def _optional_cv2_module() -> _Cv2Module | None:
        try:
            import cv2  # type: ignore
        except _CV2_IMPORT_ERRORS:
            logger.debug("OpenCV is unavailable while decoding OBB polygons", exc_info=True)
            return None
        return cast(_Cv2Module, cv2)

    @staticmethod
    def _cv2_box_points(
        cv2_module: _Cv2Module | None,
        *,
        cx: float,
        cy: float,
        w: float,
        h: float,
        angle_deg: float,
    ) -> Any | None:
        if cv2_module is None:
            return None
        try:
            return cv2_module.boxPoints(((cx, cy), (w, h), angle_deg))
        except (cv2_module.error, TypeError, ValueError):
            logger.debug("OpenCV failed to decode OBB polygon; using axis-aligned fallback", exc_info=True)
            return None

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


class OnnxYowoTemporalDetectorRuntime:
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, spec: ModelSpec, *, ort_provider: Literal["auto", "cuda", "cpu"] = "auto") -> None:
        self.spec = spec
        self._session = _OnnxSession(str(spec.onnx_path), ort_provider=ort_provider)
        self.provider_warning = self._session.provider_warning
        self.active_providers = self._session.active_providers
        input_shape = list(self._session.input_meta.shape) if isinstance(self._session.input_meta.shape, list) else []
        input_spec = self._extract_input_spec(
            input_shape,
            default_clip_length=int(spec.temporal_clip_length),
            default_height=int(spec.input_height),
            default_width=int(spec.input_width),
        )
        self._input_spec = input_spec
        if int(spec.input_height) != int(input_spec.input_height) or int(spec.input_width) != int(input_spec.input_width):
            mismatch = (
                "Model input shape is fixed and differs from yaml input size; "
                f"using model shape HxW={int(input_spec.input_height)}x{int(input_spec.input_width)} "
                f"(yaml HxW={int(spec.input_height)}x{int(spec.input_width)})."
            )
            if self.provider_warning:
                self.provider_warning = f"{self.provider_warning}\n{mismatch}"
            else:
                self.provider_warning = mismatch

    @property
    def clip_length(self) -> int:
        return int(self._input_spec.clip_length)

    @property
    def sampling_rate(self) -> int:
        return int(self.spec.temporal_sampling_rate)

    @property
    def buffer_span(self) -> int:
        return (int(self.clip_length) - 1) * int(self.sampling_rate) + 1

    def prepare_frame(self, frame_bgr: Any) -> Any:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if int(self._input_spec.channels) != 3:
            raise ValueError(f"Temporal detector currently supports 3-channel input, got {self._input_spec.channels}")
        resized = cv2.resize(
            frame_bgr,
            (int(self._input_spec.input_width), int(self._input_spec.input_height)),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        if self.spec.temporal_normalization == "imagenet":
            mean = np.asarray(self._IMAGENET_MEAN, dtype=np.float32).reshape((3, 1, 1))
            std = np.asarray(self._IMAGENET_STD, dtype=np.float32).reshape((3, 1, 1))
            x = (x - mean) / std
        return np.ascontiguousarray(x, dtype=np.float32)

    def infer_sequence(self, sequence_chw: Any, *, frame_size_hw: tuple[int, int]) -> tuple[list[Detection], dict[str, Any]]:
        import numpy as np  # type: ignore

        sequence = np.asarray(sequence_chw, dtype=np.float32)
        if sequence.ndim != 4:
            raise ValueError(f"Temporal detector sequence must have shape SxCxHxW, got {sequence.shape!r}")
        if int(sequence.shape[0]) != int(self._input_spec.clip_length):
            raise ValueError(
                f"Temporal detector expected sequence length {self._input_spec.clip_length}, "
                f"got {int(sequence.shape[0])}"
            )
        if int(sequence.shape[1]) != int(self._input_spec.channels):
            raise ValueError(
                f"Temporal detector expected channels {self._input_spec.channels}, got {int(sequence.shape[1])}"
            )
        if (
            int(sequence.shape[2]) != int(self._input_spec.input_height)
            or int(sequence.shape[3]) != int(self._input_spec.input_width)
        ):
            raise ValueError(
                "Temporal detector sequence size mismatch: "
                f"expected CxHxW={self._input_spec.channels}x{self._input_spec.input_height}x{self._input_spec.input_width}, "
                f"got {int(sequence.shape[1])}x{int(sequence.shape[2])}x{int(sequence.shape[3])}"
            )

        input_tensor = self._to_input_tensor(sequence, layout=self._input_spec.layout)
        out0 = self._session.run(input_tensor)
        pred = np.asarray(out0)
        if pred.ndim != 3:
            return [], {"reason": "unexpected_pred_ndim", "pred_shape": list(pred.shape)}
        if pred.shape[1] < pred.shape[2]:
            pred = np.transpose(pred, (0, 2, 1))
        pred = pred[0]
        detections = self._decode_det(pred, frame_size_hw=frame_size_hw)
        return detections, {"pred_shape": list(pred.shape), "input_layout": self._input_spec.layout}

    def _decode_det(self, pred: Any, *, frame_size_hw: tuple[int, int]) -> list[Detection]:
        import numpy as np  # type: ignore

        c = int(pred.shape[1])
        names = {int(i): str(v) for i, v in enumerate(self.spec.classes or [])}
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

        keep0 = scores >= float(self.spec.conf_threshold)
        if not np.any(keep0):
            return []

        xywh = xywh[keep0]
        scores = scores[keep0]
        cls_idx = cls_idx[keep0]

        boxes = OnnxYoloDetectorRuntime._xywh_to_xyxy_lb(xywh)
        keep_idx = nms_xyxy(boxes, scores, iou_thr=float(self.spec.iou_threshold))
        if not keep_idx:
            return []
        max_det = max(1, int(self.spec.temporal_max_det))
        keep_idx = keep_idx[:max_det]

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        cls_idx = cls_idx[keep_idx]

        boxes_img = self._map_boxes_to_frame(
            boxes,
            frame_size_hw=frame_size_hw,
            input_width=int(self._input_spec.input_width),
            input_height=int(self._input_spec.input_height),
        )
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

    @staticmethod
    def _to_input_tensor(sequence: Any, *, layout: Literal["bcthw", "btchw"]) -> Any:
        import numpy as np  # type: ignore

        if layout == "bcthw":
            return np.ascontiguousarray(np.transpose(sequence, (1, 0, 2, 3))[None, ...], dtype=np.float32)
        return np.ascontiguousarray(sequence[None, ...], dtype=np.float32)

    @staticmethod
    def _extract_input_spec(
        shape: list[Any],
        *,
        default_clip_length: int,
        default_height: int,
        default_width: int,
    ) -> TemporalDetectorInputSpec:
        if len(shape) != 5:
            raise ValueError(f"Temporal detector input must be rank-5, got {shape!r}")
        batch = shape[0]
        if isinstance(batch, int) and batch not in (0, 1):
            raise ValueError(f"Temporal detector batch dimension must be 1 (or dynamic), got {batch}")

        dim1 = shape[1]
        dim2 = shape[2]
        dim3 = shape[3]
        dim4 = shape[4]
        if isinstance(dim1, int) and dim1 == 3:
            layout: Literal["bcthw", "btchw"] = "bcthw"
            clip_length = OnnxYowoTemporalDetectorRuntime._resolve_positive_dim(dim2, default_clip_length, "clip_length")
            channels = 3
            input_height = OnnxYowoTemporalDetectorRuntime._resolve_positive_dim(dim3, default_height, "input_height")
            input_width = OnnxYowoTemporalDetectorRuntime._resolve_positive_dim(dim4, default_width, "input_width")
            return TemporalDetectorInputSpec(
                layout=layout,
                clip_length=clip_length,
                channels=channels,
                input_height=input_height,
                input_width=input_width,
            )
        if isinstance(dim2, int) and dim2 == 3:
            layout = "btchw"
            clip_length = OnnxYowoTemporalDetectorRuntime._resolve_positive_dim(dim1, default_clip_length, "clip_length")
            channels = 3
            input_height = OnnxYowoTemporalDetectorRuntime._resolve_positive_dim(dim3, default_height, "input_height")
            input_width = OnnxYowoTemporalDetectorRuntime._resolve_positive_dim(dim4, default_width, "input_width")
            return TemporalDetectorInputSpec(
                layout=layout,
                clip_length=clip_length,
                channels=channels,
                input_height=input_height,
                input_width=input_width,
            )
        raise ValueError(f"Temporal detector input shape must encode 3 RGB channels in dim1 or dim2, got {shape!r}")

    @staticmethod
    def _resolve_positive_dim(value: Any, default: int, label: str) -> int:
        if isinstance(value, int):
            if value <= 0:
                raise ValueError(f"Temporal detector {label} must be positive, got {value}")
            return int(value)
        if int(default) <= 0:
            raise ValueError(f"Temporal detector default {label} must be positive, got {default}")
        return int(default)

    @staticmethod
    def _map_boxes_to_frame(boxes_model: Any, *, frame_size_hw: tuple[int, int], input_width: int, input_height: int) -> Any:
        import numpy as np  # type: ignore

        frame_height = int(frame_size_hw[0])
        frame_width = int(frame_size_hw[1])
        if frame_height <= 0 or frame_width <= 0:
            raise ValueError(f"Invalid frame size for temporal detector: {frame_size_hw!r}")
        if input_width <= 0 or input_height <= 0:
            raise ValueError(f"Invalid temporal detector input size: {(input_width, input_height)!r}")

        boxes = np.asarray(boxes_model, dtype=np.float32).copy()
        boxes[:, [0, 2]] *= float(frame_width) / float(input_width)
        boxes[:, [1, 3]] *= float(frame_height) / float(input_height)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, float(frame_width))
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, float(frame_height))
        return boxes


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
        input_shape = list(self._session.input_meta.shape) if isinstance(self._session.input_meta.shape, list) else []
        is_nchw = True
        if len(input_shape) == 4:
            channel_first = input_shape[1]
            channel_last = input_shape[3]
            if channel_last in (1, 3) and channel_first not in (1, 3):
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


class OnnxNeuFlowRuntime:
    def __init__(self, spec: ModelSpec, *, ort_provider: Literal["auto", "cuda", "cpu"] = "auto") -> None:
        import onnxruntime as ort  # type: ignore

        self.spec = spec
        session_result = _create_ort_session(cast(_OrtModule, ort), str(spec.onnx_path), ort_provider=ort_provider)
        self._session = session_result.session
        self.provider_warning = session_result.provider_warning
        inputs = list(self._session.get_inputs())
        if len(inputs) != 2:
            raise ValueError(f"NeuFlow model must have exactly 2 inputs, got {len(inputs)}")
        self._input_name_prev = str(inputs[0].name)
        self._input_name_now = str(inputs[1].name)
        shape_prev = list(inputs[0].shape) if isinstance(inputs[0].shape, list) else []
        shape_now = list(inputs[1].shape) if isinstance(inputs[1].shape, list) else []
        model_hw_prev = self._extract_fixed_hw(shape_prev)
        model_hw_now = self._extract_fixed_hw(shape_now)
        self._input_height = int(spec.input_height)
        self._input_width = int(spec.input_width)
        if model_hw_prev is not None and model_hw_now is not None:
            if model_hw_prev != model_hw_now:
                raise ValueError(
                    f"NeuFlow inputs must share same HxW, got prev={model_hw_prev!r} now={model_hw_now!r}"
                )
            self._input_height = int(model_hw_prev[0])
            self._input_width = int(model_hw_prev[1])
            if int(spec.input_height) != self._input_height or int(spec.input_width) != self._input_width:
                mismatch = (
                    "Model input shape is fixed and differs from yaml input size; "
                    f"using model shape HxW={self._input_height}x{self._input_width} "
                    f"(yaml HxW={int(spec.input_height)}x{int(spec.input_width)})."
                )
                if self.provider_warning:
                    self.provider_warning = f"{self.provider_warning}\n{mismatch}"
                else:
                    self.provider_warning = mismatch
        self._output_names = [str(out.name) for out in self._session.get_outputs()]
        if not self._output_names:
            raise ValueError("NeuFlow model has no outputs.")
        self.active_providers = list(self._session.get_providers())

    def prepare_input(self, frame_bgr: Any) -> Any:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        resized = cv2.resize(
            frame_bgr,
            (int(self._input_width), int(self._input_height)),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x

    def infer_preprocessed(self, prev_tensor: Any, now_tensor: Any, *, output_size_hw: tuple[int, int]) -> Any:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if self.spec.flow_input_order == "prev_now":
            feed = {
                self._input_name_prev: prev_tensor,
                self._input_name_now: now_tensor,
            }
        else:
            feed = {
                self._input_name_prev: now_tensor,
                self._input_name_now: prev_tensor,
            }
        outputs = self._session.run(self._output_names, feed)
        if not outputs:
            raise RuntimeError("ONNX Runtime returned empty outputs for NeuFlow.")
        raw = np.asarray(outputs[0])
        flow_hw2 = self._to_flow_hw2(raw, np=np)

        out_h = int(output_size_hw[0])
        out_w = int(output_size_hw[1])
        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"Invalid output size for flow resize: {(out_h, out_w)!r}")
        in_h = int(flow_hw2.shape[0])
        in_w = int(flow_hw2.shape[1])
        resized = cv2.resize(flow_hw2, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        scale_x = float(out_w) / float(max(1, in_w))
        scale_y = float(out_h) / float(max(1, in_h))
        resized[..., 0] = resized[..., 0] * scale_x
        resized[..., 1] = resized[..., 1] * scale_y
        return np.ascontiguousarray(resized.astype(np.float32, copy=False))

    @staticmethod
    def _to_flow_hw2(raw: Any, *, np: Any) -> Any:
        if raw.ndim == 4:
            if raw.shape[0] != 1:
                raise ValueError(f"Unexpected NeuFlow batch size: {raw.shape!r}")
            if raw.shape[1] == 2:
                return np.transpose(raw[0], (1, 2, 0)).astype(np.float32, copy=False)
            if raw.shape[3] == 2:
                return raw[0].astype(np.float32, copy=False)
        if raw.ndim == 3 and raw.shape[2] == 2:
            return raw.astype(np.float32, copy=False)
        raise ValueError(f"Unsupported NeuFlow output shape: {raw.shape!r}")

    @staticmethod
    def _extract_fixed_hw(shape: list[Any]) -> tuple[int, int] | None:
        if len(shape) != 4:
            return None
        h = shape[2]
        w = shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return int(h), int(w)
        return None


class OnnxTemporalWaveRuntime:
    def __init__(
        self,
        spec: ModelSpec,
        *,
        ort_provider: Literal["auto", "cuda", "cpu"] = "auto",
        output_scale: float = 10.0,
        output_bias: float = 0.0,
    ) -> None:
        import onnxruntime as ort  # type: ignore

        self.spec = spec
        self.output_scale = float(output_scale)
        self.output_bias = float(output_bias)
        session_result = _create_ort_session(cast(_OrtModule, ort), str(spec.onnx_path), ort_provider=ort_provider)
        self._session = session_result.session
        self.provider_warning = session_result.provider_warning

        inputs = list(self._session.get_inputs())
        if len(inputs) != 1:
            raise ValueError(f"Temporal wave model must have exactly 1 input, got {len(inputs)}")
        outputs = list(self._session.get_outputs())
        if not outputs:
            raise ValueError("Temporal wave model must have at least 1 output")

        self._input_name = str(inputs[0].name)
        self._output_name = str(outputs[0].name)
        self.active_providers = list(self._session.get_providers())

        self._sequence_length = 10
        self._channels = 3
        self._input_height = int(spec.input_height)
        self._input_width = int(spec.input_width)

        input_shape = list(inputs[0].shape) if isinstance(inputs[0].shape, list) else []
        fixed = self._extract_fixed_input_shape(input_shape)
        if fixed is not None:
            self._sequence_length = int(fixed[0])
            self._channels = int(fixed[1])
            self._input_height = int(fixed[2])
            self._input_width = int(fixed[3])
            if (
                int(spec.input_height) != self._input_height
                or int(spec.input_width) != self._input_width
            ):
                mismatch = (
                    "Model input shape is fixed and differs from yaml input size; "
                    f"using model shape HxW={self._input_height}x{self._input_width} "
                    f"(yaml HxW={int(spec.input_height)}x{int(spec.input_width)})."
                )
                if self.provider_warning:
                    self.provider_warning = f"{self.provider_warning}\n{mismatch}"
                else:
                    self.provider_warning = mismatch

        output_shape = list(outputs[0].shape) if isinstance(outputs[0].shape, list) else []
        self._output_length = self._extract_output_length(output_shape)

    @property
    def sequence_length(self) -> int:
        return int(self._sequence_length)

    @property
    def channels(self) -> int:
        return int(self._channels)

    @property
    def input_height(self) -> int:
        return int(self._input_height)

    @property
    def input_width(self) -> int:
        return int(self._input_width)

    @property
    def output_length(self) -> int | None:
        return self._output_length

    def prepare_frame(self, frame_bgr: Any) -> Any:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if self._channels != 3:
            raise ValueError(f"Temporal wave runtime currently supports 3-channel input, got {self._channels}")
        resized = cv2.resize(
            frame_bgr,
            (int(self._input_width), int(self._input_height)),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        return np.ascontiguousarray(x)

    def infer_sequence(self, sequence_chw: Any) -> Any:
        import numpy as np  # type: ignore

        sequence = np.asarray(sequence_chw, dtype=np.float32)
        if sequence.ndim != 4:
            raise ValueError(f"Temporal wave sequence must have shape SxCxHxW, got {sequence.shape!r}")
        if int(sequence.shape[0]) != int(self._sequence_length):
            raise ValueError(
                f"Temporal wave expected sequence length {self._sequence_length}, got {int(sequence.shape[0])}"
            )
        if int(sequence.shape[1]) != int(self._channels):
            raise ValueError(f"Temporal wave expected channels {self._channels}, got {int(sequence.shape[1])}")
        if int(sequence.shape[2]) != int(self._input_height) or int(sequence.shape[3]) != int(self._input_width):
            raise ValueError(
                "Temporal wave sequence size mismatch: "
                f"expected CxHxW={self._channels}x{self._input_height}x{self._input_width}, "
                f"got {int(sequence.shape[1])}x{int(sequence.shape[2])}x{int(sequence.shape[3])}"
            )

        input_tensor = np.ascontiguousarray(sequence[None, ...])
        outputs = self._session.run([self._output_name], {self._input_name: input_tensor})
        if not outputs:
            raise RuntimeError("ONNX Runtime returned empty outputs for temporal wave model.")
        raw = np.asarray(outputs[0], dtype=np.float32)
        flat = self._flatten_output(raw, np=np)
        scaled = flat * self.output_scale + self.output_bias
        return np.ascontiguousarray(scaled.astype(np.float32, copy=False))

    @staticmethod
    def _extract_fixed_input_shape(shape: list[Any]) -> tuple[int, int, int, int] | None:
        if len(shape) != 5:
            raise ValueError(f"Temporal wave input must be rank-5 [B,S,C,H,W], got {shape!r}")
        batch = shape[0]
        sequence = shape[1]
        channels = shape[2]
        height = shape[3]
        width = shape[4]
        if isinstance(batch, int) and batch not in (0, 1):
            raise ValueError(f"Temporal wave batch dimension must be 1 (or dynamic), got {batch}")
        if not isinstance(sequence, int) or sequence <= 0:
            return None
        if not isinstance(channels, int) or channels <= 0:
            return None
        if not isinstance(height, int) or height <= 0:
            return None
        if not isinstance(width, int) or width <= 0:
            return None
        return int(sequence), int(channels), int(height), int(width)

    @staticmethod
    def _extract_output_length(shape: list[Any]) -> int | None:
        if not shape:
            return 1
        if len(shape) == 1:
            dim0 = shape[0]
            if isinstance(dim0, int) and dim0 > 0:
                return int(dim0)
            return None
        if len(shape) == 2:
            batch = shape[0]
            out_dim = shape[1]
            if isinstance(batch, int) and batch not in (0, 1):
                raise ValueError(f"Temporal wave output batch must be 1 (or dynamic), got {batch}")
            if isinstance(out_dim, int) and out_dim > 0:
                return int(out_dim)
            return None
        raise ValueError(f"Unsupported temporal wave output shape: {shape!r}")

    @staticmethod
    def _flatten_output(raw: Any, *, np: Any) -> Any:
        if raw.ndim == 0:
            return np.asarray([float(raw)], dtype=np.float32)
        if raw.ndim == 1:
            return raw.astype(np.float32, copy=False)
        if raw.ndim == 2:
            if int(raw.shape[0]) != 1:
                raise ValueError(f"Temporal wave output batch must be 1, got {raw.shape!r}")
            return raw[0].astype(np.float32, copy=False)
        raise ValueError(f"Unsupported temporal wave output ndim: {raw.ndim}")
