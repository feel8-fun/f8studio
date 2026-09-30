from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from f8pysdk.codec import coerce_float, coerce_int, parse_str_list


_MODEL_INDEX_LOAD_ERRORS = (Exception,)

FlowInputOrder = Literal["prev_now", "now_prev"]
TemporalResizeMode = Literal["direct_resize"]
TemporalNormalization = Literal["imagenet"]
ModelTask = Literal[
    "yolo_det",
    "yolo_pose",
    "yolo_obb",
    "yolo_cls",
    "optflow_neuflowv2",
    "tcn_wave",
    "yowo_temporal_det",
]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    provider: str
    task: ModelTask
    onnx_path: Path
    input_width: int
    input_height: int
    conf_threshold: float
    iou_threshold: float
    classes: list[str]
    skeleton_protocol: str = "none"
    keypoints: list[str] | None = None
    keypoint_dims: int = 3
    top_k: int = 5
    flow_format: str = "flow2_f16"
    flow_input_order: FlowInputOrder = "prev_now"
    temporal_clip_length: int = 16
    temporal_sampling_rate: int = 1
    temporal_max_det: int = 300
    temporal_resize_mode: TemporalResizeMode = "direct_resize"
    temporal_normalization: TemporalNormalization = "imagenet"
    onnx_url: str = ""
    onnx_sha256: str = ""
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class ModelIndexItem:
    model_id: str
    display_name: str
    task: ModelTask
    yaml_path: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Missing dependency 'pyyaml'.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid model yaml: {path}")
    return data


def _as_str(v: Any, *, default: str = "") -> str:
    try:
        s = str(v) if v is not None else ""
    except (TypeError, ValueError):
        s = ""
    s = s.strip()
    return s if s else default



def _parse_task(v: Any) -> ModelTask | None:
    s = _as_str(v).lower()
    if s in ("yolo_det", "det", "detect", "yolo_detect"):
        return "yolo_det"
    if s in ("yolo_pose", "pose", "keypoint", "keypoints", "kpt"):
        return "yolo_pose"
    if s in ("yolo_obb", "obb", "oriented_bbox", "rotated", "rotated_bbox"):
        return "yolo_obb"
    if s in ("yolo_cls", "cls", "classify", "classification", "classifier"):
        return "yolo_cls"
    if s in ("optflow_neuflowv2", "optflow", "optical_flow", "neuflowv2"):
        return "optflow_neuflowv2"
    if s in ("tcn_wave", "tcn", "temporal_wave", "temporal_conv", "conv_tcn"):
        return "tcn_wave"
    if s in ("yowo_temporal_det", "yowov3_temporal", "yowo_temporal", "temporal_det", "temporal_detector"):
        return "yowo_temporal_det"
    return None


def _parse_flow_input_order(v: Any) -> FlowInputOrder:
    s = _as_str(v, default="prev_now").lower()
    if s in ("prev_now", "prev->now", "previous_current", "previous_now"):
        return "prev_now"
    if s in ("now_prev", "now->prev", "current_previous", "now_previous"):
        return "now_prev"
    raise ValueError(f"Unsupported optflow inputOrder: {s!r}")


def _parse_temporal_resize_mode(v: Any) -> TemporalResizeMode:
    s = _as_str(v, default="direct_resize").lower()
    if s in ("direct_resize", "resize", "stretch"):
        return "direct_resize"
    raise ValueError(f"Unsupported temporal resizeMode: {s!r}")


def _parse_temporal_normalization(v: Any) -> TemporalNormalization:
    s = _as_str(v, default="imagenet").lower()
    if s in ("imagenet", "image_net"):
        return "imagenet"
    raise ValueError(f"Unsupported temporal normalization: {s!r}")


def _normalize_optional_sha256(v: Any) -> str:
    s = _as_str(v).strip().lower()
    if not s:
        return ""
    if len(s) != 64:
        raise ValueError(f"Invalid onnxSHA256 length: {len(s)} (expected 64)")
    for ch in s:
        if ch not in "0123456789abcdef":
            raise ValueError(f"Invalid onnxSHA256 character: {ch!r}")
    return s



def _normalize_skeleton_protocol(v: Any) -> str:
    text = _as_str(v).strip()
    if not text:
        return "none"
    return text


def _as_positive_int(v: Any, *, default: int, label: str, yaml_path: Path) -> int:
    out = coerce_int(v, default=default)
    if out <= 0:
        raise ValueError(f"Invalid {label} in {yaml_path}")
    return out


def load_model_spec(yaml_path: Path) -> ModelSpec:
    """
    Load an `f8onnxModel/1` model yaml.
    """
    yaml_path = Path(yaml_path).resolve()
    data = _load_yaml(yaml_path)

    schema = _as_str(data.get("schemaVersion"))
    if schema != "f8onnxModel/1":
        raise ValueError(f"Unsupported model schemaVersion in {yaml_path}: {schema!r}")

    model_value = data.get("model")
    model = model_value if isinstance(model_value, dict) else {}
    thresholds_value = data.get("thresholds")
    thresholds = thresholds_value if isinstance(thresholds_value, dict) else {}
    inp_value = data.get("input")
    inp = inp_value if isinstance(inp_value, dict) else {}
    labels_value = data.get("labels")
    labels = labels_value if isinstance(labels_value, dict) else {}
    pose_value = data.get("pose")
    pose = pose_value if isinstance(pose_value, dict) else {}
    classification_value = data.get("classification")
    classification = classification_value if isinstance(classification_value, dict) else {}
    optflow_value = data.get("optflow")
    optflow = optflow_value if isinstance(optflow_value, dict) else {}
    temporal_value = data.get("temporal")
    temporal = temporal_value if isinstance(temporal_value, dict) else {}

    task_value = _as_str(model.get("task"))
    task = _parse_task(task_value)
    if task is None or task_value != task:
        raise ValueError(f"Unsupported model.task in {yaml_path}: {task_value!r}")

    skeleton_protocol = _normalize_skeleton_protocol(model.get("skeletonProtocol"))
    model_id = _as_str(model.get("id"), default=yaml_path.stem)
    display_name = _as_str(model.get("displayName"), default=model_id)
    provider = _as_str(model.get("provider"), default="")
    onnx_rel = _as_str(model.get("onnxPath"))
    if not onnx_rel:
        raise ValueError(f"Missing model.onnxPath in {yaml_path}")
    onnx_path = (yaml_path.parent / onnx_rel).resolve() if not Path(onnx_rel).is_absolute() else Path(onnx_rel)
    onnx_url = _as_str(model.get("onnxUrl"))
    onnx_sha256 = _normalize_optional_sha256(model.get("onnxSHA256"))

    input_width = coerce_int(inp.get("width"), default=0)
    input_height = coerce_int(inp.get("height"), default=0)
    if input_width <= 0 or input_height <= 0:
        raise ValueError(f"Invalid input.width/input.height in {yaml_path}")

    conf_threshold = coerce_float(thresholds.get("conf"), default=0.25)
    iou_threshold = coerce_float(thresholds.get("iou"), default=0.45)
    classes = parse_str_list(labels.get("classes"), allow_mapping_values=True) or []
    keypoints = parse_str_list(pose.get("keypoints"), allow_mapping_values=True)
    keypoint_dims = coerce_int(pose.get("dims"), default=3)
    top_k = coerce_int(classification.get("topK"), default=5)
    flow_format = _as_str(optflow.get("flowFormat"), default="flow2_f16").lower()
    flow_input_order: FlowInputOrder = "prev_now"
    temporal_clip_length = coerce_int(temporal.get("clipLength"), default=16)
    temporal_sampling_rate = coerce_int(temporal.get("samplingRate"), default=1)
    temporal_max_det = coerce_int(temporal.get("maxDet"), default=300)
    temporal_resize_mode = _parse_temporal_resize_mode(temporal.get("resizeMode"))
    temporal_normalization = _parse_temporal_normalization(temporal.get("normalization"))

    if task == "optflow_neuflowv2":
        flow_input_order = _parse_flow_input_order(optflow.get("inputOrder"))
        if flow_format != "flow2_f16":
            raise ValueError(f"optflow.flowFormat must be 'flow2_f16' in {yaml_path}")
        if onnx_path.suffix.lower() != ".onnx":
            raise ValueError(f"Optflow model file must be .onnx in {yaml_path}")
    if task == "tcn_wave" and onnx_path.suffix.lower() != ".onnx":
        raise ValueError(f"TCN wave model file must be .onnx in {yaml_path}")
    if task == "yowo_temporal_det":
        if onnx_path.suffix.lower() != ".onnx":
            raise ValueError(f"Temporal detector model file must be .onnx in {yaml_path}")
        temporal_clip_length = _as_positive_int(
            temporal_clip_length,
            default=16,
            label="temporal.clipLength",
            yaml_path=yaml_path,
        )
        temporal_sampling_rate = _as_positive_int(
            temporal_sampling_rate,
            default=1,
            label="temporal.samplingRate",
            yaml_path=yaml_path,
        )
        temporal_max_det = _as_positive_int(
            temporal_max_det,
            default=300,
            label="temporal.maxDet",
            yaml_path=yaml_path,
        )

    meta_value = data.get("meta")
    meta = meta_value if isinstance(meta_value, dict) else {}
    return ModelSpec(
        model_id=model_id,
        display_name=display_name,
        provider=provider,
        task=task,
        onnx_path=onnx_path,
        onnx_url=onnx_url,
        onnx_sha256=onnx_sha256,
        input_width=input_width,
        input_height=input_height,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes=classes,
        skeleton_protocol=skeleton_protocol,
        keypoints=keypoints,
        keypoint_dims=max(1, int(keypoint_dims)),
        top_k=max(1, int(top_k)),
        flow_format=flow_format,
        flow_input_order=flow_input_order,
        temporal_clip_length=int(temporal_clip_length),
        temporal_sampling_rate=int(temporal_sampling_rate),
        temporal_max_det=int(temporal_max_det),
        temporal_resize_mode=temporal_resize_mode,
        temporal_normalization=temporal_normalization,
        meta=dict(meta),
    )


def discover_model_yamls(weights_dir: Path) -> list[Path]:
    d = Path(weights_dir)
    if not d.exists() or not d.is_dir():
        return []
    return sorted([*d.glob("*.yaml"), *d.glob("*.yml")])


def build_model_index_with_errors(
    weights_dir: Path,
    *,
    allowed_tasks: set[ModelTask] | None = None,
) -> tuple[list[ModelIndexItem], list[dict[str, str]]]:
    items: list[ModelIndexItem] = []
    errors: list[dict[str, str]] = []
    for y in discover_model_yamls(weights_dir):
        try:
            spec = load_model_spec(y)
        except _MODEL_INDEX_LOAD_ERRORS as exc:
            errors.append(
                {
                    "path": str(y.resolve()),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if allowed_tasks is not None and spec.task not in allowed_tasks:
            continue
        items.append(
            ModelIndexItem(
                model_id=spec.model_id,
                display_name=spec.display_name,
                task=spec.task,
                yaml_path=y,
            )
        )
    return items, errors


def build_model_index(weights_dir: Path, *, allowed_tasks: set[ModelTask] | None = None) -> list[ModelIndexItem]:
    items, _errors = build_model_index_with_errors(weights_dir, allowed_tasks=allowed_tasks)
    return items
