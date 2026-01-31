from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


ModelTask = Literal["yolo_det", "yolo_pose", "yolo_obb"]


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
    keypoints: list[str] | None = None
    keypoint_dims: int = 3
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
    except Exception as exc:
        raise RuntimeError("Missing dependency 'pyyaml'.") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid model yaml: {path}")
    return data


def _as_str(v: Any, *, default: str = "") -> str:
    try:
        s = str(v) if v is not None else ""
    except Exception:
        s = ""
    s = s.strip()
    return s if s else default


def _as_int(v: Any, *, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _as_float(v: Any, *, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _parse_task(v: Any) -> ModelTask | None:
    s = _as_str(v).lower()
    if s in ("yolo_det", "det", "detect", "yolo_detect"):
        return "yolo_det"
    if s in ("yolo_pose", "pose", "keypoint", "keypoints", "kpt"):
        return "yolo_pose"
    if s in ("yolo_obb", "obb", "oriented_bbox", "rotated", "rotated_bbox"):
        return "yolo_obb"
    return None


def _coerce_str_list(v: Any) -> list[str] | None:
    if isinstance(v, (list, tuple)):
        out: list[str] = []
        for x in v:
            s = _as_str(x)
            if s:
                out.append(s)
        return out
    return None


def load_model_spec(yaml_path: Path) -> ModelSpec:
    """
    Load a model yaml.

    Supports:
    - legacy format (type/name/provider/model_path/input_width/input_height/classes/conf_threshold/iou_threshold)
    - f8onnxModel/1 format (schemaVersion + nested fields)
    """
    yaml_path = Path(yaml_path).resolve()
    data = _load_yaml(yaml_path)

    schema = _as_str(data.get("schemaVersion"))
    if schema == "f8onnxModel/1":
        model = data.get("model") if isinstance(data.get("model"), dict) else {}
        thresholds = data.get("thresholds") if isinstance(data.get("thresholds"), dict) else {}
        inp = data.get("input") if isinstance(data.get("input"), dict) else {}
        labels = data.get("labels") if isinstance(data.get("labels"), dict) else {}
        pose = data.get("pose") if isinstance(data.get("pose"), dict) else {}

        task = _parse_task(model.get("task")) or "yolo_det"
        model_id = _as_str(model.get("id"), default=_as_str(data.get("name"), default=yaml_path.stem))
        display_name = _as_str(model.get("displayName"), default=model_id)
        provider = _as_str(model.get("provider"), default=_as_str(data.get("provider"), default=""))
        onnx_rel = _as_str(model.get("onnxPath"), default=_as_str(data.get("model_path"), default=""))
        if not onnx_rel:
            raise ValueError(f"Missing model.onnxPath in {yaml_path}")
        onnx_path = (yaml_path.parent / onnx_rel).resolve() if not Path(onnx_rel).is_absolute() else Path(onnx_rel)

        input_width = _as_int(inp.get("width"), default=_as_int(data.get("input_width"), default=0))
        input_height = _as_int(inp.get("height"), default=_as_int(data.get("input_height"), default=0))
        if input_width <= 0 or input_height <= 0:
            raise ValueError(f"Invalid input size in {yaml_path}")

        conf_threshold = _as_float(thresholds.get("conf"), default=_as_float(data.get("conf_threshold"), default=0.25))
        iou_threshold = _as_float(thresholds.get("iou"), default=_as_float(data.get("iou_threshold"), default=0.45))

        classes = _coerce_str_list(labels.get("classes")) or _coerce_str_list(data.get("classes")) or []
        keypoints = _coerce_str_list(pose.get("keypoints"))
        keypoint_dims = _as_int(pose.get("dims"), default=3)

        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        return ModelSpec(
            model_id=model_id,
            display_name=display_name,
            provider=provider,
            task=task,
            onnx_path=onnx_path,
            input_width=input_width,
            input_height=input_height,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            classes=classes,
            keypoints=keypoints,
            keypoint_dims=max(1, int(keypoint_dims)),
            meta=dict(meta),
        )

    # ---- legacy ---------------------------------------------------------
    model_id = _as_str(data.get("name"), default=yaml_path.stem)
    display_name = _as_str(data.get("display_name"), default=model_id)
    provider = _as_str(data.get("provider"), default="")

    # Try to infer task from explicit field, type, or filename.
    task = _parse_task(data.get("task")) or _parse_task(data.get("type"))
    if task is None:
        name_hint = f"{yaml_path.stem} {model_id} {display_name}".lower()
        task = "yolo_pose" if "pose" in name_hint or "kpt" in name_hint else "yolo_det"

    onnx_rel = _as_str(data.get("model_path"), default=f"{yaml_path.stem}.onnx")
    onnx_path = (yaml_path.parent / onnx_rel).resolve() if not Path(onnx_rel).is_absolute() else Path(onnx_rel)

    input_width = _as_int(data.get("input_width"), default=0)
    input_height = _as_int(data.get("input_height"), default=0)
    if input_width <= 0 or input_height <= 0:
        raise ValueError(f"Invalid input_width/input_height in {yaml_path}")

    conf_threshold = _as_float(data.get("conf_threshold"), default=0.25)
    iou_threshold = _as_float(data.get("iou_threshold"), default=0.45)

    classes = _coerce_str_list(data.get("classes")) or []
    keypoints = _coerce_str_list(data.get("keypoints"))
    keypoint_dims = _as_int(data.get("keypoint_dims"), default=3)

    return ModelSpec(
        model_id=model_id,
        display_name=display_name,
        provider=provider,
        task=task,
        onnx_path=onnx_path,
        input_width=input_width,
        input_height=input_height,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        classes=classes,
        keypoints=keypoints,
        keypoint_dims=max(1, int(keypoint_dims)),
        meta={},
    )


def discover_model_yamls(weights_dir: Path) -> list[Path]:
    d = Path(weights_dir)
    if not d.exists() or not d.is_dir():
        return []
    return sorted([*d.glob("*.yaml"), *d.glob("*.yml")])


def build_model_index(weights_dir: Path) -> list[ModelIndexItem]:
    items: list[ModelIndexItem] = []
    for y in discover_model_yamls(weights_dir):
        try:
            spec = load_model_spec(y)
        except Exception:
            continue
        items.append(ModelIndexItem(model_id=spec.model_id, display_name=spec.display_name, task=spec.task, yaml_path=y))
    return items

