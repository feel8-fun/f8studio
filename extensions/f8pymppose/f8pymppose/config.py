from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from f8pysdk.codec import coerce_float as coerce_float, coerce_int as coerce_int, coerce_str

SkeletonSource = Literal["camera", "world"]
ModelComplexity = Literal["lite", "full", "heavy"]

DEFAULT_INFER_EVERY_N = 1
DEFAULT_MODEL_COMPLEXITY: ModelComplexity = "full"
DEFAULT_MIN_DETECTION_CONFIDENCE = 0.5
DEFAULT_MIN_TRACKING_CONFIDENCE = 0.5
DEFAULT_VISIBILITY_THRESHOLD = 0.5
DEFAULT_SKELETON_SOURCE: SkeletonSource = "camera"


def coerce_skeleton_source(v: Any, *, default: SkeletonSource = DEFAULT_SKELETON_SOURCE) -> SkeletonSource:
    text = coerce_str(v, default=default).lower()
    if text == "camera":
        return "camera"
    if text == "world":
        return "world"
    return default


def coerce_model_complexity(
    v: Any, *, default: ModelComplexity = DEFAULT_MODEL_COMPLEXITY
) -> ModelComplexity:
    text = coerce_str(v, default=default).lower()
    if text == "lite":
        return "lite"
    if text == "full":
        return "full"
    if text == "heavy":
        return "heavy"
    return default


def state_or_default(state_value: Any, initial_value: Any, *, default: Any) -> Any:
    if state_value is not None:
        return state_value
    if initial_value is not None:
        return initial_value
    return default


@dataclass(frozen=True)
class PoseServiceConfig:
    infer_every_n: int = DEFAULT_INFER_EVERY_N
    model_complexity: ModelComplexity = DEFAULT_MODEL_COMPLEXITY
    min_detection_confidence: float = DEFAULT_MIN_DETECTION_CONFIDENCE
    min_tracking_confidence: float = DEFAULT_MIN_TRACKING_CONFIDENCE
    visibility_threshold: float = DEFAULT_VISIBILITY_THRESHOLD
    skeleton_source: SkeletonSource = DEFAULT_SKELETON_SOURCE
