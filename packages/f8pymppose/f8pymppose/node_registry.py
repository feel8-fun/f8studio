from __future__ import annotations

from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
    F8DataPortSpec,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    array_schema,
    any_schema,
    complex_object_schema,
    integer_schema,
    number_schema,
    string_schema,
    video_frame_port,
)
from f8pysdk.registry import Registry, RuntimeNodeRegistry, create_runtime_node_registry, shared_runtime_node_registry

from .config import (
    DEFAULT_INFER_EVERY_N,
    DEFAULT_MIN_DETECTION_CONFIDENCE,
    DEFAULT_MIN_TRACKING_CONFIDENCE,
    DEFAULT_MODEL_COMPLEXITY,
    DEFAULT_SKELETON_SOURCE,
    DEFAULT_VISIBILITY_THRESHOLD,
)
from .constants import POSE_SERVICE_CLASS
from .service_node import MediaPipePoseServiceNode


def _keypoint_schema():
    return complex_object_schema(
        properties={
            "x": any_schema(),
            "y": any_schema(),
            "z": any_schema(),
            "score": number_schema(),
        }
    )


def _detection_schema():
    return complex_object_schema(
        properties={
            "id": integer_schema(),
            "cls": string_schema(),
            "score": number_schema(),
            "bbox": array_schema(items=integer_schema()),
            "keypoints": array_schema(items=_keypoint_schema()),
            "skeletonProtocol": string_schema(),
        }
    )


def _detections_payload_schema():
    return complex_object_schema(
        properties={
            "schemaVersion": string_schema(),
            "frameId": integer_schema(),
            "tsMs": integer_schema(),
            "width": integer_schema(),
            "height": integer_schema(),
            "model": string_schema(),
            "task": string_schema(),
            "skeletonProtocol": string_schema(),
            "detections": array_schema(items=_detection_schema()),
        }
    )


def _skeleton_bone_schema():
    return complex_object_schema(
        properties={
            "name": string_schema(),
            "pos": array_schema(items=number_schema()),
            "rot": array_schema(items=number_schema()),
        }
    )


def _skeleton_payload_schema():
    return complex_object_schema(
        properties={
            "type": string_schema(),
            "schema": string_schema(),
            "modelName": string_schema(),
            "name": string_schema(),
            "timestampMs": integer_schema(),
            "frameId": integer_schema(),
            "boneCount": integer_schema(),
            "bones": array_schema(items=_skeleton_bone_schema()),
            "trailer": any_schema(),
            "skeletonProtocol": string_schema(),
        }
    )


def _state_fields() -> list[F8StateSpec]:
    return [
        F8StateSpec(
            name="inferEveryN",
            label="Infer Every N Frames",
            description="Run pose inference every N frames (>=1).",
            valueSchema=integer_schema(default=DEFAULT_INFER_EVERY_N, minimum=1, maximum=10000),
            access=F8StateAccess.rw,
            showOnNode=False,
            valueRequired=True
        ),
        F8StateSpec(
            name="modelComplexity",
            label="Model Complexity",
            description="MediaPipe pose model variant.",
            valueSchema=string_schema(default=DEFAULT_MODEL_COMPLEXITY, enum=["lite", "full", "heavy"]),
            access=F8StateAccess.rw,
            valueRequired=True,
            control=F8UiControlSpec(kind=F8UiControlKind.select),
            showOnNode=False,
        ),
        F8StateSpec(
            name="minDetectionConfidence",
            label="Min Detection Confidence",
            description="Minimum confidence threshold for pose detection.",
            valueSchema=number_schema(default=DEFAULT_MIN_DETECTION_CONFIDENCE, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="minTrackingConfidence",
            label="Min Tracking Confidence",
            description="Minimum confidence threshold for pose tracking.",
            valueSchema=number_schema(default=DEFAULT_MIN_TRACKING_CONFIDENCE, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="visibilityThreshold",
            label="Visibility Threshold",
            description="Landmark visibility threshold (below threshold => hidden point).",
            valueSchema=number_schema(default=DEFAULT_VISIBILITY_THRESHOLD, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            valueRequired=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="skeletonSource",
            label="Skeleton Source",
            description="Skeleton data source (camera-relative vs world-relative).",
            valueSchema=string_schema(default=DEFAULT_SKELETON_SOURCE, enum=["camera", "world"]),
            access=F8StateAccess.rw,
            valueRequired=True,
            control=F8UiControlSpec(kind=F8UiControlKind.select),
            showOnNode=False,
        ),
    ]


def register_specs(registry: Registry) -> Registry:
    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=POSE_SERVICE_CLASS,
            paletteCategory="svc",
            version="0.0.1",
            label="MediaPipe Pose",
            description="MediaPipe single-person pose extraction service (33 landmarks).",
            tags=["mediapipe", "vision", "human", "pose"],
            rendererClass="default_svc",
            stateFields=_state_fields(),
            dataInPorts=[
                video_frame_port(
                    name="video",
                    description="Input video frame stream.",
                    definition_protected=True,
                ),
            ],
            dataOutPorts=[
                F8DataPortSpec(
                    name="detections",
                    description="Detection output in schema f8visionDetections/1.",
                    valueSchema=_detections_payload_schema(),
                ),
                F8DataPortSpec(
                    name="skeletons",
                    description="List of UDP-skeleton-compatible JSON payloads for skeleton3d.",
                    valueSchema=array_schema(items=_skeleton_payload_schema()),
                ),
            ],
        ),
        MediaPipePoseServiceNode,
        overwrite=True,
    )
    return registry


def create_mppose_registry() -> RuntimeNodeRegistry:
    runtime_registry = create_runtime_node_registry()
    register_specs(Registry.wrap(runtime_registry))
    return runtime_registry


def shared_mppose_registry() -> RuntimeNodeRegistry:
    runtime_registry = shared_runtime_node_registry()
    register_specs(Registry.wrap(runtime_registry))
    return runtime_registry
