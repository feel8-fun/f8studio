from __future__ import annotations

from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8RuntimeNode,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from .constants import POSE_SERVICE_CLASS
from .service_node import MediaPipePoseServiceNode


def _state_fields() -> list[F8StateSpec]:
    return [
        F8StateSpec(
            name="shmName",
            label="Video SHM",
            description="Video SHM mapping name (e.g. shm.implayer.video).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="inferEveryN",
            label="Infer Every N Frames",
            description="Run pose inference every N frames (>=1).",
            valueSchema=integer_schema(default=1, minimum=1, maximum=10000),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="modelComplexity",
            label="Model Complexity",
            description="MediaPipe pose model variant.",
            valueSchema=string_schema(default="full", enum=["lite", "full", "heavy"]),
            access=F8StateAccess.rw,
            uiControl="select",
            showOnNode=True,
        ),
        F8StateSpec(
            name="minDetectionConfidence",
            label="Min Detection Confidence",
            description="Minimum confidence threshold for pose detection.",
            valueSchema=number_schema(default=0.5, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="minTrackingConfidence",
            label="Min Tracking Confidence",
            description="Minimum confidence threshold for pose tracking.",
            valueSchema=number_schema(default=0.5, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="visibilityThreshold",
            label="Visibility Threshold",
            description="Landmark visibility threshold (below threshold => hidden point).",
            valueSchema=number_schema(default=0.5, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="lastError",
            label="Last Error",
            description="Last runtime error string (best-effort).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            showOnNode=False,
        ),
        F8StateSpec(
            name="telemetryIntervalMs",
            label="Telemetry Interval (ms)",
            description="Emit telemetry summaries every N milliseconds (0 disables).",
            valueSchema=integer_schema(default=1000, minimum=0, maximum=60000),
            access=F8StateAccess.wo,
            showOnNode=False,
        ),
        F8StateSpec(
            name="telemetryWindowMs",
            label="Telemetry Window (ms)",
            description="Rolling window for telemetry averages (ms).",
            valueSchema=integer_schema(default=2000, minimum=100, maximum=60000),
            access=F8StateAccess.wo,
            showOnNode=False,
        ),
    ]


def register_specs(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()
    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=POSE_SERVICE_CLASS,
            version="0.0.1",
            label="MediaPipe Pose",
            description="MediaPipe single-person pose extraction service (33 landmarks).",
            tags=["mediapipe", "vision", "human", "pose"],
            rendererClass="default_svc",
            stateFields=_state_fields(),
            dataOutPorts=[
                F8DataPortSpec(
                    name="detections",
                    description="Detection output in schema f8visionDetections/1.",
                    valueSchema=any_schema(),
                ),
                F8DataPortSpec(
                    name="skeletons",
                    description="List of UDP-skeleton-compatible JSON payloads for skeleton3d.",
                    valueSchema=any_schema(),
                ),
                F8DataPortSpec(
                    name="telemetry",
                    description="Periodic telemetry summaries (fps + timings).",
                    valueSchema=any_schema(),
                ),
            ],
            editableStateFields=False,
            editableDataInPorts=False,
            editableDataOutPorts=False,
            editableCommands=False,
        ),
        overwrite=True,
    )

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return MediaPipePoseServiceNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register_service(POSE_SERVICE_CLASS, _factory, overwrite=True)
    return reg
