from __future__ import annotations

from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSchemaVersion,
    F8OperatorSpec,
    F8RuntimeNode,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    boolean_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from .constants import SERVICE_CLASS
from .detecttracker_node import detecttrackerServiceNode


def register_detecttracker_specs(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    # Single-node service design:
    # - service node (nodeId == serviceId) owns the data output + state
    # - no separate operator spec is required for the basic "external source service" use case
    state_fields = [
        F8StateSpec(
            name="sourceServiceId",
            label="Source Service Id",
            description="If set and shmName is empty, uses shm.<sourceServiceId>.video",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="shmName",
            label="Video SHM Name",
            description="Video SHM mapping name (e.g. shm.implayer.video). Overrides sourceServiceId.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="weightsDir",
            label="Weights Dir",
            description="Directory containing *.yaml + *.onnx pairs (defaults to services/f8/detect_tracker/weights).",
            valueSchema=string_schema(default="services/f8/detect_tracker/weights"),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="modelId",
            label="Model Id",
            description="Model id selected from weightsDir (ignored if modelYamlPath is set).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="modelYamlPath",
            label="Model YAML Path",
            description="Optional explicit model yaml path (overrides modelId).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="ortProvider",
            label="ONNX Runtime Provider",
            description="auto prefers CUDAExecutionProvider when available.",
            valueSchema=string_schema(default="auto", enum=["auto", "cuda", "cpu"]),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="trackerKind",
            label="Tracker",
            description="High-frequency CV tracker between detection frames.",
            valueSchema=string_schema(default="kcf", enum=["none", "kcf", "csrt", "mosse"]),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="detectEveryN",
            label="Detect Every N Frames",
            description="Run ONNX inference every N frames (>=1).",
            valueSchema=integer_schema(default=5, minimum=1, maximum=10000),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="maxTargets",
            label="Max Targets",
            description="Maximum tracked objects to keep.",
            valueSchema=integer_schema(default=5, minimum=1, maximum=1000),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="iouMatch",
            label="IoU Match",
            description="Detection-to-track matching threshold (IoU).",
            valueSchema=number_schema(default=0.3, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="mismatchIou",
            label="Mismatch IoU",
            description="If matched but IoU < mismatchIou, increment mismatch.",
            valueSchema=number_schema(default=0.2, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="mismatchPatience",
            label="Mismatch Patience",
            description="Drop track after N mismatches.",
            valueSchema=integer_schema(default=3, minimum=1, maximum=1000),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="maxAge",
            label="Max Age (frames)",
            description="Drop track after maxAge frames without a good update.",
            valueSchema=integer_schema(default=30, minimum=1, maximum=100000),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="reinitOnDetect",
            label="Reinit Tracker On Detect",
            description="Reinitialize the CV tracker on each detection match.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="confThreshold",
            label="Conf Threshold Override",
            description="Override conf threshold (negative uses model yaml).",
            valueSchema=number_schema(default=-1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="iouThreshold",
            label="IoU Threshold Override",
            description="Override IoU threshold for NMS (negative uses model yaml).",
            valueSchema=number_schema(default=-1.0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        # "Status" fields (kept rw so runtime can update via normal set_state).
        F8StateSpec(
            name="availableModels",
            label="Available Models",
            description="JSON list of models discovered from weightsDir.",
            valueSchema=string_schema(default="[]"),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="loadedModel",
            label="Loaded Model",
            description="Current loaded model id/task.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="ortActiveProviders",
            label="ORT Active Providers",
            description="JSON list of active ONNX Runtime providers for this session.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="lastError",
            label="Last Error",
            description="Last runtime error string (best-effort).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="lastFrameId",
            label="Last Frame Id",
            description="Last processed frame id from VideoSHM.",
            valueSchema=integer_schema(default=0, minimum=0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="lastFrameTsMs",
            label="Last Frame Ts (ms)",
            description="Last processed frame timestamp (ms) from VideoSHM.",
            valueSchema=integer_schema(default=0, minimum=0),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="telemetryIntervalMs",
            label="Telemetry Interval (ms)",
            description="Emit telemetry summaries every N milliseconds (0 disables).",
            valueSchema=integer_schema(default=1000, minimum=0, maximum=60000),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="telemetryWindowMs",
            label="Telemetry Window (ms)",
            description="Rolling window for telemetry averages (ms).",
            valueSchema=integer_schema(default=2000, minimum=100, maximum=60000),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
    ]

    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=SERVICE_CLASS,
            version="0.0.1",
            label="Detect Tracker",
            description="ONNXRuntime-based detector + tracker service (single-node source).",
            tags=["onnx", "vision", "tracker"],
            rendererClass="default_svc",
            stateFields=state_fields,
            dataOutPorts=[
                F8DataPortSpec(
                    name="detections",
                    description="Stream of per-frame detections/tracks.",
                    valueSchema=any_schema(),
                ),
                F8DataPortSpec(
                    name="telemetry",
                    description="Periodic telemetry summaries (fps + timings).",
                    valueSchema=any_schema(),
                )
            ],
            editableStateFields=False,
            editableDataInPorts=False,
            editableDataOutPorts=False,
            editableCommands=False,
        ),
        overwrite=True,
    )

    def _service_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return detecttrackerServiceNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register_service(SERVICE_CLASS, _service_factory, overwrite=True)
    return reg
