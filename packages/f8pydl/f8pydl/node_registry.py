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
    array_schema,
    boolean_schema,
    integer_schema,
    number_schema,
    string_schema,
)
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from .constants import CLASSIFIER_SERVICE_CLASS, DETECTOR_SERVICE_CLASS, HUMAN_DETECTOR_SERVICE_CLASS
from .constants import OPTFLOW_SERVICE_CLASS, TCNWAVE_SERVICE_CLASS
from .optflow_service_node import OnnxOptflowServiceNode
from .service_node import OnnxVisionServiceNode
from .tcnwave_service_node import OnnxTcnWaveServiceNode


def _common_state_fields(
    *,
    include_thresholds: bool,
    include_top_k: bool,
    include_class_filter: bool,
) -> list[F8StateSpec]:
    fields = [
        F8StateSpec(
            name="shmName",
            label="Video SHM",
            description="Video SHM mapping name (e.g. shm.implayer.video).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="weightsDir",
            label="Weights Dir",
            description="Directory containing *.yaml + *.onnx model files.",
            valueSchema=string_schema(default="services/f8/dl/weights"),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="modelId",
            label="Model Id",
            description="Model id selected from weightsDir (ignored if modelYamlPath is set).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            uiControl="select:[availableModels]",
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
            name="autoDownloadWeights",
            label="Auto Download Weights",
            description="When model file is missing, download from onnxUrl in model yaml.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="inferEveryN",
            label="Infer Every N Frames",
            description="Run model inference every N frames (>=1).",
            valueSchema=integer_schema(default=1, minimum=1, maximum=10000),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
    ]
    if include_thresholds:
        fields.extend(
            [
                F8StateSpec(
                    name="confThreshold",
                    label="Conf Threshold Override",
                    description="Override confidence threshold (negative uses model yaml).",
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
            ]
        )
    if include_top_k:
        fields.append(
            F8StateSpec(
                name="topK",
                label="Top K",
                description="Number of top classes to emit.",
                valueSchema=integer_schema(default=5, minimum=1, maximum=100),
                access=F8StateAccess.rw,
                showOnNode=True,
            )
        )
    if include_class_filter:
        fields.extend(
            [
                F8StateSpec(
                    name="enabledClasses",
                    label="Enabled Classes",
                    description="Optional class whitelist for output. Empty means all classes.",
                    valueSchema=array_schema(items=string_schema()),
                    access=F8StateAccess.rw,
                    uiControl="multiselect:[modelClasses]",
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="perClassK",
                    label="Per Class K",
                    description="Per-class top-K by score (<=0 means unlimited).",
                    valueSchema=integer_schema(default=0, minimum=0, maximum=10000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="modelClasses",
                    label="Model Classes",
                    description="Current loaded model class labels.",
                    valueSchema=array_schema(items=string_schema()),
                    access=F8StateAccess.ro,
                    showOnNode=False,
                )
            ]
        )
    fields.append(
        F8StateSpec(
            name="availableModels",
            label="Available Models",
            description="List of model ids discovered from weightsDir.",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            showOnNode=False,
        )
    )
    
    fields.extend(
        [
            F8StateSpec(
                name="loadedModel",
                label="Loaded Model",
                description="Current loaded model id/task.",
                valueSchema=string_schema(default=""),
                access=F8StateAccess.ro,
                showOnNode=False,
            ),
            F8StateSpec(
                name="ortActiveProviders",
                label="ORT Active Providers",
                description="JSON list of active ONNX Runtime providers for this session.",
                valueSchema=string_schema(default=""),
                access=F8StateAccess.ro,
                showOnNode=False,
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
    )
    return fields


def _optflow_state_fields() -> list[F8StateSpec]:
    return [
        F8StateSpec(
            name="inputShmName",
            label="Input Video SHM",
            description="Input SHM name (e.g. shm.xxx.video).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="computeEveryNFrames",
            label="Compute Every N Frames",
            description="Compute optical flow once per N new frames.",
            valueSchema=integer_schema(default=2, minimum=1, maximum=120),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="weightsDir",
            label="Weights Dir",
            description="Directory containing *.yaml + *.onnx model files.",
            valueSchema=string_schema(default="services/f8/dl/weights"),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="modelId",
            label="Model Id",
            description="Model id selected from weightsDir (ignored if modelYamlPath is set).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            uiControl="select:[availableModels]",
            showOnNode=False,
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
            showOnNode=False,
        ),
        F8StateSpec(
            name="autoDownloadWeights",
            label="Auto Download Weights",
            description="When model file is missing, download from onnxUrl in model yaml.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="availableModels",
            label="Available Models",
            description="List of model ids discovered from weightsDir.",
            valueSchema=array_schema(items=string_schema()),
            access=F8StateAccess.ro,
            showOnNode=False,
        ),
        F8StateSpec(
            name="loadedModel",
            label="Loaded Model",
            description="Current loaded model id/task.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            showOnNode=False,
        ),
        F8StateSpec(
            name="ortActiveProviders",
            label="ORT Active Providers",
            description="JSON list of active ONNX Runtime providers for this session.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            showOnNode=False,
        ),
        F8StateSpec(
            name="flowShmName",
            label="Flow SHM Name",
            description="Output flow SHM name.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            showOnNode=True,
        ),
        F8StateSpec(
            name="flowShmFormat",
            label="Flow SHM Format",
            description="Flow payload format. Fixed to flow2_f16.",
            valueSchema=string_schema(default="flow2_f16"),
            access=F8StateAccess.ro,
            showOnNode=False,
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


def _tcn_wave_state_fields() -> list[F8StateSpec]:
    fields = _common_state_fields(
        include_thresholds=False,
        include_top_k=False,
        include_class_filter=False,
    )
    fields.extend(
        [
            F8StateSpec(
                name="outputScale",
                label="Output Scale",
                description="Denormalization scale applied to raw model output values.",
                valueSchema=number_schema(default=10.0),
                access=F8StateAccess.rw,
                showOnNode=False,
            ),
            F8StateSpec(
                name="outputBias",
                label="Output Bias",
                description="Denormalization bias applied after outputScale.",
                valueSchema=number_schema(default=0.0),
                access=F8StateAccess.rw,
                showOnNode=False,
            ),
        ]
    )
    fields.append(
        F8StateSpec(
            name="useVrFocusCrop",
            label="VR Focus Crop",
            description=(
                "Apply focus crop before inference. "
                "This assumes SHM already provides the target eye view and crops top 20% + left/right 10%."
            ),
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            showOnNode=False,
        )
    )
    return fields


def _register_classifier(reg: RuntimeNodeRegistry) -> None:
    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=CLASSIFIER_SERVICE_CLASS,
            version="0.0.1",
            label="DL Classifier",
            description="ONNXRuntime image classifier service (no tracking).",
            tags=["onnx", "vision", "classification"],
            rendererClass="default_svc",
            stateFields=_common_state_fields(
                include_thresholds=False,
                include_top_k=True,
                include_class_filter=False,
            ),
            dataOutPorts=[
                F8DataPortSpec(
                    name="classifications",
                    description="Classification output in schema f8visionClassifications/1.",
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
        return OnnxVisionServiceNode(
            node_id=node_id,
            node=node,
            initial_state=initial_state,
            service_class=CLASSIFIER_SERVICE_CLASS,
            service_task="classifier",
            output_port="classifications",
            allowed_tasks={"yolo_cls"},
        )

    reg.register_service(CLASSIFIER_SERVICE_CLASS, _factory, overwrite=True)


def _register_detector(reg: RuntimeNodeRegistry) -> None:
    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=DETECTOR_SERVICE_CLASS,
            version="0.0.1",
            label="DL Detector",
            description="ONNXRuntime object detector service (no tracking).",
            tags=["onnx", "vision", "detection"],
            rendererClass="default_svc",
            stateFields=_common_state_fields(
                include_thresholds=True,
                include_top_k=False,
                include_class_filter=True,
            ),
            dataOutPorts=[
                F8DataPortSpec(
                    name="detections",
                    description="Detection output in schema f8visionDetections/1.",
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
        return OnnxVisionServiceNode(
            node_id=node_id,
            node=node,
            initial_state=initial_state,
            service_class=DETECTOR_SERVICE_CLASS,
            service_task="detector",
            output_port="detections",
            allowed_tasks={"yolo_det", "yolo_obb"},
        )

    reg.register_service(DETECTOR_SERVICE_CLASS, _factory, overwrite=True)


def _register_human_detector(reg: RuntimeNodeRegistry) -> None:
    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=HUMAN_DETECTOR_SERVICE_CLASS,
            version="0.0.1",
            label="DL Human Detector",
            description="ONNXRuntime human detection/pose service (no tracking).",
            tags=["onnx", "vision", "human", "pose"],
            rendererClass="default_svc",
            stateFields=_common_state_fields(
                include_thresholds=True,
                include_top_k=False,
                include_class_filter=True,
            ),
            dataOutPorts=[
                F8DataPortSpec(
                    name="detections",
                    description="Detection output in schema f8visionDetections/1.",
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
        return OnnxVisionServiceNode(
            node_id=node_id,
            node=node,
            initial_state=initial_state,
            service_class=HUMAN_DETECTOR_SERVICE_CLASS,
            service_task="humandetector",
            output_port="detections",
            allowed_tasks={"yolo_det", "yolo_pose"},
        )

    reg.register_service(HUMAN_DETECTOR_SERVICE_CLASS, _factory, overwrite=True)


def _register_optflow(reg: RuntimeNodeRegistry) -> None:
    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=OPTFLOW_SERVICE_CLASS,
            version="0.0.1",
            label="DL Optical Flow",
            description="ONNXRuntime NeuFlowV2 dense optical flow service (flow SHM output).",
            tags=["onnx", "vision", "optical_flow", "flow_shm"],
            rendererClass="default_svc",
            stateFields=_optflow_state_fields(),
            dataOutPorts=[
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
        return OnnxOptflowServiceNode(
            node_id=node_id,
            node=node,
            initial_state=initial_state,
            service_class=OPTFLOW_SERVICE_CLASS,
            allowed_tasks={"optflow_neuflowv2"},
        )

    reg.register_service(OPTFLOW_SERVICE_CLASS, _factory, overwrite=True)


def _register_tcn_wave(reg: RuntimeNodeRegistry) -> None:
    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=TCNWAVE_SERVICE_CLASS,
            version="0.0.1",
            label="DL TCN Wave",
            description="ONNXRuntime temporal convolution wave inference service (port output).",
            tags=["onnx", "vision", "temporal", "wave", "signal"],
            rendererClass="default_svc",
            stateFields=_tcn_wave_state_fields(),
            dataOutPorts=[
                F8DataPortSpec(
                    name="predictedChange",
                    description="Temporal model output value per frame.",
                    valueSchema=number_schema(),
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
        return OnnxTcnWaveServiceNode(
            node_id=node_id,
            node=node,
            initial_state=initial_state,
            service_class=TCNWAVE_SERVICE_CLASS,
            allowed_tasks={"tcn_wave"},
        )

    reg.register_service(TCNWAVE_SERVICE_CLASS, _factory, overwrite=True)


def register_specs(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()
    _register_classifier(reg)
    _register_detector(reg)
    _register_human_detector(reg)
    _register_optflow(reg)
    _register_tcn_wave(reg)
    return reg
