from __future__ import annotations

from typing import Any

from f8pysdk.specs import (
    F8UiControlKind,
    F8UiControlSpec,
    F8DataPortSpec,
    F8RuntimeNode,
    F8ServiceSchemaVersion,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    array_schema,
    boolean_schema,
    complex_object_schema,
    integer_schema,
    number_schema,
    string_schema,
    video_frame_port,
)
from f8pysdk.nodes import RuntimeNode
from f8pysdk.registry import Registry, RuntimeNodeRegistry, create_runtime_node_registry, shared_runtime_node_registry

from .constants import CLASSIFIER_SERVICE_CLASS, DETECTOR_SERVICE_CLASS, DETECTION_SORTER_SERVICE_CLASS, HUMAN_DETECTOR_SERVICE_CLASS
from .constants import OPTFLOW_SERVICE_CLASS, TCNWAVE_SERVICE_CLASS
from .detection_sorter_service_node import DetectionSorterServiceNode
from .optflow_service_node import OnnxOptflowServiceNode
from .service_node import OnnxVisionServiceNode
from .tcnwave_service_node import OnnxTcnWaveServiceNode


def _classification_item_schema():
    return complex_object_schema(
        properties={
            "cls": string_schema(),
            "score": number_schema(),
        }
    )


def _classifications_payload_schema():
    return complex_object_schema(
        properties={
            "schemaVersion": string_schema(),
            "frameId": integer_schema(),
            "tsMs": integer_schema(),
            "model": string_schema(),
            "top1": _classification_item_schema(),
            "topk": array_schema(items=_classification_item_schema()),
        }
    )


def _keypoint_schema():
    return complex_object_schema(
        properties={
            "x": number_schema(),
            "y": number_schema(),
            "score": number_schema(),
        }
    )


def _obb_point_schema():
    return array_schema(items=number_schema())


def _detection_item_schema():
    return complex_object_schema(
        properties={
            "cls": string_schema(),
            "score": number_schema(),
            "bbox": array_schema(items=integer_schema()),
            "keypoints": array_schema(items=_keypoint_schema()),
            "obb": array_schema(items=_obb_point_schema()),
            "skeletonProtocol": string_schema(),
        }
    )


def _detections_payload_schema():
    return complex_object_schema(
        properties={
            "schemaVersion": string_schema(),
            "frameId": integer_schema(),
            "tsMs": integer_schema(),
            "streamId": string_schema(),
            "streamEpoch": string_schema(),
            "width": integer_schema(),
            "height": integer_schema(),
            "model": string_schema(),
            "task": string_schema(),
            "skeletonProtocol": string_schema(),
            "detections": array_schema(items=_detection_item_schema()),
        }
    )


def _video_input_port() -> F8DataPortSpec:
    return video_frame_port(
        name="video",
        description="Input video frame stream.",
        required=True,
    )


def _detection_sorter_state_fields() -> list[F8StateSpec]:
    return [
        F8StateSpec(
            name="clsWeights",
            label="Class Weights",
            description=(
                "JSON map of detection cls -> weight multiplier applied to score-map metric."
                " Keys without a prefix are exact cls matches."
                " Keys with 're:' prefix are Python regex patterns matched via fullmatch()."
                " All matching rules are multiplied together."
                " Unspecified classes default to weight 1.0."
                ' Example: {"person": 2.0, "car": 0.7, "re:^dog_.*$": 1.3}'
            ),
            valueSchema=string_schema(default="{}"),
            access=F8StateAccess.rw,
            required=True,
            control=F8UiControlSpec(kind=F8UiControlKind.code, language="json"),
            showOnNode=False,
        ),
        F8StateSpec(
            name="sortDirection",
            label="Sort Direction",
            description="Prefer larger scores first (desc) or smaller scores first (asc).",
            valueSchema=string_schema(default="desc", enum=["desc", "asc"]),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="scoreAggregation",
            label="Score Aggregation",
            description="ROI reduction mode used to rank each bbox.",
            valueSchema=string_schema(default="mean", enum=["mean", "max", "sum", "median"]),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="temperature",
            label="Temperature",
            description=(
                "Randomness applied to score-ranked detections. 0 keeps deterministic sorting;"
                " larger values increasingly explore lower-ranked detections."
            ),
            valueSchema=number_schema(default=0.0, minimum=0.0),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
    ]


def _common_state_fields(
    *,
    include_thresholds: bool,
    include_top_k: bool,
    include_class_filter: bool,
) -> list[F8StateSpec]:
    fields = [
        F8StateSpec(
            name="weightsDir",
            label="Weights Dir",
            description="Directory containing *.yaml + *.onnx model files. Reset to the default relative path when exporting publish JSON.",
            valueSchema=string_schema(default="services/f8/dl/weights"),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="modelId",
            label="Model Id",
            description="Model id selected from weightsDir (ignored if modelYamlPath is set).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            required=True,
            control=F8UiControlSpec(kind=F8UiControlKind.select, optionsFromState="availableModels"),
            showOnNode=True,
        ),
        F8StateSpec(
            name="modelYamlPath",
            label="Model YAML Path",
            description="Optional explicit model yaml path (overrides modelId). Cleared when exporting publish JSON.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="ortProvider",
            label="ONNX Runtime Provider",
            description="auto prefers CUDAExecutionProvider when available.",
            valueSchema=string_schema(default="auto", enum=["auto", "cuda", "cpu"]),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=True,
        ),
        F8StateSpec(
            name="autoDownloadWeights",
            label="Auto Download Weights",
            description="When model file is missing, download from onnxUrl in model yaml.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="inferEveryN",
            label="Infer Every N Frames",
            description="Run model inference every N frames (>=1).",
            valueSchema=integer_schema(default=1, minimum=1, maximum=10000),
            access=F8StateAccess.rw,
            required=True,
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
                    required=True,
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="iouThreshold",
                    label="IoU Threshold Override",
                    description="Override IoU threshold for NMS (negative uses model yaml).",
                    valueSchema=number_schema(default=-1.0),
                    access=F8StateAccess.rw,
                    required=True,
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
                required=True,
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
                    valueSchema=array_schema(items=string_schema(), default=[]),
                    access=F8StateAccess.rw,
                    required=True,
                    control=F8UiControlSpec(kind=F8UiControlKind.multiselect, optionsFromState="modelClasses"),
                    showOnNode=False,
                ),
                F8StateSpec(
                    name="perClassK",
                    label="Per Class K",
                    description="Per-class top-K by score (<=0 means unlimited).",
                    valueSchema=integer_schema(default=0, minimum=0, maximum=10000),
                    access=F8StateAccess.rw,
                    required=True,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="modelClasses",
                    label="Model Classes",
                    description="Current loaded model class labels.",
                    valueSchema=array_schema(items=string_schema(), default=[]),
                    access=F8StateAccess.ro,
                    required=True,
                    showOnNode=False,
                )
            ]
        )
    fields.append(
        F8StateSpec(
            name="availableModels",
            label="Available Models",
            description="List of model ids discovered from weightsDir.",
            valueSchema=array_schema(items=string_schema(), default=[]),
            access=F8StateAccess.ro,
            required=True,
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
                required=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="ortActiveProviders",
                label="ORT Active Providers",
                description="JSON list of active ONNX Runtime providers for this session.",
                valueSchema=string_schema(default=""),
                access=F8StateAccess.ro,
                required=True,
                showOnNode=False,
            ),
        ]
    )
    return fields


def _optflow_state_fields() -> list[F8StateSpec]:
    return [
        F8StateSpec(
            name="computeEveryNFrames",
            label="Compute Every N Frames",
            description="Compute optical flow once per N new frames.",
            valueSchema=integer_schema(default=2, minimum=1, maximum=120),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="weightsDir",
            label="Weights Dir",
            description="Directory containing *.yaml + *.onnx model files. Reset to the default relative path when exporting publish JSON.",
            valueSchema=string_schema(default="services/f8/dl/weights"),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="modelId",
            label="Model Id",
            description="Model id selected from weightsDir (ignored if modelYamlPath is set).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            required=True,
            control=F8UiControlSpec(kind=F8UiControlKind.select, optionsFromState="availableModels"),
            showOnNode=False,
        ),
        F8StateSpec(
            name="modelYamlPath",
            label="Model YAML Path",
            description="Optional explicit model yaml path (overrides modelId). Cleared when exporting publish JSON.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
            redactOnPublish=True,
        ),
        F8StateSpec(
            name="ortProvider",
            label="ONNX Runtime Provider",
            description="auto prefers CUDAExecutionProvider when available.",
            valueSchema=string_schema(default="auto", enum=["auto", "cuda", "cpu"]),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="autoDownloadWeights",
            label="Auto Download Weights",
            description="When model file is missing, download from onnxUrl in model yaml.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="availableModels",
            label="Available Models",
            description="List of model ids discovered from weightsDir.",
            valueSchema=array_schema(items=string_schema(), default=[]),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="loadedModel",
            label="Loaded Model",
            description="Current loaded model id/task.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="ortActiveProviders",
            label="ORT Active Providers",
            description="JSON list of active ONNX Runtime providers for this session.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            required=True,
            showOnNode=False,
        ),
        F8StateSpec(
            name="flowFormat",
            label="Flow Format",
            description="Flow payload format. Fixed to flow2_f16.",
            valueSchema=string_schema(default="flow2_f16", enum=["flow2_f16"]),
            access=F8StateAccess.ro,
            required=True,
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
                required=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="outputBias",
                label="Output Bias",
                description="Denormalization bias applied after outputScale.",
                valueSchema=number_schema(default=0.0),
                access=F8StateAccess.rw,
                required=True,
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
                "This assumes the selected video input already provides the target eye view and crops top 20% + left/right 10%."
            ),
            valueSchema=boolean_schema(default=False),
            access=F8StateAccess.rw,
            required=True,
            showOnNode=False,
        )
    )
    return fields


def _register_classifier(registry: Registry) -> None:
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

    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=CLASSIFIER_SERVICE_CLASS,
            paletteCategory="svc",
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
            dataInPorts=[_video_input_port()],
            dataOutPorts=[
                F8DataPortSpec(
                    name="classifications",
                    description="Classification output in schema f8visionClassifications/1.",
                    valueSchema=_classifications_payload_schema(),
                ),
            ],
        ),
        _factory,
        overwrite=True,
    )


def _register_detector(registry: Registry) -> None:
    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return OnnxVisionServiceNode(
            node_id=node_id,
            node=node,
            initial_state=initial_state,
            service_class=DETECTOR_SERVICE_CLASS,
            service_task="detector",
            output_port="detections",
            allowed_tasks={"yolo_det", "yolo_obb", "yowo_temporal_det"},
        )

    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=DETECTOR_SERVICE_CLASS,
            paletteCategory="svc",
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
            dataInPorts=[_video_input_port()],
            dataOutPorts=[
                F8DataPortSpec(
                    name="detections",
                    description="Detection output in schema f8visionDetections/1.",
                    valueSchema=_detections_payload_schema(),
                ),
            ],
        ),
        _factory,
        overwrite=True,
    )


def _register_human_detector(registry: Registry) -> None:
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

    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=HUMAN_DETECTOR_SERVICE_CLASS,
            paletteCategory="svc",
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
            dataInPorts=[_video_input_port()],
            dataOutPorts=[
                F8DataPortSpec(
                    name="detections",
                    description="Detection output in schema f8visionDetections/1.",
                    valueSchema=_detections_payload_schema(),
                ),
            ],
        ),
        _factory,
        overwrite=True,
    )


def _register_optflow(registry: Registry) -> None:
    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return OnnxOptflowServiceNode(
            node_id=node_id,
            node=node,
            initial_state=initial_state,
            service_class=OPTFLOW_SERVICE_CLASS,
            allowed_tasks={"optflow_neuflowv2"},
        )

    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=OPTFLOW_SERVICE_CLASS,
            paletteCategory="svc",
            version="0.0.1",
            label="DL Optical Flow",
            description="ONNXRuntime NeuFlowV2 dense optical flow service (Zenoh latest-frame flow output).",
            tags=["onnx", "vision", "optical_flow", "zenoh_flow"],
            rendererClass="default_svc",
            stateFields=_optflow_state_fields(),
            dataInPorts=[_video_input_port()],
            dataOutPorts=[
                video_frame_port(
                    name="flow",
                    description="Dense optical-flow frame stream.",
                    required=True,
                ),
            ],
        ),
        _factory,
        overwrite=True,
    )


def _register_detection_sorter(registry: Registry) -> None:
    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=DETECTION_SORTER_SERVICE_CLASS,
            paletteCategory="svc",
            version="0.0.1",
            label="DL Detection Sorter",
            description="Sort detection payloads by a score-map metric.",
            tags=["vision", "detection", "sort", "score_map"],
            rendererClass="default_svc",
            stateFields=_detection_sorter_state_fields(),
            dataInPorts=[
                F8DataPortSpec(
                    name="detections",
                    description="Detection input in schema f8visionDetections/1.",
                    valueSchema=_detections_payload_schema(),
                    required=True,
                ),
                video_frame_port(
                    name="score",
                    description="Scalar or flow score-map frame stream used to rank detections.",
                    required=True,
                ),
            ],
            dataOutPorts=[
                F8DataPortSpec(
                    name="detections",
                    description="Sorted detections in schema f8visionDetections/1.",
                    valueSchema=_detections_payload_schema(),
                    required=True,
                ),
            ],
        ),
        DetectionSorterServiceNode,
        overwrite=True,
    )

def _register_tcn_wave(registry: Registry) -> None:
    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return OnnxTcnWaveServiceNode(
            node_id=node_id,
            node=node,
            initial_state=initial_state,
            service_class=TCNWAVE_SERVICE_CLASS,
            allowed_tasks={"tcn_wave"},
        )

    registry.register_service(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=TCNWAVE_SERVICE_CLASS,
            paletteCategory="svc",
            version="0.0.1",
            label="DL TCN Wave",
            description="ONNXRuntime temporal convolution wave inference service (port output).",
            tags=["onnx", "vision", "temporal", "wave", "signal"],
            rendererClass="default_svc",
            stateFields=_tcn_wave_state_fields(),
            dataInPorts=[_video_input_port()],
            dataOutPorts=[
                F8DataPortSpec(
                    name="predictedChange",
                    description="Temporal model output value per frame.",
                    valueSchema=number_schema(),
                ),
            ],
        ),
        _factory,
        overwrite=True,
    )


def register_specs(registry: Registry) -> Registry:
    _register_classifier(registry)
    _register_detector(registry)
    _register_human_detector(registry)
    _register_optflow(registry)
    _register_detection_sorter(registry)
    _register_tcn_wave(registry)
    return registry


def create_pydl_registry() -> RuntimeNodeRegistry:
    runtime_registry = create_runtime_node_registry()
    register_specs(Registry.wrap(runtime_registry))
    return runtime_registry


def shared_pydl_registry() -> RuntimeNodeRegistry:
    runtime_registry = shared_runtime_node_registry()
    register_specs(Registry.wrap(runtime_registry))
    return runtime_registry
