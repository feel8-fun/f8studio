from __future__ import annotations

from f8pysdk import (
    F8Command,
    F8CommandParam,
    F8DataPortSpec,
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
from .template_tracker_node import TemplateTrackerServiceNode


def register_template_tracker_specs(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

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
            name="active",
            label="Active",
            description="Enable tracking when true.",
            valueSchema=boolean_schema(default=True),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="trackerKind",
            label="Tracker",
            description="OpenCV tracker to use for high-frequency tracking.",
            valueSchema=string_schema(default="csrt", enum=["none", "csrt", "kcf", "mosse"]),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="matchMethod",
            label="Match Method",
            description="OpenCV matchTemplate method.",
            valueSchema=string_schema(default="TM_CCOEFF_NORMED"),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="matchThreshold",
            label="Match Threshold",
            description="Minimum match score to accept a template match (0..1 for *_NORMED methods).",
            valueSchema=number_schema(default=0.75, minimum=0.0, maximum=1.0),
            access=F8StateAccess.rw,
            showOnNode=True,
        ),
        F8StateSpec(
            name="searchMarginPx",
            label="Search Margin (px)",
            description="When tracking, restrict template search to an expanded ROI around last bbox (0 = full frame).",
            valueSchema=integer_schema(default=200, minimum=0, maximum=100000),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="reacquireIntervalMs",
            label="Reacquire Interval (ms)",
            description="When tracking is lost/unavailable, run template matching at most once per N ms to reacquire (0 disables auto-reacquire).",
            valueSchema=integer_schema(default=500, minimum=0, maximum=60000),
            access=F8StateAccess.rw,
            showOnNode=False,
        ),
        F8StateSpec(
            name="templatePngB64",
            label="Template (PNG, base64)",
            description="Template image (PNG, base64). Use captureFrame + setTemplateFromCaptureRoi.",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.rw,
            showOnNode=True,
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
        F8StateSpec(
            name="lastError",
            label="Last Error",
            description="Last runtime error string (best-effort).",
            valueSchema=string_schema(default=""),
            access=F8StateAccess.ro,
            showOnNode=False,
        ),
    ]

    reg.register_service_spec(
        F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1,
            serviceClass=SERVICE_CLASS,
            version="0.0.1",
            label="Template Tracker",
            description="Single-target template matching + OpenCV tracker service.",
            tags=["vision", "tracker", "template"],
            rendererClass="pystudio_template_tracker",
            stateFields=state_fields,
            commands=[
                F8Command(
                    name="captureFrame",
                    description="Capture the current SHM frame as an encoded image (base64).",
                    showOnNode=True,
                    params=[
                        F8CommandParam(name="format", valueSchema=string_schema(default="jpg", enum=["jpg", "png"]), required=False),
                        F8CommandParam(name="quality", valueSchema=integer_schema(default=85, minimum=1, maximum=100), required=False),
                        F8CommandParam(name="maxBytes", valueSchema=integer_schema(default=900000, minimum=10000, maximum=5000000), required=False),
                        F8CommandParam(name="maxWidth", valueSchema=integer_schema(default=1280, minimum=0, maximum=10000), required=False),
                        F8CommandParam(name="maxHeight", valueSchema=integer_schema(default=720, minimum=0, maximum=10000), required=False),
                    ],
                ),
                F8Command(
                    name="setTemplateFromCaptureRoi",
                    description="Set template from the most recent captureFrame using ROI coords.",
                    showOnNode=True,
                    params=[
                        F8CommandParam(name="captureFrameId", valueSchema=integer_schema(), required=True),
                        F8CommandParam(name="x1", valueSchema=integer_schema(minimum=0), required=True),
                        F8CommandParam(name="y1", valueSchema=integer_schema(minimum=0), required=True),
                        F8CommandParam(name="x2", valueSchema=integer_schema(minimum=0), required=True),
                        F8CommandParam(name="y2", valueSchema=integer_schema(minimum=0), required=True),
                    ],
                ),
                F8Command(
                    name="matchNow",
                    description="Force a template matching pass on the latest frame.",
                    showOnNode=True,
                    params=[],
                ),
                F8Command(
                    name="clearTemplate",
                    description="Clear the current template.",
                    showOnNode=True,
                    params=[],
                ),
            ],
            dataOutPorts=[
                F8DataPortSpec(
                    name="tracking",
                    description="Per-frame single-target tracking result (bbox + score + status).",
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

    def _service_factory(node_id: str, node, initial_state: dict) -> RuntimeNode:
        return TemplateTrackerServiceNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register_service(SERVICE_CLASS, _service_factory, overwrite=True)
    return reg
