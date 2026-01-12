from __future__ import annotations

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSpec,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    F8ServiceSchemaVersion,
    F8OperatorSchemaVersion,
    any_schema,
    boolean_schema,
    integer_schema,
    number_schema,
)


STUDIO_SERVICE_CLASS = "f8.studio"


def studio_service_spec() -> F8ServiceSpec:
    return F8ServiceSpec(
        schemaVersion=F8ServiceSchemaVersion.f8service_1,
        serviceClass=STUDIO_SERVICE_CLASS,
        version="0.0.1",
        label="Editor",
        description="In-process editor service for UI operators.",
        tags=["service", "editor", "ui"],
        rendererClass="default",
        editableStateFields=False,
        editableDataInPorts=True,
        editableDataOutPorts=True,
        editableCommands=True,
    )


def studio_operator_specs() -> list[F8OperatorSpec]:
    return [
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=STUDIO_SERVICE_CLASS,
            operatorClass="f8.studio.log",
            version="0.0.1",
            label="Editor Log",
            description="Receives data via cross edges and prints / displays it in the editor.",
            tags=["ui", "editor"],
            rendererClass="editor_log",
            dataInPorts=[F8DataPortSpec(name="in", description="data input", valueSchema=any_schema())],
            stateFields=[
                F8StateSpec(
                    name="refreshMs",
                    label="Refresh (ms)",
                    valueSchema=integer_schema(default=200, minimum=16, maximum=5000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="print",
                    label="Print",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
            ],
        ),
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=STUDIO_SERVICE_CLASS,
            operatorClass="f8.studio.oscilloscope",
            version="0.0.1",
            label="Oscilloscope",
            description="Receives numeric data via cross edges and visualizes it (placeholder UI).",
            tags=["ui", "editor"],
            rendererClass="ui",
            dataInPorts=[F8DataPortSpec(name="in", description="signal input", valueSchema=number_schema())],
            stateFields=[
                F8StateSpec(
                    name="refreshMs",
                    label="Refresh (ms)",
                    valueSchema=integer_schema(default=50, minimum=16, maximum=5000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="window",
                    label="Window",
                    valueSchema=integer_schema(default=240, minimum=10, maximum=5000),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
            ],
        ),
    ]


def describe_payload_json() -> dict:
    return {
        "service": studio_service_spec().model_dump(mode="json"),
        "operators": [s.model_dump(mode="json") for s in studio_operator_specs()],
    }
