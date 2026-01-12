from __future__ import annotations

from f8pysdk import (
    F8DataPortSpec,
    F8OperatorSpec,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    F8ServiceSchemaVersion,
    F8OperatorSchemaVersion,
    F8ServiceDescribe,
    any_schema,
    array_schema,
    boolean_schema,
    complex_object_schema,
    integer_schema,
    number_schema,
    string_schema,
)


ENGINE_SERVICE_CLASS = "f8.engine"


def engine_service_spec() -> F8ServiceSpec:
    return F8ServiceSpec(
        schemaVersion=F8ServiceSchemaVersion.f8service_1,
        serviceClass=ENGINE_SERVICE_CLASS,
        version="0.0.1",
        label="Engine",
        description="Runs an embedded OperatorGraph as a managed service process.",
        tags=["service", "engine"],
        rendererClass="backdrop",
        stateFields=[
            F8StateSpec(
                name="natsUrl",
                label="NATS URL",
                valueSchema=string_schema(default="nats://127.0.0.1:4222"),
                access=F8StateAccess.init,
                required=True,
                showOnNode=False,
            ),
            F8StateSpec(
                name="topology",
                label="Topology",
                description="Operator graph topology snapshot (published to NATS KV).",
                valueSchema=any_schema(),
                access=F8StateAccess.ro,
                required=False,
                showOnNode=False,
            ),
        ],
        editableStateFields=False,
        editableDataInPorts=True,
        editableDataOutPorts=True,
        editableCommands=True,
    )


def engine_operator_specs() -> list[F8OperatorSpec]:
    return [
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=ENGINE_SERVICE_CLASS,
            operatorClass="f8.start",
            version="0.0.1",
            label="Start",
            description="Entry trigger for demo graphs.",
            execOutPorts=["exec"],
        ),
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=ENGINE_SERVICE_CLASS,
            operatorClass="f8.constant",
            version="0.0.1",
            label="Constant",
            description="Emits a configured numeric value.",
            execInPorts=["exec"],
            execOutPorts=["exec"],
            dataOutPorts=[F8DataPortSpec(name="value", description="constant output", valueSchema=number_schema())],
            stateFields=[
                F8StateSpec(
                    name="value2",
                    label="Value",
                    valueSchema=number_schema(default=1.0, minimum=-1000, maximum=1000),
                    access=F8StateAccess.rw,
                )
            ],
        ),
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=ENGINE_SERVICE_CLASS,
            operatorClass="f8.add",
            version="0.0.1",
            label="Add",
            description="Adds two inputs and forwards the result.",
            execInPorts=["exec"],
            execOutPorts=["exec"],
            dataInPorts=[
                F8DataPortSpec(name="a", description="lhs", valueSchema=number_schema()),
                F8DataPortSpec(name="b", description="rhs", valueSchema=number_schema()),
            ],
            dataOutPorts=[F8DataPortSpec(name="sum", description="a+b", valueSchema=number_schema())],
        ),
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=ENGINE_SERVICE_CLASS,
            operatorClass="f8.log",
            version="0.0.1",
            label="Log",
            description="Terminal node that inspects incoming data.",
            tags=["ui"],
            dataInPorts=[
                F8DataPortSpec(name="value", description="value to log", required=False, valueSchema=number_schema())
            ],
            stateFields=[
                F8StateSpec(
                    name="label",
                    label="Label",
                    valueSchema=string_schema(default="Log"),
                    access=F8StateAccess.ro,
                ),
                F8StateSpec(
                    name="bool",
                    label="bool",
                    valueSchema=boolean_schema(default=True),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="count",
                    label="Count",
                    valueSchema=integer_schema(default=5, minimum=0, maximum=100),
                    access=F8StateAccess.rw,
                    showOnNode=True,
                ),
                F8StateSpec(
                    name="options",
                    label="Options",
                    valueSchema=string_schema(enum=["Option A", "Option B", "Option C"]),
                    access=F8StateAccess.ro,
                ),
                F8StateSpec(
                    name="metadata",
                    label="Metadata",
                    valueSchema=complex_object_schema(
                        properties={
                            "author": string_schema(default="Unknown"),
                            "version": integer_schema(default=1, minimum=1),
                            "tags": array_schema(items=string_schema()),
                        }
                    ),
                    access=F8StateAccess.ro,
                ),
            ],
        ),
    ]


def describe_payload_json() -> dict:
    return F8ServiceDescribe(
        service=engine_service_spec(),
        operators=engine_operator_specs(),
    ).model_dump(mode="json")
