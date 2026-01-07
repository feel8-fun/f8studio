from __future__ import annotations

import sys

from f8pysdk import (
    F8ServiceLaunchSpec,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    string_schema,
)


ENGINE_SERVICE_CLASS = "svc.feel8.engine"
EDITOR_SERVICE_CLASS = "svc.feel8.editor"


def engine_service_spec() -> F8ServiceSpec:
    """
    Engine as a service (v1).

    The actual operator graph topology + per-node state are stored in NATS KV:
      - `svc.<serviceId>.topology`
      - `svc.<serviceId>.nodes.<nodeId>.state.<field>`
    where `serviceId` is the service node id.
    """
    return F8ServiceSpec(
        serviceClass=ENGINE_SERVICE_CLASS,
        version="0.0.1",
        label="Engine",
        description="Runs an embedded OperatorGraph as a managed service process.",
        tags=["service", "engine"],
        rendererClass="backdrop",
        launch=F8ServiceLaunchSpec(
            commandSpec=sys.executable,
            args=["-m", "f8pyengineqt.engine.engine_service_process"],
            env={},
            workdir="./",
        ),
        states=[
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
        editableStates=False,
        editableDataInPorts=True,
        editableDataOutPorts=True,
        editableCommands=True,
    )


def editor_service_spec() -> F8ServiceSpec:
    """
    Editor as a service (v1).

    Runs in-process (the editor UI itself), but participates in cross-edge
    routing + state propagation the same way as other services.
    """
    return F8ServiceSpec(
        serviceClass=EDITOR_SERVICE_CLASS,
        version="0.0.1",
        label="Editor",
        description="In-process editor service for UI operators.",
        tags=["service", "editor", "ui"],
        rendererClass="default",
        states=[],
        editableStates=False,
        editableDataInPorts=True,
        editableDataOutPorts=True,
        editableCommands=True,
    )
