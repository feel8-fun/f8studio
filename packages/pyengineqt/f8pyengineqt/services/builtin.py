from __future__ import annotations

from f8pysdk import (
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
    any_schema,
    string_schema,
)


ENGINE_SERVICE_CLASS = "fun.feel8.engine"


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
