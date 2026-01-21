from __future__ import annotations

from typing import Any

from f8pysdk import F8OperatorSchemaVersion, F8OperatorSpec, F8RuntimeNode
from f8pysdk.nats_naming import ensure_token
from f8pysdk.runtime_node import RuntimeNode
from f8pysdk.runtime_node_registry import RuntimeNodeRegistry

from ..service_host.constants import SERVICE_CLASS

OPERATOR_CLASS = "f8.monitor_state"


class MonitorStateRuntimeNode(RuntimeNode):
    """
    Studio-internal node used as a fan-in target for cross-service state monitoring.

    This node is not intended to be user-facing; Studio routes updates by decoding
    the synthetic state field names (eg. "<serviceId>|<nodeId>|<field>").
    """

    def __init__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any] | None = None) -> None:
        super().__init__(
            node_id=ensure_token(node_id, label="node_id"),
            data_in_ports=[],
            data_out_ports=[],
            state_fields=[s.name for s in (node.stateFields or [])],
        )


def register_operator(registry: RuntimeNodeRegistry | None = None) -> RuntimeNodeRegistry:
    reg = registry or RuntimeNodeRegistry.instance()

    def _factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
        return MonitorStateRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register(SERVICE_CLASS, OPERATOR_CLASS, _factory, overwrite=True)
    reg.register_operator_spec(
        F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1,
            serviceClass=SERVICE_CLASS,
            operatorClass=OPERATOR_CLASS,
            version="0.0.1",
            label="Monitor State (internal)",
            description="Internal node used by Studio to monitor remote state via cross-state edges.",
            tags=["internal", "monitor", "state"],
        ),
        overwrite=True,
    )
    return reg

