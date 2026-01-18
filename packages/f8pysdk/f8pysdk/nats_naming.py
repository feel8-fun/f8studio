from __future__ import annotations

import uuid


def ensure_token(value: str, *, label: str) -> str:
    """
    Ensure a string is safe to use as a single NATS subject token.

    We use "." as the separator across the project, so ids must not contain ".".
    """
    value = str(value).strip()
    if not value:
        raise ValueError(f"{label} must be non-empty")
    if "." in value:
        raise ValueError(f'{label} must not contain "." (got {value!r}).')
    return value


def kv_bucket_for_service(service_id: str) -> str:
    """
    Default bucket name for a service instance.
    """
    return f"svc_{ensure_token(service_id, label='service_id')}"


def kv_key_rungraph() -> str:
    return "rungraph"


def kv_key_node_state(*, node_id: str, field: str) -> str:
    node_id = ensure_token(node_id, label="node_id")
    field = str(field).strip()
    if not field:
        raise ValueError("field must be non-empty")
    return f"nodes.{node_id}.state.{field}"


def edge_subject(receiver_service_id: str, edge_id: str) -> str:
    """
    Legacy cross-instance data edge subject.

    Prefer `data_subject(from_service_id, from_node_id, port_id)` so fan-out
    publishes once per output port and receivers subscribe independently.
    """
    receiver_service_id = ensure_token(receiver_service_id, label="receiver_service_id")
    edge_id = ensure_token(edge_id, label="edge_id")
    return f"svc.{receiver_service_id}.edges.{edge_id}"


def data_subject(from_service_id: str, *, from_node_id: str, port_id: str) -> str:
    """
    Cross-instance data bus subject for an output port.

    Fan-out design: publish once per (service,node,out_port); multiple receivers
    subscribe to the same subject.
    """
    from_service_id = ensure_token(from_service_id, label="from_service_id")
    from_node_id = ensure_token(from_node_id, label="from_node_id")
    port_id = ensure_token(port_id, label="port_id")
    return f"svc.{from_service_id}.nodes.{from_node_id}.data.{port_id}"


def cmd_subject(service_id: str, cmd: str) -> str:
    """
    Command subject for a service instance.

    Examples:
    - `svc.<serviceId>.cmd.run` (execute once)
    """
    service_id = ensure_token(service_id, label="service_id")
    cmd = ensure_token(cmd, label="cmd")
    return f"svc.{service_id}.cmd.{cmd}"


def new_id() -> str:
    """
    Stable, token-safe id (uuid4 hex).
    """
    return uuid.uuid4().hex
