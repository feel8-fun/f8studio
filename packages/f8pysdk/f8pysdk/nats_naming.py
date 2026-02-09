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

def kv_key_ready() -> str:
    """
    Service readiness key in the per-service KV bucket.

    Payload is JSON bytes written by the service runtime.
    """
    return "ready"


def kv_key_node_state(*, node_id: str, field: str) -> str:
    node_id = ensure_token(node_id, label="node_id")
    field = str(field).strip()
    if not field:
        raise ValueError("field must be non-empty")
    return f"nodes.{node_id}.state.{field}"


def parse_kv_key_node_state(key: str) -> tuple[str, str] | None:
    """
    Parse a KV key in the form: nodes.<nodeId>.state.<field...>.

    Inverse of `kv_key_node_state(node_id=..., field=...)`.
    """
    parts = str(key).strip(".").split(".")
    if len(parts) < 4:
        return None
    if parts[0] != "nodes" or parts[2] != "state":
        return None
    node_id = parts[1]
    field = ".".join(parts[3:])
    if not node_id or not field:
        return None
    return node_id, field

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

def cmd_channel_subject(service_id: str) -> str:
    """
    Reserved command channel for user-defined service commands.

    The request payload should include a JSON envelope (reqId/call/args/meta).
    """
    service_id = ensure_token(service_id, label="service_id")
    return f"svc.{service_id}.cmd"


def svc_endpoint_subject(service_id: str, endpoint: str) -> str:
    """
    Built-in lifecycle/control endpoints (typically backed by NATS micro).
    """
    service_id = ensure_token(service_id, label="service_id")
    endpoint = ensure_token(str(endpoint), label="endpoint")
    return f"svc.{service_id}.{endpoint}"


def svc_micro_name(service_id: str) -> str:
    """
    NATS micro service name for a service instance.

    Micro service names cannot contain ".", so we use `svc_<serviceId>`.
    """
    return kv_bucket_for_service(ensure_token(service_id, label="service_id"))


def new_id() -> str:
    """
    Stable, token-safe id (uuid4 hex).
    """
    return uuid.uuid4().hex
