from __future__ import annotations

import uuid

from .generated import runtime_keys


class TokenValidationError(ValueError):
    """A runtime identifier cannot be represented as one transport path token."""


def ensure_token(value: str, *, label: str) -> str:
    """
    Ensure a string is safe to use as a single runtime path token.

    We use "." and "/" as transport path separators, so ids must not contain them.
    """
    value = str(value).strip()
    if not value:
        raise TokenValidationError(f"{label} must be non-empty")
    if "." in value or "/" in value:
        raise TokenValidationError(f'{label} must not contain "." or "/" (got {value!r}).')
    return value


def state_path_node_field(*, node_id: str, field: str) -> str:
    node_id = ensure_token(node_id, label="node_id")
    field = str(field).strip()
    if not field:
        raise ValueError("field must be non-empty")
    return f"nodes.{node_id}.state.{field}"


def parse_state_path_node_field(path: str) -> tuple[str, str] | None:
    """
    Parse a state path in the form: nodes.<nodeId>.state.<field...>.

    Inverse of `state_path_node_field(node_id=..., field=...)`.
    """
    parts = str(path).strip(".").split(".")
    if len(parts) < 4:
        return None
    if parts[0] != "nodes" or parts[2] != "state":
        return None
    node_id = parts[1]
    field = ".".join(parts[3:])
    if not node_id or not field:
        return None
    return node_id, field

def data_key(from_service_id: str, *, from_node_id: str, port_id: str) -> str:
    """
    Cross-instance Zenoh key for an output port.

    Fan-out design: publish once per (service,node,out_port); multiple receivers
    subscribe to the same key expression.
    """
    from_service_id = ensure_token(from_service_id, label="from_service_id")
    from_node_id = ensure_token(from_node_id, label="from_node_id")
    port_id = ensure_token(port_id, label="port_id")
    return runtime_keys.data(from_service_id, from_node_id, port_id)

def cmd_channel_key(service_id: str) -> str:
    """
    Reserved command channel for user-defined service commands.

    The request payload should include a JSON envelope (reqId/call/args/meta).
    """
    service_id = ensure_token(service_id, label="service_id")
    return runtime_keys.command(service_id, "cmd")


def svc_endpoint_key(service_id: str, endpoint: str) -> str:
    """
    Built-in lifecycle/control endpoint command key.
    """
    service_id = ensure_token(service_id, label="service_id")
    endpoint = ensure_token(str(endpoint), label="endpoint")
    return runtime_keys.command(service_id, endpoint)


def new_id() -> str:
    """
    Stable, token-safe id (uuid4 hex).
    """
    return uuid.uuid4().hex
