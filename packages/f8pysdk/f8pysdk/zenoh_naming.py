from __future__ import annotations

from .generated import runtime_keys
from .f8_naming import ensure_token, parse_state_path_node_field

_F8_PREFIX = "f8"


def _field_to_path(field: str) -> str:
    parts = [part for part in str(field or "").strip(".").split(".") if part]
    if not parts:
        raise ValueError("field must be non-empty")
    return "/".join(parts)


def _path_to_field(path: str) -> str:
    parts = [part for part in str(path or "").strip("/").split("/") if part]
    if not parts:
        raise ValueError("field path must be non-empty")
    return ".".join(parts)


def zenoh_data_key(service_id: str, *, node_id: str, port_id: str) -> str:
    sid = ensure_token(service_id, label="service_id")
    nid = ensure_token(node_id, label="node_id")
    pid = ensure_token(port_id, label="port_id")
    return runtime_keys.data(sid, nid, pid)


def zenoh_command_key(service_id: str, command: str) -> str:
    sid = ensure_token(service_id, label="service_id")
    cmd = ensure_token(command, label="command")
    return runtime_keys.command(sid, cmd)


def zenoh_service_liveliness_key(service_id: str, runtime_instance_id: str) -> str:
    sid = ensure_token(service_id, label="service_id")
    rid = ensure_token(runtime_instance_id, label="runtime_instance_id")
    return runtime_keys.service_liveliness(sid, rid)


def zenoh_studio_liveliness_key(studio_service_id: str) -> str:
    sid = ensure_token(studio_service_id, label="studio_service_id")
    return runtime_keys.studio_liveliness(sid)


def zenoh_state_key(service_id: str, *, node_id: str, field: str) -> str:
    sid = ensure_token(service_id, label="service_id")
    nid = ensure_token(node_id, label="node_id")
    field_path = _field_to_path(field)
    return runtime_keys.node_state(sid, nid, field_path)


def zenoh_state_path_key(service_id: str, path: str) -> str:
    parsed = parse_state_path_node_field(path)
    if parsed is not None:
        node_id, field = parsed
        return zenoh_state_key(service_id, node_id=node_id, field=field)
    sid = ensure_token(service_id, label="service_id")
    key_path = "/".join(part for part in str(path or "").strip(".").split(".") if part)
    if not key_path:
        raise ValueError("state path must be non-empty")
    return runtime_keys.state(sid, key_path)


def zenoh_state_path_pattern(service_id: str, path_pattern: str) -> str:
    pattern = str(path_pattern or "").strip()
    if not pattern:
        raise ValueError("state path pattern must be non-empty")
    if pattern.endswith(">"):
        prefix = pattern[:-1].rstrip(".")
        parsed_prefix = [part for part in prefix.split(".") if part]
        if parsed_prefix and parsed_prefix[0] == "nodes":
            sid = ensure_token(service_id, label="service_id")
            if len(parsed_prefix) == 1:
                return f"{_F8_PREFIX}/svc/{sid}/state/nodes/**"
            node_id = ensure_token(parsed_prefix[1], label="node_id")
            if len(parsed_prefix) == 2:
                return f"{_F8_PREFIX}/svc/{sid}/state/nodes/{node_id}/**"
            if len(parsed_prefix) >= 3 and parsed_prefix[2] == "state":
                return f"{_F8_PREFIX}/svc/{sid}/state/nodes/{node_id}/state/**"
        sid = ensure_token(service_id, label="service_id")
        prefix_path = "/".join(part for part in prefix.split(".") if part)
        return f"{_F8_PREFIX}/svc/{sid}/state/{prefix_path}/**"
    return zenoh_state_path_key(service_id, pattern)


def zenoh_key_to_state_path(key: str) -> str | None:
    text = str(key or "").strip("/")
    parts = text.split("/")
    if len(parts) >= 8 and parts[0] == _F8_PREFIX and parts[1] == "svc" and parts[3] == "state":
        if parts[4] != "nodes" or parts[6] != "state":
            return None
        node_id = parts[5]
        field = _path_to_field("/".join(parts[7:]))
        return f"nodes.{node_id}.state.{field}"
    if len(parts) >= 5 and parts[0] == _F8_PREFIX and parts[1] == "svc" and parts[3] == "state":
        return ".".join(parts[4:])
    return None


__all__ = [
    "zenoh_command_key",
    "zenoh_data_key",
    "zenoh_key_to_state_path",
    "zenoh_service_liveliness_key",
    "zenoh_state_key",
    "zenoh_state_path_key",
    "zenoh_state_path_pattern",
    "zenoh_studio_liveliness_key",
]
