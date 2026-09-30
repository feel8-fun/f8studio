"""State-side helper functions owned by `service_bus`."""
from __future__ import annotations

from typing import Any

from ...state import StateWriteSource
from ._errors import STATE_VALUE_COERCION_ERRORS


_STATE_PAYLOAD_RESERVED_KEYS = frozenset(("value", "actor", "ts", "tsMs", "ts_ms", "source", "origin"))


def _resolved_source_text(source: StateWriteSource | str | None, *, default: str) -> str:
    if isinstance(source, StateWriteSource):
        return source.value
    text = str(source or "").strip()
    return text or str(default)


def build_intra_state_route_meta(*, from_node_id: str, from_field: str) -> dict[str, Any]:
    """
    Metadata for intra-service state edge fanout writes.
    """
    return {
        "fromNodeId": str(from_node_id),
        "fromField": str(from_field),
    }


def build_cross_state_meta(
    *, peer_service_id: str, remote_key: str, inbound_meta: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    Metadata for cross-service state binding application.
    """
    meta_out: dict[str, Any] = {
        "peerServiceId": str(peer_service_id),
        "remoteKey": str(remote_key),
    }
    for key, value in dict(inbound_meta or {}).items():
        if key in _STATE_PAYLOAD_RESERVED_KEYS:
            continue
        meta_out[str(key)] = value
    return meta_out


def build_state_validation_meta(
    *, source: StateWriteSource | str | None, meta: dict[str, Any] | None
) -> dict[str, Any]:
    """
    Metadata passed into `validate_state(...)` implementations.

    This keeps source tagging explicit while preserving caller metadata.
    """
    out = dict(meta or {})
    out.setdefault("source", _resolved_source_text(source, default=""))
    if not out.get("source"):
        out.pop("source", None)
    return out


def coerce_inbound_ts_ms(ts_raw: Any, *, default: int) -> int:
    """
    Best-effort coercion of inbound timestamps to milliseconds.

    Accepts common cases:
    - missing/invalid -> `default`
    - seconds since epoch -> convert to ms (heuristic by magnitude)
    - microseconds/nanoseconds since epoch -> downscale to ms (heuristic)
    """
    try:
        if ts_raw is None:
            return int(default)
        if isinstance(ts_raw, float):
            ts = int(ts_raw)
        elif isinstance(ts_raw, str):
            ts = int(ts_raw.strip() or "0")
        else:
            ts = int(ts_raw)
    except STATE_VALUE_COERCION_ERRORS:
        return int(default)

    if ts <= 0:
        return int(default)

    if ts < 100_000_000_000:
        return int(ts * 1000)

    if ts >= 100_000_000_000_000_000:
        return int(ts // 1_000_000)
    if ts >= 100_000_000_000_000:
        return int(ts // 1000)

    return int(ts)


def extract_ts_field(payload: dict[str, Any]) -> Any:
    if "tsMs" in payload:
        return payload.get("tsMs")
    if "ts" in payload:
        return payload.get("ts")
    if "ts_ms" in payload:
        return payload.get("ts_ms")
    return None


__all__ = [
    "build_cross_state_meta",
    "build_intra_state_route_meta",
    "build_state_validation_meta",
    "coerce_inbound_ts_ms",
    "extract_ts_field",
]
