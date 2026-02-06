from __future__ import annotations

from typing import Any


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
    except Exception:
        return int(default)

    if ts <= 0:
        return int(default)

    # Heuristic: epoch seconds are ~1e9, epoch ms are ~1e12 (2026).
    if ts < 100_000_000_000:
        return int(ts * 1000)

    # Heuristic: epoch microseconds ~1e15, nanoseconds ~1e18.
    if ts >= 100_000_000_000_000_000:
        return int(ts // 1_000_000)
    if ts >= 100_000_000_000_000:
        return int(ts // 1000)

    return int(ts)


def extract_ts_field(payload: dict[str, Any]) -> Any:
    # Back-compat: tolerate alternative keys used by ad-hoc writers.
    if "ts" in payload:
        return payload.get("ts")
    if "ts_ms" in payload:
        return payload.get("ts_ms")
    if "tsMs" in payload:
        return payload.get("tsMs")
    return None


def parse_state_key(key: str) -> tuple[str, str] | None:
    """
    Parse a KV key in the form: nodes.<nodeId>.state.<field...>
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

