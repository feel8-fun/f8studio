from __future__ import annotations

from typing import Any


# QGraphicsItem custom data keys (keep stable: stored on port view items).
# Keep these separate from keys used by GenericNode layout widgets.
PORT_KIND_DATA_KEY = 10021
PORT_SCHEMA_SIG_DATA_KEY = 10022


def infer_port_kind(port_name: str) -> str | None:
    name = str(port_name)
    if name.startswith("[E]") or name.endswith("[E]"):
        return "exec"
    if name.startswith("[D]") or name.endswith("[D]"):
        return "data"
    if name.startswith("[S]") or name.endswith("[S]"):
        return "state"
    return None


def schema_signature(schema: Any) -> dict[str, Any] | None:
    if schema is None:
        return None
    schema_type = getattr(schema, "type", None)
    if schema_type is None:
        return None

    try:
        t = schema_type.value  # enum
    except Exception:
        t = str(schema_type)
    t = str(t)
    if t == "any":
        return {"type": "any"}
    if t == "array":
        return {"type": "array", "items": schema_signature(getattr(schema, "items", None))}
    if t == "object":
        props = getattr(schema, "properties", None) or {}
        return {"type": "object", "properties": {str(k): schema_signature(v) for k, v in dict(props).items()}}
    return {"type": t}


def schema_is_superset(out_sig: dict[str, Any] | None, in_sig: dict[str, Any] | None) -> bool:
    """
    Return True if `out_sig` is a schema superset of `in_sig`.

    Rules (intentionally shape-based, ignores constraints like min/max/enum):
    - if either side is `any` => compatible.
    - `number` is treated as a superset of `integer`.
    - `object`: output must contain all input properties (recursive).
    - `array`: items must be compatible (recursive).
    - primitives: types must match (except number>=integer).
    """
    if out_sig is None or in_sig is None:
        return True

    out_t = str(out_sig.get("type"))
    in_t = str(in_sig.get("type"))

    if out_t == "any" or in_t == "any":
        return True
    if out_t == "number" and in_t == "integer":
        return True
    if out_t != in_t:
        return False

    if out_t == "array":
        return schema_is_superset(out_sig.get("items"), in_sig.get("items"))
    if out_t == "object":
        out_props = out_sig.get("properties") or {}
        in_props = in_sig.get("properties") or {}
        for key, in_prop in dict(in_props).items():
            if key not in out_props:
                return False
            if not schema_is_superset(out_props.get(key), in_prop):
                return False
        return True

    return True
