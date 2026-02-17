from __future__ import annotations

from typing import Any


def coerce_json_value(value: Any) -> Any:
    """
    Convert runtime values into JSON-serializable primitives.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [coerce_json_value(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = coerce_json_value(item)
        return out

    try:
        dumped = value.model_dump(mode="json")
        return coerce_json_value(dumped)
    except AttributeError:
        pass
    except TypeError:
        pass

    try:
        inner = value.root
        return coerce_json_value(inner)
    except AttributeError:
        pass

    return str(value)


def coerce_json_dict(value: Any) -> dict[str, Any]:
    coerced = coerce_json_value(value)
    if isinstance(coerced, dict):
        return coerced
    return {"value": coerced}
