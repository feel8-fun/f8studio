from __future__ import annotations

from typing import Any, cast


def unwrap_json_value(value: Any) -> Any:
    """
    Unwrap generated JSON RootModel wrappers into plain Python values.

    Today this is primarily `generated.F8JsonValue` (pydantic RootModel).
    """
    if value is None or isinstance(value, (str, int, float, bool, list, dict, tuple)):
        return value

    # Prefer explicit support for our generated wrapper to keep behavior stable.
    try:
        from .generated import F8JsonValue  # local import to avoid heavy imports at module load
    except ImportError:
        F8JsonValue = None  # type: ignore[assignment]

    if F8JsonValue is not None:
        try:
            if isinstance(value, F8JsonValue):
                return value.root
        except TypeError:
            # Defensive: `F8JsonValue` should always be a proper type.
            pass

    # Fallback: RootModel-like wrappers with a `.root` attribute.
    try:
        return cast(Any, value).root
    except AttributeError:
        return value
