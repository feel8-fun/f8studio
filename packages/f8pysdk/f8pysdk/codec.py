from __future__ import annotations

from copy import deepcopy
import json
import math
import numbers
from typing import Any, TypeVar, cast

import msgspec

T = TypeVar("T")
_UNSET_SENTINEL = object()
_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "no", "off"})


def _json_value(value: Any, seen: set[int]) -> Any:
    if isinstance(value, msgspec.UnsetType):
        return _UNSET_SENTINEL
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    identity = id(value)
    if identity in seen:
        raise ValueError("cyclic value cannot be serialized")
    seen.add(identity)
    try:
        if isinstance(value, msgspec.Struct):
            # Generic serialization is the intentional dynamic-schema boundary.
            result = {
                field.encode_name: converted
                for field in msgspec.structs.fields(value)
                if (converted := _json_value(getattr(value, field.name), seen)) is not _UNSET_SENTINEL
            }
            config = value.__struct_config__
            if config.tag_field is not None and config.tag is not None:
                result[config.tag_field] = config.tag
            return result
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, (str, int, float, bool, type(None))):
                    raise TypeError(f"unsupported JSON key type: {type(key).__name__}")
                converted = _json_value(item, seen)
                if converted is not _UNSET_SENTINEL:
                    result[str(key)] = converted
            return result
        if isinstance(value, (list, tuple, set)):
            return [converted for item in value if (converted := _json_value(item, seen)) is not _UNSET_SENTINEL]
        # NumPy scalar types expose item(); this is not application attribute dispatch.
        item_fn = getattr(value, "item", None)
        if callable(item_fn):
            return _json_value(item_fn(), seen)
        try:
            converted = msgspec.to_builtins(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"unsupported JSON value type: {type(value).__name__}") from exc
        if converted is value:
            raise TypeError(f"unsupported JSON value type: {type(value).__name__}")
        return _json_value(converted, seen)
    finally:
        seen.remove(identity)


def validate_as(model_type: type[T], value: Any) -> T:
    return msgspec.convert(value, type=model_type)


def dump_json(value: Any) -> Any:
    converted = _json_value(value, set())
    return None if converted is _UNSET_SENTINEL else converted


def copy_model(value: T, *, update: dict[str, Any] | None = None, deep: bool = False) -> T:
    if deep:
        value = deepcopy(value)
    update_obj = update
    if isinstance(value, msgspec.Struct):
        if isinstance(update_obj, dict):
            return msgspec.structs.replace(value, **update_obj)
        return msgspec.structs.replace(value)
    if isinstance(value, dict):
        copied = dict(value)
        if isinstance(update_obj, dict):
            copied.update(update_obj)
        return cast(T, copied)
    if isinstance(update_obj, dict) and (value is None or isinstance(value, msgspec.UnsetType)):
        return cast(T, dict(update_obj))
    raise TypeError(f"unsupported model copy type: {type(value).__name__}")


def unwrap_json_value(value: Any) -> Any:
    """
    Convert possible schema/value wrappers into plain Python JSON-like values.
    """
    if value is None or isinstance(value, (str, int, float, bool, list, dict, tuple)):
        return value
    try:
        return dump_json(value)
    except (AttributeError, TypeError, ValueError):
        return value


def parse_bool(value: Any, *, empty_as_false: bool = False) -> bool | None:
    """Best-effort bool parse. Return None when the input is not interpretable."""
    normalized = unwrap_json_value(value)
    if isinstance(normalized, bool):
        return normalized
    if isinstance(normalized, (int, float)):
        return bool(normalized)
    if normalized is None:
        return False if empty_as_false else None
    text = str(normalized).strip().lower()
    if not text:
        return False if empty_as_false else None
    if text in _TRUE_STRINGS:
        return True
    if text in _FALSE_STRINGS:
        return False
    return None


def coerce_bool(value: Any, *, default: bool, empty_as_false: bool = False) -> bool:
    """Parse a bool and fall back to ``default`` when parsing fails."""
    parsed = parse_bool(value, empty_as_false=empty_as_false)
    if parsed is None:
        return bool(default)
    return parsed


def coerce_flag(value: Any, *, default: bool) -> bool:
    return coerce_bool(value, default=default, empty_as_false=True)


def parse_int(value: Any, *, allow_bool: bool = True) -> int | None:
    """Best-effort int parse. Return None when the input is not interpretable."""
    normalized = unwrap_json_value(value)
    if normalized is None:
        return None
    if isinstance(normalized, bool):
        if not allow_bool:
            return None
        return int(normalized)
    try:
        return int(normalized)
    except (TypeError, ValueError):
        return None


def coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
    allow_bool: bool = True,
) -> int:
    """Parse an int, then fall back to ``default`` and optional bounds."""
    parsed = parse_int(value, allow_bool=allow_bool)
    out = int(default) if parsed is None else int(parsed)
    if minimum is not None and out < int(minimum):
        out = int(minimum)
    if maximum is not None and out > int(maximum):
        out = int(maximum)
    return out


def parse_float(value: Any, *, allow_bool: bool = False, finite_only: bool = False) -> float | None:
    """Best-effort float parse. Return None when the input is not interpretable."""
    normalized = unwrap_json_value(value)
    if normalized is None:
        return None
    if isinstance(normalized, bool) and not allow_bool:
        return None
    try:
        out = float(normalized)
    except (TypeError, ValueError):
        return None
    if finite_only and not math.isfinite(out):
        return None
    return float(out)


def parse_number(value: Any) -> float | None:
    """Parse a finite non-bool number, or return None when parsing fails."""
    return parse_float(value, allow_bool=False, finite_only=True)


def parse_number_sequence(value: Any) -> tuple[float, ...] | None:
    """Parse a scalar/sequence into finite floats, or return None when parsing fails."""
    normalized = unwrap_json_value(value)
    if normalized is None or isinstance(normalized, bool):
        return None
    if isinstance(normalized, (list, tuple)):
        if not normalized:
            return ()
        out: list[float] = []
        for item in normalized:
            parsed = parse_number(item)
            if parsed is None:
                return None
            out.append(float(parsed))
        return tuple(out)
    parsed = parse_number(normalized)
    if parsed is None:
        return None
    return (float(parsed),)


def coerce_float(
    value: Any,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_bool: bool = True,
    finite_only: bool = False,
) -> float:
    """Parse a float, then fall back to ``default`` and optional bounds."""
    parsed = parse_float(value, allow_bool=allow_bool, finite_only=finite_only)
    out = float(default) if parsed is None else float(parsed)
    if minimum is not None and out < float(minimum):
        out = float(minimum)
    if maximum is not None and out > float(maximum):
        out = float(maximum)
    return float(out)


def coerce_number(
    value: Any,
    *,
    default: float,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse a finite non-bool number, then fall back to ``default`` and bounds."""
    return coerce_float(value, default=default, minimum=minimum, maximum=maximum, allow_bool=False, finite_only=True)


def coerce_str(value: Any, *, default: str = "") -> str:
    """Return trimmed text, or ``default`` when the input is empty/missing."""
    normalized = unwrap_json_value(value)
    if normalized is None:
        return str(default)
    text = str(normalized).strip()
    if text:
        return text
    return str(default)


def parse_str_list(
    value: Any,
    *,
    allow_json_string: bool = False,
    allow_mapping_values: bool = False,
) -> list[str] | None:
    """Parse a list of strings, or return None when the input shape is unsupported."""
    normalized = unwrap_json_value(value)
    if isinstance(normalized, (list, tuple)):
        out: list[str] = []
        for item in normalized:
            text = coerce_str(item, default="")
            if text:
                out.append(text)
        return out
    if allow_mapping_values and isinstance(normalized, dict):
        items = list(normalized.items())

        def _sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
            key = item[0]
            parsed = parse_int(key, allow_bool=False)
            if parsed is not None:
                return (0, f"{parsed:08d}")
            return (1, coerce_str(key, default=""))

        items.sort(key=_sort_key)
        out2: list[str] = []
        for _key, item in items:
            text = coerce_str(item, default="")
            if text:
                out2.append(text)
        return out2
    if allow_json_string and isinstance(normalized, str):
        raw = normalized.strip()
        if not raw:
            return []
        try:
            parsed_json = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parse_str_list(parsed_json, allow_mapping_values=allow_mapping_values)
    return None


def _msgpack_enc_hook(obj: Any) -> Any:
    if isinstance(obj, msgspec.UnsetType):
        return None
    normalized = dump_json(obj)
    if normalized is obj:
        raise TypeError(f"unsupported msgpack object: {type(obj).__name__}")
    return normalized


_MSGPACK_ENCODER = msgspec.msgpack.Encoder(enc_hook=_msgpack_enc_hook)
_MSGPACK_DICT_DECODER = msgspec.msgpack.Decoder(type=dict[str, Any])


def encode_obj(obj: Any) -> bytes:
    try:
        return _MSGPACK_ENCODER.encode(obj)
    except (TypeError, ValueError, msgspec.EncodeError) as exc:
        raise ValueError(f"msgpack encode failed: {exc}") from exc


def decode_obj(raw: bytes) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        decoded = _MSGPACK_DICT_DECODER.decode(raw)
    except (TypeError, ValueError, msgspec.DecodeError, msgspec.ValidationError) as exc:
        raise ValueError(f"msgpack decode failed: {exc}") from exc
    return decoded


def decode_as(raw: bytes, model_type: type[T]) -> T:
    if not raw:
        raise ValueError("msgpack decode failed: empty payload")
    try:
        return msgspec.msgpack.decode(raw, type=model_type)
    except (TypeError, ValueError, msgspec.DecodeError, msgspec.ValidationError) as exc:
        raise ValueError(f"msgpack decode failed: {exc}") from exc

__all__ = [
    "coerce_bool",
    "coerce_flag",
    "coerce_float",
    "coerce_int",
    "coerce_number",
    "coerce_str",
    "copy_model",
    "decode_as",
    "decode_obj",
    "dump_json",
    "encode_obj",
    "parse_bool",
    "parse_float",
    "parse_int",
    "parse_number",
    "parse_number_sequence",
    "parse_str_list",
    "unwrap_json_value",
    "validate_as",
]
