from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import msgspec

from .state_binding import ValueAdapter


def normalize_script_output_value(value: Any, *, _seen: set[int] | None = None) -> Any:
    if type(value) in (str, int, float, bool, type(None)):
        return value
    if isinstance(value, msgspec.Struct):
        return msgspec.to_builtins(value)
    value = ValueAdapter.unwrap(value)
    if type(value) in (str, int, float, bool, type(None)):
        return value
    if isinstance(value, dict):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return None
        _seen.add(value_id)
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[str(key)] = normalize_script_output_value(item, _seen=_seen)
        _seen.discard(value_id)
        return out
    if isinstance(value, list):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        out_list = [normalize_script_output_value(item, _seen=_seen) for item in value]
        _seen.discard(value_id)
        return out_list
    if isinstance(value, tuple):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return ()
        _seen.add(value_id)
        out_tuple = tuple(normalize_script_output_value(item, _seen=_seen) for item in value)
        _seen.discard(value_id)
        return out_tuple
    if isinstance(value, set):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        out_set = [normalize_script_output_value(item, _seen=_seen) for item in value]
        _seen.discard(value_id)
        return out_set
    return value


def normalize_script_output_value_fast(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return normalize_script_output_value(value)


@dataclass(frozen=True)
class ScriptOutputPorts:
    data_out_ports: frozenset[str]
    single_data_out_port: str | None
    has_out_port: bool


def extract_script_outputs(
    result: Any,
    *,
    ports: ScriptOutputPorts,
    normalize_one: Callable[[Any], Any] = normalize_script_output_value_fast,
) -> dict[str, Any]:
    if result is None:
        return {}

    data_out_ports = ports.data_out_ports
    outputs: dict[str, Any] = {}
    if isinstance(result, dict):
        raw_outputs = result.get("outputs")
        if isinstance(raw_outputs, dict):
            single_out_port = ports.single_data_out_port
            if single_out_port is not None and len(raw_outputs) == 1:
                value = raw_outputs.get(single_out_port, None)
                if single_out_port in raw_outputs:
                    return {single_out_port: normalize_one(value)}
            # Fast-path for dominant script pattern: {"outputs": {"tcode": value}}.
            if len(raw_outputs) == 1:
                for raw_key, raw_value in raw_outputs.items():
                    if isinstance(raw_key, str):
                        if raw_key in data_out_ports:
                            return {raw_key: normalize_one(raw_value)}
                        return {}
                    key_s = str(raw_key)
                    if key_s in data_out_ports:
                        return {key_s: normalize_one(raw_value)}
                    return {}
            for key, value in raw_outputs.items():
                key_s = str(key)
                if key_s in data_out_ports:
                    outputs[key_s] = normalize_one(value)
            return outputs

        if "outputs" in result:
            raise ValueError("script return field 'outputs' must be a dict")
        raise ValueError("script dict return must include an 'outputs' dict")

    if ports.has_out_port:
        outputs["out"] = normalize_one(result)
    return outputs
