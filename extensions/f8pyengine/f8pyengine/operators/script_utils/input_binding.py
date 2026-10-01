from __future__ import annotations

from f8pysdk.specs import data_port_value_schema

import ast
import keyword
import logging
import re
from typing import Any, Literal

import msgspec
from f8pysdk.specs import (
    F8AnyTypeSchema,
    F8ArrayTypeSchema,
    F8BooleanTypeSchema,
    F8ComplexObjectTypeSchema,
    F8DataPortSpec,
    F8IntegerTypeSchema,
    F8NullTypeSchema,
    F8NumberTypeSchema,
    F8StringTypeSchema,
)

from .state_binding import ValueAdapter

logger = logging.getLogger(__name__)

InputMode = Literal["raw_dict", "input_view", "msgspec_struct"]
INPUT_MODE_RAW_DICT: InputMode = "raw_dict"
INPUT_MODE_INPUT_VIEW: InputMode = "input_view"
INPUT_MODE_MSGSPEC_STRUCT: InputMode = "msgspec_struct"
VALID_INPUT_MODES: tuple[InputMode, ...] = (
    INPUT_MODE_RAW_DICT,
    INPUT_MODE_INPUT_VIEW,
    INPUT_MODE_MSGSPEC_STRUCT,
)

_DICT_STYLE_INPUT_METHODS: frozenset[str] = frozenset({"get", "keys", "items", "values", "to_dict"})
_SCRIPT_INPUT_STYLE_INFER_ERRORS = (SyntaxError, TypeError, ValueError)


class _MappedInputsView:
    __slots__ = ("_source", "_attr_to_raw")

    def __init__(self, source: dict[str, Any], *, attr_to_raw: dict[str, str], raw_to_attr: dict[str, str]) -> None:
        del raw_to_attr
        self._source = source
        self._attr_to_raw = attr_to_raw

    def _resolve_key(self, key: object) -> str | None:
        key_s = str(key or "")
        if key_s in self._source:
            return key_s
        raw = self._attr_to_raw.get(key_s)
        if raw is not None:
            return raw
        return None

    def __getitem__(self, key: str) -> Any:
        raw = self._resolve_key(key)
        if raw is None:
            raise KeyError(str(key))
        return ValueAdapter.wrap(self._source.get(raw))

    def get(self, key: str, default: Any = None) -> Any:
        raw = self._resolve_key(key)
        if raw is None:
            return default
        return ValueAdapter.wrap(self._source.get(raw))

    def __getattr__(self, name: str) -> Any:
        raw = self._attr_to_raw.get(str(name or ""))
        if raw is None:
            raise AttributeError(f"Unknown attribute: {name}")
        return ValueAdapter.wrap(self._source.get(raw))

    def __contains__(self, key: object) -> bool:
        return self._resolve_key(key) is not None

    def _iter_keys(self):
        seen: set[str] = set()
        for raw in self._source.keys():
            key = str(raw)
            if key in seen:
                continue
            seen.add(key)
            yield key
        for attr in self._attr_to_raw.keys():
            if attr in seen:
                continue
            seen.add(attr)
            yield attr

    def keys(self):
        return tuple(self._iter_keys())

    def items(self):
        return ((k, self.get(k)) for k in self._iter_keys())

    def values(self):
        return (self.get(k) for k in self._iter_keys())

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def to_dict(self) -> dict[str, Any]:
        return {k: ValueAdapter.unwrap(self.get(k)) for k in self._iter_keys()}

    def __repr__(self) -> str:
        return repr(self.to_dict())

    def __str__(self) -> str:
        return str(self.to_dict())


def coerce_input_mode(raw: Any, *, default: InputMode = INPUT_MODE_INPUT_VIEW) -> InputMode:
    text = str(raw or "").strip().lower().replace("-", "_")
    if text in VALID_INPUT_MODES:
        return text  # type: ignore[return-value]
    return default


def _is_valid_identifier(name: str) -> bool:
    return bool(name) and name.isidentifier() and not keyword.iskeyword(name)


def _normalize_attr_name(raw_name: str, *, fallback: str) -> str:
    text = str(raw_name or "").strip()
    if _is_valid_identifier(text):
        return text
    text = re.sub(r"[^A-Za-z0-9_]", "_", text)
    if not text:
        text = fallback
    if text[0].isdigit():
        text = f"_{text}"
    if keyword.iskeyword(text):
        text = f"{text}_"
    if not _is_valid_identifier(text):
        text = fallback
        if keyword.iskeyword(text):
            text = f"{text}_"
    return text


def script_uses_inputs_object_access(code: str) -> bool:
    """
    Returns True when script body appears to rely on dot-style inputs access
    (e.g. inputs.msg).
    """
    try:
        module = ast.parse(str(code or ""), mode="exec")
    except SyntaxError:
        return True

    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_name = str(node.name or "")
            if fn_name == "onMsg":
                param_name = _inputs_param_name(node, expected_pos=1)
                if param_name and _function_uses_dot_inputs_access(node, param_name):
                    return True
            elif fn_name == "onExec":
                param_name = _inputs_param_name(node, expected_pos=2)
                if param_name and _function_uses_dot_inputs_access(node, param_name):
                    return True
    return False


def _inputs_param_name(node: ast.FunctionDef | ast.AsyncFunctionDef, *, expected_pos: int) -> str | None:
    pos_args = list(node.args.posonlyargs) + list(node.args.args)
    if expected_pos >= len(pos_args):
        return None
    raw_name = str(pos_args[expected_pos].arg or "").strip()
    return raw_name or None


def _function_uses_dot_inputs_access(node: ast.FunctionDef | ast.AsyncFunctionDef, param_name: str) -> bool:
    alias_names: set[str] = {param_name}
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign):
            continue
        if len(child.targets) != 1:
            continue
        target = child.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = child.value
        if isinstance(value, ast.Name) and str(value.id or "") in alias_names:
            alias_names.add(str(target.id or ""))

    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            fn_name = str(child.func.id or "")
            if fn_name in {"getattr", "setattr", "hasattr", "delattr"} and child.args:
                first_arg = child.args[0]
                if isinstance(first_arg, ast.Name) and str(first_arg.id or "") in alias_names:
                    return True
        if not isinstance(child, ast.Attribute):
            continue
        base = child.value
        if not isinstance(base, ast.Name):
            continue
        if str(base.id or "") not in alias_names:
            continue
        attr_name = str(child.attr or "")
        if attr_name in _DICT_STYLE_INPUT_METHODS:
            continue
        return True
    return False


def infer_script_input_style(code: str) -> str:
    try:
        return "dot" if script_uses_inputs_object_access(code) else "mapping"
    except _SCRIPT_INPUT_STYLE_INFER_ERRORS as exc:
        logger.debug("failed to infer python script input style", exc_info=exc)
        return "unknown"


class _InputsModelBuilder:
    def __init__(self, *, node_id: str) -> None:
        self._node_id = str(node_id or "")
        self._struct_seq = 0
        self._schema_struct_cache: dict[int, type[Any]] = {}
        self._schema_building: set[int] = set()
        self.warnings: list[str] = []
        self.root_attr_to_raw: dict[str, str] = {}
        self.root_raw_to_attr: dict[str, str] = {}

    def build_root(self, data_in_ports: list[F8DataPortSpec]) -> type[Any]:
        fields: list[tuple[Any, ...]] = []
        used_attrs: set[str] = set()
        for index, port in enumerate(list(data_in_ports or [])):
            raw_name = str(port.name or "").strip()
            if not raw_name:
                continue
            attr_name = self._unique_attr_name(raw_name, used_attrs, scope=f"port[{index}]")
            self.root_attr_to_raw[attr_name] = raw_name
            self.root_raw_to_attr[raw_name] = attr_name
            field_type = self._schema_to_type(data_port_value_schema(port), hint=f"Port_{attr_name}")
            optional_type = field_type | None
            if attr_name == raw_name:
                fields.append((attr_name, optional_type, msgspec.field(default=None)))
            else:
                fields.append((attr_name, optional_type, msgspec.field(name=raw_name, default=None)))
        return self._defstruct("F8Inputs", fields)

    def _defstruct(self, prefix: str, fields: list[tuple[Any, ...]]) -> type[Any]:
        self._struct_seq += 1
        struct_name = f"{prefix}_{self._struct_seq}"
        return msgspec.defstruct(
            struct_name,
            fields,
            kw_only=True,
            module=__name__,
        )

    def _unique_attr_name(self, raw_name: str, used: set[str], *, scope: str) -> str:
        base_name = _normalize_attr_name(raw_name, fallback="field")
        if base_name not in used:
            used.add(base_name)
            return base_name
        suffix = 1
        while True:
            candidate = f"{base_name}_{suffix}"
            if candidate not in used:
                used.add(candidate)
                self.warnings.append(
                    f"input model field name collision ({scope}): raw='{raw_name}' mapped to '{candidate}'"
                )
                return candidate
            suffix += 1

    def _schema_to_type(self, schema: Any, *, hint: str) -> Any:
        if schema is None or isinstance(schema, msgspec.UnsetType):
            return Any
        if isinstance(schema, dict):
            schema_type = str(schema.get("type") or "").strip().lower()
            if schema_type == "string":
                return str
            if schema_type == "number":
                return float
            if schema_type == "integer":
                return int
            if schema_type == "boolean":
                return bool
            if schema_type == "null":
                return type(None)
            if schema_type == "any":
                return Any
            if schema_type == "array":
                return list[self._schema_to_type(schema.get("items"), hint=f"{hint}_item")]
            if schema_type == "object":
                props = schema.get("properties")
                if not isinstance(props, dict) or not props:
                    return dict[str, Any]
                additional = schema.get("additionalProperties")
                if additional is True:
                    return dict[str, Any]
                return self._schema_object_to_struct_dict(
                    schema_id=id(schema),
                    properties=props,
                    required_values=schema.get("required"),
                    hint=hint,
                )
            return Any
        if isinstance(schema, F8StringTypeSchema):
            return str
        if isinstance(schema, F8NumberTypeSchema):
            return float
        if isinstance(schema, F8IntegerTypeSchema):
            return int
        if isinstance(schema, F8BooleanTypeSchema):
            return bool
        if isinstance(schema, F8NullTypeSchema):
            return type(None)
        if isinstance(schema, F8AnyTypeSchema):
            return Any
        if isinstance(schema, F8ArrayTypeSchema):
            return list[self._schema_to_type(schema.items, hint=f"{hint}_item")]
        if isinstance(schema, F8ComplexObjectTypeSchema):
            if bool(schema.additionalProperties):
                return dict[str, Any]
            properties = dict(schema.properties or {})
            if not properties:
                return dict[str, Any]
            return self._schema_object_to_struct_dict(
                schema_id=id(schema),
                properties=properties,
                required_values=schema.required,
                hint=hint,
            )
        return Any

    def _schema_object_to_struct_dict(
        self,
        *,
        schema_id: int,
        properties: dict[str, Any],
        required_values: Any,
        hint: str,
    ) -> type[Any]:
        if schema_id in self._schema_struct_cache:
            return self._schema_struct_cache[schema_id]
        if schema_id in self._schema_building:
            return dict[str, Any]
        self._schema_building.add(schema_id)
        try:
            required_names = self._required_name_set(required_values)
            fields: list[tuple[Any, ...]] = []
            used_attrs: set[str] = set()
            for key in sorted(properties.keys()):
                raw_name = str(key or "").strip()
                if not raw_name:
                    continue
                attr_name = self._unique_attr_name(raw_name, used_attrs, scope=f"{hint}.{raw_name}")
                field_type = self._schema_to_type(properties.get(key), hint=f"{hint}_{attr_name}")
                is_required = raw_name in required_names
                if is_required:
                    if attr_name == raw_name:
                        fields.append((attr_name, field_type))
                    else:
                        fields.append((attr_name, field_type, msgspec.field(name=raw_name)))
                    continue
                optional_type = field_type | None
                if attr_name == raw_name:
                    fields.append((attr_name, optional_type, msgspec.field(default=None)))
                else:
                    fields.append((attr_name, optional_type, msgspec.field(name=raw_name, default=None)))
            struct_type = self._defstruct(_normalize_attr_name(hint, fallback="InputObject"), fields)
            self._schema_struct_cache[schema_id] = struct_type
            return struct_type
        finally:
            self._schema_building.discard(schema_id)

    @staticmethod
    def _required_name_set(required_values: Any) -> set[str]:
        if isinstance(required_values, msgspec.UnsetType) or required_values is None:
            return set()
        if not isinstance(required_values, (list, tuple, set)):
            return set()
        out: set[str] = set()
        for item in required_values:
            name = str(item or "").strip()
            if name:
                out.add(name)
        return out


class InputBinding:
    def __init__(self, *, node_id: str, data_in_ports: list[F8DataPortSpec], mode: InputMode) -> None:
        self._node_id = str(node_id or "")
        self._mode: InputMode = coerce_input_mode(mode)
        model_builder = _InputsModelBuilder(node_id=self._node_id)
        self._model_type = model_builder.build_root(list(data_in_ports or []))
        self.warnings: tuple[str, ...] = tuple(model_builder.warnings)
        self._root_attr_to_raw = dict(model_builder.root_attr_to_raw)
        self._root_raw_to_attr = dict(model_builder.root_raw_to_attr)

    @property
    def mode(self) -> InputMode:
        return self._mode

    def set_mode(self, mode: InputMode) -> None:
        next_mode = coerce_input_mode(mode)
        if next_mode != self._mode:
            logger.debug("[%s:python_script] switch input mode: %s -> %s", self._node_id, self._mode, next_mode)
        self._mode = next_mode

    def decode(self, inputs: dict[str, Any]) -> Any:
        if self._mode == INPUT_MODE_RAW_DICT:
            return inputs
        if self._mode == INPUT_MODE_INPUT_VIEW:
            return _MappedInputsView(
                inputs,
                attr_to_raw=self._root_attr_to_raw,
                raw_to_attr=self._root_raw_to_attr,
            )
        return msgspec.convert(inputs, type=self._model_type)
