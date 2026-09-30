from __future__ import annotations

from f8pysdk.specs import data_port_value_schema

from f8studio_server.errors import InvalidRequestError, NotFoundError

import keyword
from typing import Any

import msgspec
from f8pysdk.generated import (
    F8ArrayTypeSchema,
    F8BooleanTypeSchema,
    F8ComplexObjectTypeSchema,
    F8DataTypeSchema,
    F8IntegerTypeSchema,
    F8NullTypeSchema,
    F8NumberTypeSchema,
    F8StringTypeSchema,
)
from f8studio_core.graph import GraphNode

from .editor import EditorSupportFile


def _python_type(schema: F8DataTypeSchema) -> str:
    if isinstance(schema, F8StringTypeSchema):
        base = "str"
    elif isinstance(schema, F8IntegerTypeSchema):
        base = "int"
    elif isinstance(schema, F8NumberTypeSchema):
        base = "float"
    elif isinstance(schema, F8BooleanTypeSchema):
        base = "bool"
    elif isinstance(schema, F8NullTypeSchema):
        return "None"
    elif isinstance(schema, F8ArrayTypeSchema):
        return f"list[{_python_type(schema.items)}]"
    elif isinstance(schema, F8ComplexObjectTypeSchema):
        return "dict[str, Any]"
    else:
        return "Any"
    enum = schema.enum
    if isinstance(enum, msgspec.UnsetType) or not enum:
        return base
    values: list[Any] = list(enum)
    if all(isinstance(value, (str, bool, int, float)) for value in values):
        return f"Literal[{', '.join(repr(value) for value in values)}]"
    return base


def _dynamic_module(type_name: str, fields: list[tuple[str, F8DataTypeSchema]], *, mapping: bool) -> str:
    if not type_name.isidentifier() or keyword.iskeyword(type_name):
        raise InvalidRequestError(f"invalid editor binding type name: {type_name}")
    lines = ["from __future__ import annotations", "from typing import Any, Literal, Protocol", "", f"class {type_name}(Protocol):"]
    if mapping:
        lines.append("    def __getitem__(self, key: str) -> Any: ...")
    for name, schema in fields:
        if name.isidentifier() and not keyword.iskeyword(name):
            lines.append(f"    {name}: {_python_type(schema)}")
    if len(lines) == 4:
        lines.append("    ...")
    return "\n".join(lines) + "\n"


def editor_support_files(node: GraphNode, field_name: str) -> tuple[EditorSupportFile, ...]:
    state_fields = node.spec.stateFields
    fields = () if isinstance(state_fields, msgspec.UnsetType) else state_fields
    field = next((item for item in fields if item.name == field_name), None)
    if field is None:
        raise NotFoundError(f"code field not found: {node.node_id}.{field_name}")
    control = field.control
    if isinstance(control, msgspec.UnsetType) or control.kind.value != "code":
        raise InvalidRequestError(f"state field is not a code editor: {node.node_id}.{field_name}")
    assist = field.editorAssist
    if isinstance(assist, msgspec.UnsetType) or isinstance(assist.python, msgspec.UnsetType):
        return ()
    python = assist.python
    files = {path: content for path, content in python.support_files.items()}
    bindings = python.dynamic_bindings
    if not isinstance(bindings, msgspec.UnsetType):
        inputs = bindings.inputs
        if not isinstance(inputs, msgspec.UnsetType) and inputs.enabled:
            module = inputs.module_name
            type_name = inputs.type_name
            if isinstance(module, msgspec.UnsetType) or isinstance(type_name, msgspec.UnsetType) or not module.isidentifier():
                raise InvalidRequestError("invalid inputs editor binding metadata")
            data_ports = node.spec.dataInPorts
            ports = () if isinstance(data_ports, msgspec.UnsetType) else data_ports
            files[f"{module}.pyi"] = _dynamic_module(
                type_name, [(port.name, data_port_value_schema(port)) for port in ports], mapping=True,
            )
        states = bindings.states
        if not isinstance(states, msgspec.UnsetType) and states.enabled:
            module = states.module_name
            type_name = states.type_name
            if isinstance(module, msgspec.UnsetType) or isinstance(type_name, msgspec.UnsetType) or not module.isidentifier():
                raise InvalidRequestError("invalid states editor binding metadata")
            files[f"{module}.pyi"] = _dynamic_module(
                type_name, [(item.name, item.valueSchema) for item in fields], mapping=True,
            )
    for path in tuple(files):
        if path.endswith(".pyi"):
            files.setdefault(f"{path[:-1]}", "# Provided by the service runtime.\n")
    return tuple(EditorSupportFile(path=path, content=content) for path, content in sorted(files.items()))
