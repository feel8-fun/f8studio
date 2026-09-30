#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Schema:
    name: str
    obj: dict[str, Any]


def _load_schemas(protocol_yml: Path) -> dict[str, Schema]:
    root = yaml.safe_load(protocol_yml.read_text(encoding="utf-8"))
    schemas = root.get("components", {}).get("schemas", {})
    out: dict[str, Schema] = {}
    for name, obj in schemas.items():
        if isinstance(name, str) and isinstance(obj, dict):
            out[name] = Schema(name=name, obj=obj)
    return out


def _cpp_ident(s: str) -> str:
    # Keep it simple: schemas/titles are already stable identifiers in this repo.
    # We only need to fix a few invalid patterns.
    s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in str(s))
    if not s:
        return "X"
    if s[0].isdigit():
        s = "X_" + s
    if s in _CPP_KEYWORDS:
        s = s + "_"
    return s


def _cpp_escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


_CPP_KEYWORDS = {
    # common keywords (C++17)
    "alignas",
    "alignof",
    "and",
    "and_eq",
    "asm",
    "atomic_cancel",
    "atomic_commit",
    "atomic_noexcept",
    "auto",
    "bitand",
    "bitor",
    "bool",
    "break",
    "case",
    "catch",
    "char",
    "char8_t",
    "char16_t",
    "char32_t",
    "class",
    "compl",
    "concept",
    "const",
    "consteval",
    "constexpr",
    "constinit",
    "const_cast",
    "continue",
    "co_await",
    "co_return",
    "co_yield",
    "decltype",
    "default",
    "delete",
    "do",
    "double",
    "dynamic_cast",
    "else",
    "enum",
    "explicit",
    "export",
    "extern",
    "false",
    "float",
    "for",
    "friend",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "mutable",
    "namespace",
    "new",
    "noexcept",
    "not",
    "not_eq",
    "nullptr",
    "operator",
    "or",
    "or_eq",
    "private",
    "protected",
    "public",
    "register",
    "reinterpret_cast",
    "requires",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "static_assert",
    "static_cast",
    "struct",
    "switch",
    "synchronized",
    "template",
    "this",
    "thread_local",
    "throw",
    "true",
    "try",
    "typedef",
    "typeid",
    "typename",
    "union",
    "unsigned",
    "using",
    "virtual",
    "void",
    "volatile",
    "wchar_t",
    "while",
    "xor",
    "xor_eq",
}


def _cpp_member_name(json_key: str) -> str:
    """
    Convert an arbitrary JSON key into a safe C++ member identifier.

    Unknown/extra JSON fields are ignored; this only affects field naming of known schema properties.
    """

    s = str(json_key)
    if s.startswith("$"):
        s = s[1:]
    s = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)
    if not s:
        s = "field"
    if s[0].isdigit():
        s = "field_" + s
    if s in _CPP_KEYWORDS:
        s = s + "_"
    return s


def _enum_member(v: str) -> str:
    return _cpp_ident(v)


@dataclass
class EnumType:
    name: str
    values: list[str]


def _collect_enum_types(schemas: dict[str, Schema]) -> dict[str, EnumType]:
    enums: dict[str, EnumType] = {}

    def add_enum(name: str, values: list[str]) -> None:
        nm = _cpp_ident(name)
        vals = [str(v) for v in values if isinstance(v, (str, int, float, bool))]  # stringify for stability
        vals_s = [str(v) for v in vals]
        if nm in enums:
            if enums[nm].values != vals_s:
                raise ValueError(f"Conflicting enum definitions: {nm}")
            return
        enums[nm] = EnumType(name=nm, values=vals_s)

    # schema-level enums
    for s in schemas.values():
        obj = s.obj
        if isinstance(obj.get("enum"), list):
            add_enum(s.name, obj["enum"])

    # property-level enums (use `title` when present, otherwise derive a stable name)
    for s in schemas.values():
        props = s.obj.get("properties")
        if not isinstance(props, dict):
            continue
        for prop_name, prop_schema in props.items():
            if not isinstance(prop_schema, dict):
                continue
            values = prop_schema.get("enum")
            if not isinstance(values, list):
                continue
            title = prop_schema.get("title")
            if isinstance(title, str) and title.strip():
                add_enum(title.strip(), values)
            else:
                add_enum(f"{s.name}_{prop_name}_Enum", values)

    return enums


def _gen_header(*, schemas: dict[str, Schema], namespace: str) -> str:
    """Generate concrete JSON models; only explicitly open JSON stays untyped."""
    import json

    def flatten(obj: dict[str, Any]) -> dict[str, Any]:
        if "allOf" not in obj:
            return dict(obj)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for part in obj["allOf"]:
            if "$ref" in part:
                part = schemas[part["$ref"].rsplit("/", 1)[-1]].obj
            part = flatten(part)
            if part.get("type") != "object":
                raise ValueError("Only object allOf composition is supported")
            properties.update(part.get("properties", {}))
            required.extend(part.get("required", []))
        return {**obj, "type": "object", "properties": properties, "required": list(dict.fromkeys(required))}

    models = {name: Schema(name, flatten(s.obj)) for name, s in schemas.items()}

    # Lift inline objects into named models so fields remain searchable.
    def lift(obj: dict[str, Any], name: str) -> dict[str, Any]:
        if obj.get("type") == "object" and "properties" in obj:
            if name in models:
                raise ValueError(f"Inline model name collision: {name}")
            models[name] = Schema(name, obj)
            return {"$ref": "#/components/schemas/" + name, **({"nullable": True} if obj.get("nullable") else {})}
        if obj.get("type") == "array":
            return {**obj, "items": lift(obj.get("items", {}), name + "Item")}
        if isinstance(obj.get("additionalProperties"), dict):
            return {**obj, "additionalProperties": lift(obj["additionalProperties"], name + "Value")}
        return obj

    visited: set[str] = set()
    while pending := [name for name in models if name not in visited]:
        for name in pending:
            visited.add(name)
            obj = models[name].obj
            if "properties" in obj:
                obj["properties"] = {
                    key: lift(value, name + "_" + _cpp_ident(key)) for key, value in obj["properties"].items()
                }
    enums = _collect_enum_types(models)
    unions = {
        name: s.obj for name, s in models.items() if name != "F8JsonValue" and ("oneOf" in s.obj or "anyOf" in s.obj)
    }
    objects = {
        name: s.obj
        for name, s in models.items()
        if s.obj.get("type") == "object" and ("properties" in s.obj or s.obj.get("additionalProperties") is False)
    }

    def cpp_type(obj: dict[str, Any], parent: str, field: str) -> str:
        if "anyOf" in obj or "oneOf" in obj:
            alternatives = obj.get("anyOf", obj.get("oneOf"))
            non_null = [part for part in alternatives if part.get("type") != "null"]
            if len(alternatives) == 2 and len(non_null) == 1:
                return "std::optional<" + cpp_type(non_null[0], parent, field) + ">"
            raise ValueError(f"Inline union needs a named discriminator: {parent}.{field}")
        if "$ref" in obj:
            name = obj["$ref"].rsplit("/", 1)[-1]
            if name not in models:
                raise ValueError(f"Unknown schema reference: {name}")
            result = name
        elif "enum" in obj:
            result = _cpp_ident(obj.get("title") or f"{parent}_{field}_Enum")
            if result not in enums:
                raise ValueError(f"Unknown enum: {result}")
        elif obj.get("type") in ("string", "integer", "number", "boolean", "null"):
            result = {
                "string": "std::string",
                "integer": "std::int64_t",
                "number": "double",
                "boolean": "bool",
                "null": "std::nullptr_t",
            }[obj["type"]]
        elif obj.get("type") == "array":
            result = "std::vector<" + cpp_type(obj.get("items", {}), parent, field) + ">"
        elif obj.get("type") == "object":
            extra = obj.get("additionalProperties", True)
            result = (
                "std::map<std::string, "
                + (cpp_type(extra, parent, field) if isinstance(extra, dict) else "F8JsonValue")
                + ">"
            )
        elif not obj or set(obj) <= {"description", "title", "default"}:
            result = "F8JsonValue"
        else:
            raise ValueError(f"Unsupported typed schema at {parent}.{field}: {obj}")
        return f"std::optional<{result}>" if obj.get("nullable") else result

    def refs(value: Any) -> set[str]:
        if isinstance(value, dict):
            result = {value["$ref"].rsplit("/", 1)[-1]} if "$ref" in value else set()
            for child in value.values():
                result |= refs(child)
            return result
        if isinstance(value, list):
            return set().union(*(refs(child) for child in value))
        return set()

    remaining = {name: (refs(obj.get("properties", {})) & objects.keys()) - {name} for name, obj in objects.items()}
    order: list[str] = []
    while remaining:
        ready = sorted(name for name, deps in remaining.items() if not deps)
        if not ready:
            raise ValueError(f"Object cycle requires an explicit recursive union: {remaining}")
        order.extend(ready)
        remaining = {name: deps - set(ready) for name, deps in remaining.items() if name not in ready}
    lines = [
        "// Generated by scripts/gen_cpp_protocol_models.py; do not edit.",
        "#pragma once",
        "#include <cstdint>",
        "#include <limits>",
        "#include <map>",
        "#include <memory>",
        "#include <optional>",
        "#include <string>",
        "#include <stdexcept>",
        "#include <utility>",
        "#include <variant>",
        "#include <vector>",
        "#include <nlohmann/json.hpp>",
        f"namespace {namespace} {{",
        "using F8JsonValue = nlohmann::json;",
        "struct ParseError { std::string code; std::string message; };",
        'inline bool invalid(ParseError& e, const std::string& message) { e.code="INVALID_SCHEMA"; e.message=message; return false; }',
    ]
    for enum in sorted(enums.values(), key=lambda e: e.name):
        lines += [
            f"enum class {enum.name} {{ " + ", ".join(_enum_member(v) for v in enum.values) + " };",
            f"inline std::optional<{enum.name}> parse_{enum.name}(const std::string& s) {{",
        ]
        lines += [f"  if (s == {json.dumps(v)}) return {enum.name}::{_enum_member(v)};" for v in enum.values]
        lines += [
            "  return std::nullopt;",
            "}",
            f"inline void to_json(nlohmann::json& j, {enum.name} v) {{",
            "  switch (v) {",
        ]
        lines += [f"    case {enum.name}::{_enum_member(v)}: j={json.dumps(v)}; return;" for v in enum.values]
        lines += ["  }", '  throw std::invalid_argument("invalid enum value");', "}"]
    lines += [f"struct {name};" for name in sorted(objects)]
    for name, obj in unions.items():
        alternatives = obj.get("oneOf", obj.get("anyOf"))
        if not all("$ref" in part for part in alternatives):
            raise ValueError(f"Named union {name} requires model references")
        names = [part["$ref"].rsplit("/", 1)[-1] for part in alternatives]
        lines += [
            f"struct {name} {{",
            "  using Value = std::variant<" + ", ".join("std::shared_ptr<" + part + ">" for part in names) + ">;",
            "  Value value;",
            "};",
        ]
    for name in order:
        obj = objects[name]
        lines += [f"struct {name} {{"]
        for field, prop in obj.get("properties", {}).items():
            typ = cpp_type(prop, name, field)
            if field not in obj.get("required", []):
                typ = f"std::optional<{typ}>"
            lines += [f"  {typ} {_cpp_member_name(field)}{{}};"]
        lines += ["};"]
    for name, model in models.items():
        if name not in objects and name not in unions and name not in enums and name != "F8JsonValue":
            lines += [f"using {name} = {cpp_type(model.obj, name, 'value')};"]
    all_models = list(enums) + list(objects) + list(unions)
    for name in all_models:
        lines += [f"inline bool decode_value(const nlohmann::json&, {name}&, ParseError&);"]
    for name in list(objects) + list(unions):
        lines += [f"inline void to_json(nlohmann::json&, const {name}&);"]
    lines += [
        r"""
inline bool decode_value(const nlohmann::json& j, nlohmann::json& out, ParseError&) { out=j; return true; }
inline bool decode_value(const nlohmann::json& j, std::string& out, ParseError& e) {
  if (!j.is_string()) return invalid(e,"string expected"); out=j.get<std::string>(); return true;
}
inline bool decode_value(const nlohmann::json& j, std::int64_t& out, ParseError& e) {
  if (!j.is_number_integer() || (j.is_number_unsigned() && j.get<std::uint64_t>() > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))) return invalid(e,"int64 expected");
  out=j.get<std::int64_t>(); return true;
}
inline bool decode_value(const nlohmann::json& j, double& out, ParseError& e) {
  if (!j.is_number()) return invalid(e,"number expected"); out=j.get<double>(); return true;
}
inline bool decode_value(const nlohmann::json& j, bool& out, ParseError& e) {
  if (!j.is_boolean()) return invalid(e,"boolean expected"); out=j.get<bool>(); return true;
}
inline bool decode_value(const nlohmann::json& j, std::nullptr_t& out, ParseError& e) {
  if (!j.is_null()) return invalid(e,"null expected"); out=nullptr; return true;
}
template<class T> bool decode_value(const nlohmann::json&, std::vector<T>&, ParseError&);
template<class T> bool decode_value(const nlohmann::json&, std::map<std::string,T>&, ParseError&);
template<class T> bool decode_value(const nlohmann::json&, std::optional<T>&, ParseError&);
template<class T> bool decode_value(const nlohmann::json& j, std::optional<T>& out, ParseError& e) {
  if (j.is_null()) { out.reset(); return true; } T value{}; if (!decode_value(j,value,e)) return false; out=std::move(value); return true;
}
template<class T> bool decode_value(const nlohmann::json& j, std::vector<T>& out, ParseError& e) {
  if (!j.is_array()) return invalid(e,"array expected"); std::vector<T> value;
  for (const auto& item : j) { T v{}; if (!decode_value(item,v,e)) return false; value.push_back(std::move(v)); } out=std::move(value); return true;
}
template<class T> bool decode_value(const nlohmann::json& j, std::map<std::string,T>& out, ParseError& e) {
  if (!j.is_object()) return invalid(e,"object map expected"); std::map<std::string,T> value;
  for (const auto& item : j.items()) { T v{}; if (!decode_value(item.value(),v,e)) return false; value.emplace(item.key(),std::move(v)); } out=std::move(value); return true;
}
template<class T> nlohmann::json wire_json(const T& value) { return nlohmann::json(value); }
template<class T> nlohmann::json wire_json(const std::optional<T>&);
template<class T> nlohmann::json wire_json(const std::vector<T>&);
template<class T> nlohmann::json wire_json(const std::map<std::string,T>&);
template<class T> nlohmann::json wire_json(const std::optional<T>& value) { return value ? wire_json(*value) : nlohmann::json(nullptr); }
template<class T> nlohmann::json wire_json(const std::vector<T>& value) { auto j=nlohmann::json::array(); for (const auto& v:value) j.push_back(wire_json(v)); return j; }
template<class T> nlohmann::json wire_json(const std::map<std::string,T>& value) { auto j=nlohmann::json::object(); for (const auto& v:value) j[v.first]=wire_json(v.second); return j; }
"""
    ]
    for enum in enums.values():
        lines += [
            f"inline bool decode_value(const nlohmann::json& j, {enum.name}& out, ParseError& e) {{",
            '  if (!j.is_string()) return invalid(e,"enum string expected");',
            f'  auto v=parse_{enum.name}(j.get<std::string>()); if (!v) return invalid(e,"invalid enum value"); out=*v; return true;',
            "}",
        ]
    for name in order:
        obj = objects[name]
        lines += [
            f"inline bool decode_value(const nlohmann::json& j, {name}& out, ParseError& e) {{",
            '  if (!j.is_object()) return invalid(e,"object expected");',
            f"  {name} value{{}};",
        ]
        for field, prop in obj.get("properties", {}).items():
            key, member = json.dumps(field), _cpp_member_name(field)
            required = field in obj.get("required", [])
            lines += [f"  if (j.contains({key})) {{"]
            if "const" in prop:
                lines += [
                    f'    if (j[{key}] != {json.dumps(prop["const"])}) return invalid(e,"invalid constant: {field}");'
                ]
            if not required:
                lines += [
                    f"    {cpp_type(prop, name, field)} decoded{{}};",
                    f'    if (!decode_value(j[{key}],decoded,e)) {{ e.message={key}+std::string(": ")+e.message; return false; }}',
                    f"    value.{member}=std::move(decoded);",
                ]
            else:
                lines += [
                    f'    if (!decode_value(j[{key}],value.{member},e)) {{ e.message={key}+std::string(": ")+e.message; return false; }}'
                ]
            lines += ["  }"]
            if required:
                lines += [f'  else return invalid(e,"missing required field: {field}");']
        # Preserve API/1 unknown-field tolerance, matching msgspec models.
        if name == "F8ComponentRecord":
            allowed = " && ".join(f"item.key() != {json.dumps(field)}" for field in obj.get("properties", {})) or "true"
            lines += [
                f'  for (const auto& item : j.items()) if ({allowed}) return invalid(e,"unknown field: "+item.key());'
            ]
        lines += [
            "  out=std::move(value); return true;",
            "}",
            f"inline bool parse_{name}(const nlohmann::json& j, {name}& out, ParseError& e) {{ return decode_value(j,out,e); }}",
            f"inline void to_json(nlohmann::json& j, const {name}& value) {{",
            "  j=nlohmann::json::object();",
        ]
        for field in obj.get("properties", {}):
            member = _cpp_member_name(field)
            if field in obj.get("required", []):
                lines += [f"  j[{json.dumps(field)}]=wire_json(value.{member});"]
            else:
                lines += [f"  if (value.{member}) j[{json.dumps(field)}]=wire_json(*value.{member});"]
        lines += ["}"]
    for name, obj in unions.items():
        lines += [f"inline bool decode_value(const nlohmann::json& j, {name}& out, ParseError& e) {{"]
        discriminator = obj.get("discriminator")
        if discriminator:
            key = json.dumps(discriminator["propertyName"])
            lines += [
                f'  if (!j.is_object() || !j.contains({key}) || !j[{key}].is_string()) return invalid(e,"missing union discriminator");'
            ]
            for tag, ref in discriminator["mapping"].items():
                target = ref.rsplit("/", 1)[-1]
                lines += [
                    f"  if (j[{key}] == {json.dumps(tag)}) {{ auto v=std::make_shared<{target}>(); if (!decode_value(j,*v,e)) return false; out.value=std::move(v); return true; }}"
                ]
        else:
            for part in obj.get("oneOf", obj.get("anyOf")):
                target = part["$ref"].rsplit("/", 1)[-1]
                lines += [
                    f"  {{ auto v=std::make_shared<{target}>(); if (decode_value(j,*v,e)) {{ out.value=std::move(v); return true; }} }}"
                ]
        lines += [
            '  return invalid(e,"invalid union discriminator/shape");',
            "}",
            f"inline bool parse_{name}(const nlohmann::json& j, {name}& out, ParseError& e) {{ return decode_value(j,out,e); }}",
            f'inline void to_json(nlohmann::json& j, const {name}& value) {{ std::visit([&j](const auto& p) {{ if (!p) throw std::invalid_argument("empty union"); j=wire_json(*p); }}, value.value); }}',
        ]
    lines += [f"}}  // namespace {namespace}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="schemas/protocol.yml")
    ap.add_argument("--out", default="packages/f8cppsdk/include/f8cppsdk/generated/protocol_models.h")
    ap.add_argument("--namespace", default="f8::cppsdk::generated")
    args = ap.parse_args()

    protocol = Path(args.protocol)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    schemas = _load_schemas(protocol)
    header = _gen_header(schemas=schemas, namespace=args.namespace)
    out.write_text(header, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
