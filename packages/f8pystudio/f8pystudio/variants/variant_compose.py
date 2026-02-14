from __future__ import annotations

from copy import deepcopy
from typing import Any

import shortuuid

from f8pysdk import F8OperatorSpec, F8ServiceSpec

from .variant_models import F8NodeVariantRecord, F8VariantKind


def _state_fields_by_name(spec_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fields = spec_json.get("stateFields")
    if not isinstance(fields, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for entry in fields:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        out[name] = entry
    return out


def _apply_state_ui_overrides(spec_json: dict[str, Any], ui: dict[str, Any]) -> None:
    state_over = ui.get("stateFields")
    if not isinstance(state_over, dict):
        return
    fields = _state_fields_by_name(spec_json)
    allowed_keys = {"showOnNode", "uiControl", "uiLanguage", "label", "description"}
    for name, patch in state_over.items():
        if not isinstance(patch, dict):
            continue
        field = fields.get(str(name))
        if field is None:
            continue
        for k in allowed_keys:
            if k in patch:
                field[k] = patch[k]


def _apply_data_port_ui_overrides(spec_json: dict[str, Any], ui: dict[str, Any]) -> None:
    data_ports = ui.get("dataPorts")
    if not isinstance(data_ports, dict):
        return
    for key, spec_key in (("in", "dataInPorts"), ("out", "dataOutPorts")):
        patch_map = data_ports.get(key)
        ports = spec_json.get(spec_key)
        if not isinstance(patch_map, dict) or not isinstance(ports, list):
            continue
        by_name: dict[str, dict[str, Any]] = {}
        for p in ports:
            if not isinstance(p, dict):
                continue
            name = str(p.get("name") or "").strip()
            if name:
                by_name[name] = p
        for name, patch in patch_map.items():
            if not isinstance(patch, dict):
                continue
            port = by_name.get(str(name))
            if port is None:
                continue
            if "showOnNode" in patch:
                port["showOnNode"] = bool(patch.get("showOnNode"))


def _apply_command_ui_overrides(spec_json: dict[str, Any], ui: dict[str, Any]) -> None:
    commands_over = ui.get("commands")
    commands = spec_json.get("commands")
    if not isinstance(commands_over, dict) or not isinstance(commands, list):
        return
    by_name: dict[str, dict[str, Any]] = {}
    for c in commands:
        if not isinstance(c, dict):
            continue
        name = str(c.get("name") or "").strip()
        if name:
            by_name[name] = c
    for name, patch in commands_over.items():
        if not isinstance(patch, dict):
            continue
        cmd = by_name.get(str(name))
        if cmd is None:
            continue
        if "showOnNode" in patch:
            cmd["showOnNode"] = bool(patch.get("showOnNode"))


def _apply_state_defaults_from_values(spec_json: dict[str, Any], custom_properties: dict[str, Any]) -> None:
    if not isinstance(custom_properties, dict) or not custom_properties:
        return
    fields = _state_fields_by_name(spec_json)
    for name, field in fields.items():
        if name not in custom_properties:
            continue
        value_schema = field.get("valueSchema")
        if not isinstance(value_schema, dict):
            value_schema = {}
            field["valueSchema"] = value_schema
        value_schema["default"] = custom_properties.get(name)


def _locked_identity(base_spec_json: dict[str, Any], out_spec_json: dict[str, Any]) -> None:
    out_spec_json["serviceClass"] = base_spec_json.get("serviceClass")
    out_spec_json["schemaVersion"] = base_spec_json.get("schemaVersion")
    if "operatorClass" in base_spec_json:
        out_spec_json["operatorClass"] = base_spec_json.get("operatorClass")


def compose_variant_spec(
    *,
    spec_obj: F8OperatorSpec | F8ServiceSpec,
    ui_overrides: dict[str, Any],
    state_values: dict[str, Any],
    label: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    base = spec_obj.model_dump(mode="json")
    out = deepcopy(base)
    _apply_state_ui_overrides(out, ui_overrides)
    _apply_data_port_ui_overrides(out, ui_overrides)
    _apply_command_ui_overrides(out, ui_overrides)
    _apply_state_defaults_from_values(out, state_values)
    if label is not None:
        out["label"] = str(label or "")
    if description is not None:
        out["description"] = str(description or "")
    if tags is not None:
        out["tags"] = [str(t) for t in list(tags or []) if str(t).strip()]
    _locked_identity(base, out)
    if isinstance(spec_obj, F8OperatorSpec):
        return F8OperatorSpec.model_validate(out).model_dump(mode="json")
    return F8ServiceSpec.model_validate(out).model_dump(mode="json")


def build_variant_record_from_node(
    *,
    node: Any,
    name: str,
    description: str,
    tags: list[str],
    variant_id: str | None = None,
) -> F8NodeVariantRecord:
    spec_obj = node.spec
    if not isinstance(spec_obj, (F8OperatorSpec, F8ServiceSpec)):
        raise TypeError("Node spec must be F8OperatorSpec or F8ServiceSpec")

    ui_overrides = node.ui_overrides()
    if not isinstance(ui_overrides, dict):
        ui_overrides = {}
    model = node.model
    custom_properties = model.custom_properties
    state_values = dict(custom_properties) if isinstance(custom_properties, dict) else {}

    spec_json = compose_variant_spec(
        spec_obj=spec_obj,
        ui_overrides=ui_overrides,
        state_values=state_values,
        label=name,
        description=description,
        tags=tags,
    )

    is_operator = isinstance(spec_obj, F8OperatorSpec)
    now = F8NodeVariantRecord.now_iso()
    return F8NodeVariantRecord(
        variantId=str(variant_id or shortuuid.ShortUUID().random(12)),
        kind=F8VariantKind.operator if is_operator else F8VariantKind.service,
        baseNodeType=str(node.type_ or ""),
        serviceClass=str(spec_obj.serviceClass),
        operatorClass=str(spec_obj.operatorClass) if is_operator else None,
        name=str(name or "").strip(),
        description=str(description or "").strip(),
        tags=[str(t) for t in list(tags or []) if str(t).strip()],
        spec=spec_json,
        createdAt=now,
        updatedAt=now,
    )
