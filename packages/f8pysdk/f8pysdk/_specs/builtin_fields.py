from __future__ import annotations

from typing import Any

from ..generated import (
    F8DataPortSpec,
    F8OperatorSpec,
    F8ServiceSpec,
    F8StateAccess,
    F8StateFieldEditPolicy,
    F8StateSpec,
)
from ..monitoring import (
    MONITOR_PORT_NAME,
    monitor_snapshot_data_port,
    monitor_snapshot_schema_dict_cached,
)
from .schema import boolean_schema, string_schema


ACTIVE_FIELD_NAME = "active"
SVC_ID_FIELD_NAME = "svcId"
OPERATOR_ID_FIELD_NAME = "operatorId"


def _locked_builtin_state_policy() -> F8StateFieldEditPolicy:
    return F8StateFieldEditPolicy(
        canRename=False,
        canEditAccess=False,
        canEditValueRequired=False,
        canEditValueSchema=False,
    )


def _locked_builtin_state_policy_dict() -> dict[str, bool]:
    return {
        "canRename": False,
        "canEditAccess": False,
        "canEditValueRequired": False,
        "canEditValueSchema": False,
    }


def _service_active_state_spec() -> F8StateSpec:
    return F8StateSpec(
        name=ACTIVE_FIELD_NAME,
        label="Active",
        description="Service lifecycle state (activate/deactivate).",
        valueSchema=boolean_schema(default=True),
        access=F8StateAccess.rw,
        valueRequired=True,
        editPolicy=_locked_builtin_state_policy(),
        showOnNode=False,
    )


def _svc_id_state_spec() -> F8StateSpec:
    return F8StateSpec(
        name=SVC_ID_FIELD_NAME,
        label="Service Id",
        description="Readonly: current service instance id (svcId).",
        valueSchema=string_schema(),
        access=F8StateAccess.ro,
        valueRequired=True,
        editPolicy=_locked_builtin_state_policy(),
        showOnNode=False,
    )


def _operator_id_state_spec() -> F8StateSpec:
    return F8StateSpec(
        name=OPERATOR_ID_FIELD_NAME,
        label="Operator Id",
        description="Readonly: current operator/node id (operatorId).",
        valueSchema=string_schema(),
        access=F8StateAccess.ro,
        valueRequired=True,
        editPolicy=_locked_builtin_state_policy(),
        showOnNode=False,
    )


def _copy_state_specs_without_names(
    state_fields: list[F8StateSpec] | None, *, names_to_remove: set[str]
) -> list[F8StateSpec]:
    filtered: list[F8StateSpec] = []
    for field in list(state_fields or []):
        field_name = str(field.name or "").strip()
        if field_name in names_to_remove:
            continue
        filtered.append(field)
    return filtered


def _copy_data_port_specs_without_names(
    ports: list[F8DataPortSpec] | None, *, names_to_remove: set[str]
) -> list[F8DataPortSpec]:
    filtered: list[F8DataPortSpec] = []
    for port in list(ports or []):
        port_name = str(port.name or "").strip()
        if port_name in names_to_remove:
            continue
        filtered.append(port)
    return filtered


def service_state_fields_with_builtins(state_fields: list[F8StateSpec] | None) -> list[F8StateSpec]:
    fields = _copy_state_specs_without_names(
        state_fields,
        names_to_remove={ACTIVE_FIELD_NAME, SVC_ID_FIELD_NAME},
    )
    fields.append(_service_active_state_spec())
    fields.append(_svc_id_state_spec())
    return fields


def service_data_out_ports_with_builtins(data_out_ports: list[F8DataPortSpec] | None) -> list[F8DataPortSpec]:
    ports = _copy_data_port_specs_without_names(
        data_out_ports,
        names_to_remove={MONITOR_PORT_NAME},
    )
    ports.append(monitor_snapshot_data_port())
    return ports


def operator_state_fields_with_builtins(state_fields: list[F8StateSpec] | None) -> list[F8StateSpec]:
    fields = _copy_state_specs_without_names(
        state_fields,
        names_to_remove={SVC_ID_FIELD_NAME, OPERATOR_ID_FIELD_NAME},
    )
    fields.append(_svc_id_state_spec())
    fields.append(_operator_id_state_spec())
    return fields


def upsert_builtin_state_fields_for_service_spec(service_spec: F8ServiceSpec) -> None:
    service_spec.stateFields = service_state_fields_with_builtins(list(service_spec.stateFields or []))
    service_spec.dataOutPorts = service_data_out_ports_with_builtins(list(service_spec.dataOutPorts or []))


def upsert_builtin_state_fields_for_operator_spec(operator_spec: F8OperatorSpec) -> None:
    operator_spec.stateFields = operator_state_fields_with_builtins(list(operator_spec.stateFields or []))


def _service_active_field_dict() -> dict[str, Any]:
    return {
        "name": ACTIVE_FIELD_NAME,
        "label": "Active",
        "description": "Service lifecycle state (activate/deactivate).",
        "valueSchema": {"type": "boolean", "default": True},
        "access": "rw",
        "valueRequired": True,
        "editPolicy": _locked_builtin_state_policy_dict(),
        "showOnNode": False,
    }


def _svc_id_field_dict() -> dict[str, Any]:
    return {
        "name": SVC_ID_FIELD_NAME,
        "label": "Service Id",
        "description": "Readonly: current service instance id (svcId).",
        "valueSchema": {"type": "string"},
        "access": "ro",
        "valueRequired": True,
        "editPolicy": _locked_builtin_state_policy_dict(),
        "showOnNode": False,
    }


def _operator_id_field_dict() -> dict[str, Any]:
    return {
        "name": OPERATOR_ID_FIELD_NAME,
        "label": "Operator Id",
        "description": "Readonly: current operator/node id (operatorId).",
        "valueSchema": {"type": "string"},
        "access": "ro",
        "valueRequired": True,
        "editPolicy": _locked_builtin_state_policy_dict(),
        "showOnNode": False,
    }


def _monitor_port_dict() -> dict[str, Any]:
    return {
        "name": MONITOR_PORT_NAME,
        "description": "Unified runtime monitor snapshots (health/resource/perf/error).",
        "definitionProtected": True,
        "showOnNode": False,
        "payload": {"kind": "json", "valueSchema": monitor_snapshot_schema_dict_cached()},
    }


def _descriptor_dicts(value: Any, *, path: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be a list of objects")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{path}[{index}] must be an object")
        result.append(dict(item))
    return result


def _with_builtin_descriptors(
    value: Any, *, path: str, builtins: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    names = {item["name"] for item in builtins}
    descriptors = _descriptor_dicts(value, path=path)
    return [item for item in descriptors if item.get("name") not in names] + builtins


def normalize_describe_payload_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Inject runtime-owned descriptors into a current describe envelope."""
    out = dict(payload)
    service_obj = out.get("service")
    if not isinstance(service_obj, dict):
        raise ValueError("describe.service must be an object")
    service_spec = dict(service_obj)
    service_spec["stateFields"] = _with_builtin_descriptors(
        service_spec.get("stateFields", []), path="service.stateFields",
        builtins=[_service_active_field_dict(), _svc_id_field_dict()],
    )
    service_spec["dataOutPorts"] = _with_builtin_descriptors(
        service_spec.get("dataOutPorts", []), path="service.dataOutPorts",
        builtins=[_monitor_port_dict()],
    )
    out["service"] = service_spec
    operators = _descriptor_dicts(out.get("operators", []), path="operators")
    for index, operator in enumerate(operators):
        operator["stateFields"] = _with_builtin_descriptors(
            operator.get("stateFields", []), path=f"operators[{index}].stateFields",
            builtins=[_svc_id_field_dict(), _operator_id_field_dict()],
        )
    out["operators"] = operators
    return out


__all__ = [
    "ACTIVE_FIELD_NAME",
    "MONITOR_PORT_NAME",
    "OPERATOR_ID_FIELD_NAME",
    "SVC_ID_FIELD_NAME",
    "normalize_describe_payload_dict",
    "operator_state_fields_with_builtins",
    "service_data_out_ports_with_builtins",
    "service_state_fields_with_builtins",
    "upsert_builtin_state_fields_for_operator_spec",
    "upsert_builtin_state_fields_for_service_spec",
]
