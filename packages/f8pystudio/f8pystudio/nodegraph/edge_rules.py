from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from f8pysdk import F8OperatorSpec, F8ServiceSpec

EDGE_KIND_EXEC = "exec"
EDGE_KIND_DATA = "data"
EDGE_KIND_STATE = "state"
EDGE_KINDS: tuple[str, str, str] = (EDGE_KIND_EXEC, EDGE_KIND_DATA, EDGE_KIND_STATE)


@dataclass(frozen=True)
class EdgeRuleNodeInfo:
    node_id: str
    service_id: str
    is_operator: bool


def normalize_edge_kind(kind: str) -> str | None:
    raw = str(kind or "").strip().lower()
    if raw in EDGE_KINDS:
        return raw
    return None


def port_kind(name: str) -> str | None:
    n = str(name or "").strip()
    if n.startswith("[E]") or n.endswith("[E]"):
        return EDGE_KIND_EXEC
    if n.startswith("[D]") or n.endswith("[D]"):
        return EDGE_KIND_DATA
    if n.startswith("[S]") or n.endswith("[S]"):
        return EDGE_KIND_STATE
    return None


def connection_kind(out_port_name: str, in_port_name: str) -> str | None:
    out_kind = port_kind(out_port_name)
    in_kind = port_kind(in_port_name)
    if out_kind is None or in_kind is None:
        return None
    if out_kind != in_kind:
        return None
    return out_kind


def port_view_name(port_view: Any) -> str:
    try:
        return str(port_view.name or "").strip()
    except (AttributeError, RuntimeError, TypeError):
        pass
    try:
        return str(port_view.name() or "").strip()
    except (AttributeError, RuntimeError, TypeError):
        return ""


def runtime_node_info(node: Any) -> EdgeRuleNodeInfo | None:
    if node is None:
        return None

    node_id = ""
    try:
        node_id = str(node.id or "").strip()
    except (AttributeError, RuntimeError, TypeError):
        node_id = ""

    try:
        spec = node.spec
    except (AttributeError, RuntimeError, TypeError):
        spec = None

    if isinstance(spec, F8OperatorSpec):
        service_id = ""
        try:
            service_id = str(node.svcId or "").strip()
        except (AttributeError, RuntimeError, TypeError):
            service_id = ""
        return EdgeRuleNodeInfo(node_id=node_id, service_id=service_id, is_operator=True)

    if isinstance(spec, F8ServiceSpec):
        return EdgeRuleNodeInfo(node_id=node_id, service_id=node_id, is_operator=False)

    return None


def layout_node_info(node_id: str, node_data: dict[str, Any]) -> EdgeRuleNodeInfo | None:
    node_payload = node_data if isinstance(node_data, dict) else {}
    spec_payload = node_payload.get("f8_spec")

    is_operator = False
    if isinstance(spec_payload, F8OperatorSpec):
        is_operator = True
    elif isinstance(spec_payload, F8ServiceSpec):
        is_operator = False
    elif isinstance(spec_payload, dict):
        is_operator = "operatorClass" in spec_payload
    else:
        return None

    node_id_str = str(node_id or "").strip()
    if is_operator:
        service_id = ""
        custom = node_payload.get("custom")
        if isinstance(custom, dict):
            service_id = str(custom.get("svcId") or "").strip()
        if not service_id:
            service_id = str(node_payload.get("svcId") or "").strip()
        return EdgeRuleNodeInfo(node_id=node_id_str, service_id=service_id, is_operator=True)

    return EdgeRuleNodeInfo(node_id=node_id_str, service_id=node_id_str, is_operator=False)


def validate_connection_by_infos(
    *,
    out_port_name: str,
    in_port_name: str,
    out_info: EdgeRuleNodeInfo | None,
    in_info: EdgeRuleNodeInfo | None,
) -> tuple[bool, str]:
    kind = connection_kind(out_port_name, in_port_name)
    if kind is None:
        return False, "port kinds must match and be one of exec/data/state"

    if kind == EDGE_KIND_EXEC:
        if out_info is None or in_info is None:
            return False, "exec edge endpoint metadata is missing"
        if not out_info.is_operator or not in_info.is_operator:
            return False, "exec edges require operator-to-operator endpoints"
        if not out_info.service_id or not in_info.service_id:
            return False, "exec edges require non-empty svcId for both endpoints"
        if out_info.service_id != in_info.service_id:
            return False, "cross-service exec edges are not allowed"
        return True, ""

    # data/state only need same-kind connection; cross-service is allowed.
    return True, ""


def validate_runtime_connection(
    *,
    out_port_name: str,
    in_port_name: str,
    out_node: Any,
    in_node: Any,
) -> tuple[bool, str]:
    return validate_connection_by_infos(
        out_port_name=out_port_name,
        in_port_name=in_port_name,
        out_info=runtime_node_info(out_node),
        in_info=runtime_node_info(in_node),
    )


def validate_layout_connection(
    *,
    out_node_id: str,
    out_port_name: str,
    in_node_id: str,
    in_port_name: str,
    node_info_by_id: dict[str, EdgeRuleNodeInfo | None],
) -> tuple[bool, str]:
    return validate_connection_by_infos(
        out_port_name=out_port_name,
        in_port_name=in_port_name,
        out_info=node_info_by_id.get(str(out_node_id)),
        in_info=node_info_by_id.get(str(in_node_id)),
    )

