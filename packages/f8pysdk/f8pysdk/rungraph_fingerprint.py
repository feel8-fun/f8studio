from __future__ import annotations

import json
from enum import Enum
from typing import Any

import msgspec

from .codec import dump_json
from .generated import F8DataPortSpec, F8Edge, F8RuntimeGraph, F8RuntimeNode, F8RuntimeService, F8StateSpec


def build_rungraph_deploy_fingerprint(graph: F8RuntimeGraph | Any) -> str:
    snapshot = build_rungraph_deploy_snapshot(graph)
    return json.dumps(snapshot, sort_keys=True, separators=(",", ":"))


def build_rungraph_deploy_snapshot(graph: F8RuntimeGraph | Any) -> dict[str, Any]:
    if isinstance(graph, F8RuntimeGraph):
        return _build_runtime_graph_deploy_snapshot(graph)

    graph_payload = dump_json(graph, mode="json", by_alias=True)
    if not isinstance(graph_payload, dict):
        return {"services": [], "nodes": [], "edges": []}

    raw_services = graph_payload.get("services")
    raw_nodes = graph_payload.get("nodes")
    raw_edges = graph_payload.get("edges")

    services = []
    if isinstance(raw_services, list):
        services = sorted(
            (_normalize_service_payload(item) for item in raw_services),
            key=_normalized_service_sort_key,
        )

    nodes = []
    if isinstance(raw_nodes, list):
        nodes = sorted(
            (_normalize_node_payload(item) for item in raw_nodes),
            key=_normalized_node_sort_key,
        )

    edges = []
    if isinstance(raw_edges, list):
        edges = sorted(
            (_normalize_edge_payload(item) for item in raw_edges),
            key=_normalized_edge_sort_key,
        )

    return {
        "services": services,
        "nodes": nodes,
        "edges": edges,
    }


def _build_runtime_graph_deploy_snapshot(graph: F8RuntimeGraph) -> dict[str, Any]:
    cache: dict[int, Any] = {}
    services = []
    if not _is_unset(graph.services):
        services = sorted(
            (_normalize_runtime_service(item, cache) for item in graph.services),
            key=_normalized_service_sort_key,
        )
    nodes = []
    if not _is_unset(graph.nodes):
        nodes = sorted(
            (_normalize_runtime_node(item, cache) for item in graph.nodes),
            key=_normalized_node_sort_key,
        )
    edges = []
    if not _is_unset(graph.edges):
        edges = sorted((_normalize_runtime_edge(item) for item in graph.edges), key=_normalized_edge_sort_key)
    return {"services": services, "nodes": nodes, "edges": edges}


def _normalize_spec_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        normalized: dict[str, Any] = {}
        for key in sorted(str(k) for k in payload.keys()):
            normalized[key] = _normalize_spec_payload(payload[key])
        return normalized
    if isinstance(payload, list):
        return [_normalize_spec_payload(item) for item in payload]
    return payload


def _is_unset(value: Any) -> bool:
    return value is msgspec.UNSET or isinstance(value, msgspec.UnsetType)


def _enum_payload(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _put_optional_payload(payload: dict[str, Any], key: str, value: Any) -> None:
    if _is_unset(value):
        return
    payload[key] = _enum_payload(value)


def _normalize_cached_payload(value: Any, cache: dict[int, Any]) -> Any:
    cache_key = id(value)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    normalized = _normalize_spec_payload(dump_json(value, mode="json"))
    cache[cache_key] = normalized
    return normalized


def _normalize_data_port_spec(spec: Any, cache: dict[int, Any]) -> dict[str, Any]:
    cache_key = id(spec)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached if isinstance(cached, dict) else {}
    if not isinstance(spec, F8DataPortSpec):
        normalized = _normalize_cached_payload(spec, cache)
        return normalized if isinstance(normalized, dict) else {}

    payload: dict[str, Any] = {
        "name": str(spec.name),
        "valueSchema": _normalize_cached_payload(spec.valueSchema, cache),
    }
    if not _is_unset(spec.payload):
        payload["payload"] = _normalize_cached_payload(spec.payload, cache)
    if not _is_unset(spec.stream):
        payload["stream"] = _normalize_cached_payload(spec.stream, cache)
    _put_optional_payload(payload, "description", spec.description)
    _put_optional_payload(payload, "showOnNode", spec.showOnNode)
    _put_optional_payload(payload, "payloadKind", spec.payloadKind)
    _put_optional_payload(payload, "delivery", spec.delivery)
    cache[cache_key] = payload
    return payload


def _normalize_state_spec(spec: Any, cache: dict[int, Any]) -> dict[str, Any]:
    cache_key = id(spec)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached if isinstance(cached, dict) else {}
    if not isinstance(spec, F8StateSpec):
        normalized = _normalize_cached_payload(spec, cache)
        return normalized if isinstance(normalized, dict) else {}

    payload: dict[str, Any] = {
        "name": str(spec.name),
        "valueSchema": _normalize_cached_payload(spec.valueSchema, cache),
        "access": _enum_payload(spec.access),
    }
    _put_optional_payload(payload, "label", spec.label)
    _put_optional_payload(payload, "description", spec.description)
    _put_optional_payload(payload, "valueRequired", spec.valueRequired)
    if not _is_unset(spec.editPolicy):
        payload["editPolicy"] = _normalize_cached_payload(spec.editPolicy, cache)
    _put_optional_payload(payload, "showOnNode", spec.showOnNode)
    _put_optional_payload(payload, "redactOnPublish", spec.redactOnPublish)
    editor_assist = msgspec.UNSET
    if not _is_unset(spec.editorAssist):
        editor_assist = _normalize_cached_payload(spec.editorAssist, cache)
    _put_optional_payload(payload, "editorAssist", editor_assist)
    cache[cache_key] = payload
    return payload


def _normalize_data_port_specs(specs: Any, cache: dict[int, Any]) -> list[dict[str, Any]]:
    cache_key = id(specs)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    normalized = sorted(
        (_normalize_data_port_spec(item, cache) for item in specs),
        key=_normalized_named_spec_sort_key,
    )
    cache[cache_key] = normalized
    return normalized


def _normalize_state_specs(specs: Any, cache: dict[int, Any]) -> list[dict[str, Any]]:
    cache_key = id(specs)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    normalized = sorted(
        (_normalize_state_spec(item, cache) for item in specs),
        key=_normalized_named_spec_sort_key,
    )
    cache[cache_key] = normalized
    return normalized


def _normalize_runtime_service(service: F8RuntimeService, cache: dict[int, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "serviceId": str(service.serviceId),
        "serviceClass": str(service.serviceClass),
    }
    _put_optional_payload(payload, "label", service.label)
    if not _is_unset(service.meta):
        payload["meta"] = _normalize_cached_payload(service.meta, cache)
    if not _is_unset(service.autoSampleRequests):
        payload["autoSampleRequests"] = _normalize_cached_payload(service.autoSampleRequests, cache)
    return payload


def _normalize_runtime_node(node: F8RuntimeNode, cache: dict[int, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "nodeId": str(node.nodeId),
        "serviceId": str(node.serviceId),
        "serviceClass": str(node.serviceClass),
    }
    if not _is_unset(node.operatorClass):
        payload["operatorClass"] = node.operatorClass
    if not _is_unset(node.execInPorts):
        payload["execInPorts"] = sorted(str(item) for item in node.execInPorts)
    if not _is_unset(node.execOutPorts):
        payload["execOutPorts"] = sorted(str(item) for item in node.execOutPorts)
    if not _is_unset(node.dataInPorts):
        payload["dataInPorts"] = _normalize_data_port_specs(node.dataInPorts, cache)
    if not _is_unset(node.dataOutPorts):
        payload["dataOutPorts"] = _normalize_data_port_specs(node.dataOutPorts, cache)
    if not _is_unset(node.stateFields):
        payload["stateFields"] = _normalize_state_specs(node.stateFields, cache)
    return payload


def _normalize_runtime_edge(edge: F8Edge) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "fromServiceId": str(edge.fromServiceId),
        "fromPort": str(edge.fromPort),
        "toServiceId": str(edge.toServiceId),
        "toPort": str(edge.toPort),
        "kind": _enum_payload(edge.kind),
    }
    if not _is_unset(edge.fromOperatorId):
        payload["fromOperatorId"] = edge.fromOperatorId
    if not _is_unset(edge.toOperatorId):
        payload["toOperatorId"] = edge.toOperatorId
    if not _is_unset(edge.strategy):
        payload["strategy"] = _enum_payload(edge.strategy)
    if not _is_unset(edge.queueSize):
        payload["queueSize"] = edge.queueSize
    if not _is_unset(edge.timeoutMs):
        payload["timeoutMs"] = edge.timeoutMs
    if not _is_unset(edge.direction):
        payload["direction"] = _enum_payload(edge.direction)
    return payload


def _normalized_named_spec_sort_key(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return ("", json.dumps(payload, sort_keys=True, separators=(",", ":")))
    name = str(payload.get("name") or "")
    payload_kind = str(payload.get("type") or payload.get("access") or "")
    return (name, payload_kind)


def _normalized_service_sort_key(payload: Any) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return ("", "")
    return (str(payload.get("serviceId") or ""), str(payload.get("serviceClass") or ""))


def _normalized_node_sort_key(payload: Any) -> tuple[str, str, str]:
    if not isinstance(payload, dict):
        return ("", "", "")
    return (
        str(payload.get("serviceId") or ""),
        str(payload.get("nodeId") or ""),
        str(payload.get("operatorClass") or ""),
    )


def _normalized_edge_sort_key(payload: Any) -> tuple[str, str, str, str, str, str]:
    if not isinstance(payload, dict):
        return ("", "", "", "", "", "")
    return (
        str(payload.get("kind") or ""),
        str(payload.get("fromServiceId") or ""),
        str(payload.get("fromOperatorId") or ""),
        str(payload.get("fromPort") or ""),
        str(payload.get("toServiceId") or ""),
        str(payload.get("toPort") or ""),
    )


def _normalize_service_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in sorted(str(k) for k in payload.keys()):
        normalized[key] = _normalize_spec_payload(payload[key])
    return normalized


def _normalize_node_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, Any] = {}
    for key in sorted(str(k) for k in payload.keys()):
        if key == "stateValues":
            continue
        value = payload[key]
        if key in {"execInPorts", "execOutPorts"} and isinstance(value, list):
            normalized[key] = sorted(str(item) for item in value)
            continue
        if key in {"dataInPorts", "dataOutPorts", "stateFields"} and isinstance(value, list):
            omitted = {"definitionProtected"} if key != "stateFields" else {"control"}
            normalized[key] = sorted(
                (_normalize_spec_payload({name: field for name, field in item.items() if name not in omitted})
                 if isinstance(item, dict) else _normalize_spec_payload(item) for item in value),
                key=_normalized_named_spec_sort_key,
            )
            continue
        normalized[key] = _normalize_spec_payload(value)
    return normalized


def _normalize_edge_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, Any] = {}
    for key in sorted(str(k) for k in payload.keys()):
        if key == "edgeId":
            continue
        normalized[key] = _normalize_spec_payload(payload[key])
    return normalized


__all__ = [
    "build_rungraph_deploy_fingerprint",
    "build_rungraph_deploy_snapshot",
]
