from __future__ import annotations

import hashlib
import json
from typing import cast

import msgspec
from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import (
    OperatorNode,
    PatchResult,
    StudioDocument,
)

from f8studio_server.errors import InvalidRequestError

from ..catalog import CatalogSnapshot
from ..models import DeployJob
from .models import (
    AgentSessionRecord,
    ToolCallStatus,
)


def json_value(value: object) -> F8JsonValue:
    return cast(F8JsonValue, msgspec.to_builtins(value, str_keys=True))


def tool_text(value: object) -> str:
    return json.dumps(json_value(value), ensure_ascii=False)


def arguments_hash(arguments: dict[str, F8JsonValue]) -> str:
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def catalog_evidence(catalog: CatalogSnapshot) -> F8JsonValue:
    return {"serviceCount": len(catalog.services), "operatorCount": len(catalog.operators)}


def catalog_index(catalog: CatalogSnapshot) -> F8JsonValue:
    return {
        "services": [{"serviceClass": str(spec.serviceClass), "label": str(spec.label)} for spec in catalog.services],
        "operators": [{"serviceClass": str(spec.serviceClass), "operatorClass": str(spec.operatorClass),
                       "label": str(spec.label)} for spec in catalog.operators],
    }


def catalog_search(catalog: CatalogSnapshot, query: str) -> F8JsonValue:
    needle = query.strip().casefold()
    if len(needle) < 2:
        raise InvalidRequestError("Catalog search needs at least two characters")
    terms = needle.split()
    matches = [spec for spec in catalog.operators if any(term in " ".join((
        str(spec.serviceClass), str(spec.operatorClass), str(spec.label), str(spec.description),
    )).casefold() for term in terms)]
    return {
        "query": query,
        "matches": [{"serviceClass": str(spec.serviceClass), "operatorClass": str(spec.operatorClass),
                     "label": str(spec.label), "description": str(spec.description)[:240]}
                    for spec in matches[:25]],
        "total": len(matches),
    }


def catalog_operator(catalog: CatalogSnapshot, service_class: str, operator_class: str) -> F8JsonValue:
    spec = next((item for item in catalog.operators
                 if str(item.serviceClass) == service_class and str(item.operatorClass) == operator_class), None)
    if spec is None:
        raise InvalidRequestError(f"Unknown operator: {service_class}/{operator_class}")
    return json_value(spec)


def graph_outline(document: StudioDocument) -> F8JsonValue:
    return {
        "projectId": document.project_id,
        "graphRevision": document.graph_revision,
        "layoutRevision": document.layout_revision,
        "nodes": [{"nodeId": node.node_id, "name": node.name, "kind": "operator" if isinstance(node, OperatorNode) else "service",
                   "serviceId": node.service_id, "serviceClass": node.service_class,
                   "operatorClass": node.operator_class if isinstance(node, OperatorNode) else None,
                   "stateValues": node.state_values,
                   "ports": [{"portId": port.port_id, "name": port.name, "kind": port.kind.value,
                              "direction": port.direction.value} for port in node.ports]}
                  for node in document.nodes],
        "edges": json_value(document.edges),
        "layout": json_value(document.layout),
    }


def document_evidence(document: StudioDocument) -> F8JsonValue:
    return {
        "projectId": document.project_id,
        "graphRevision": document.graph_revision,
        "layoutRevision": document.layout_revision,
        "nodeCount": len(document.nodes),
        "edgeCount": len(document.edges),
    }


def patch_evidence(result: PatchResult) -> F8JsonValue:
    return {
        "requestId": result.request_id,
        "graphChanged": result.graph_changed,
        "layoutChanged": result.layout_changed,
        "graphRevision": result.document.graph_revision,
        "layoutRevision": result.document.layout_revision,
        "runtimeErrors": list(result.runtime_errors),
    }


def deploy_evidence(job: DeployJob) -> F8JsonValue:
    return {
        "jobId": job.job_id,
        "status": job.status.value,
        "sourceGraphRevision": job.source_graph_revision,
        "serviceCount": len(job.service_results),
    }


def monitor_evidence(snapshot: F8JsonValue) -> F8JsonValue:
    return {"sampleCount": len(snapshot) if isinstance(snapshot, list) else 0}


def conversation_prompt(record: AgentSessionRecord, prompt: str) -> str:
    earlier = record.messages[:-1][-8:]
    if not earlier:
        return prompt
    history = [{"role": message.role, "content": message.content[:4000],
                "images": [image.name for image in message.images]} for message in earlier]
    return f"Previous conversation:\n{json.dumps(history, ensure_ascii=False)}\nCurrent request:\n{prompt}"


def evidence_prompt(record: AgentSessionRecord) -> str:
    succeeded_tools = [call.tool_name for call in record.tool_calls if call.status is ToolCallStatus.succeeded]
    revisions = [
        call.target_graph_revision
        for call in record.tool_calls
        if call.target_graph_revision is not None and call.status is ToolCallStatus.succeeded
    ]
    return (
        f"Completed tools: {', '.join(succeeded_tools)}. "
        f"Observed graph revisions: {revisions}. "
        f"Artifacts: {len(record.artifacts)}. "
        "The run used authoritative Studio application services and retained tool evidence."
    )
