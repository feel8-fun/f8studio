"""Explicit HTTP wire contracts. msgspec models also drive generated TS types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import msgspec
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from f8pysdk.specs import F8JsonValue, F8MonitorSnapshot
from f8pysdk.decision import DecisionResult
from f8studio_core import HealthStatus, ServerCapabilities
from f8studio_core.graph import GraphNode, PatchResult
from f8studio_core.graph.exchange import GraphExchange
from f8media_protocol import models as media
from . import assets, editor, local_integration as local, models
from .agents import models as agents, provider_settings as settings
from .agents.provider_probe import ProviderProbeResult
from .catalog import CatalogSnapshot
from .events import EventEnvelope
from .extension_models import (
    EnvironmentStatus, ExtensionImportRequest, ExtensionInstallPlan, ExtensionStatus, ExtensionToggleRequest,
    PresetEnvironmentStatus,
)
from .processes import ManagedProcessResult
from .schema_generation import model_schemas
from .local_integration import ApplyUnityInstallRequest
from .assets import AssetExport
from f8media_protocol.models import AudioSessionOffer
from .agents import CreateAgentSessionRequest
from .assets import CreateAssetRequest
from .extension_tools import ToolView, ToolRunRequest, ToolJob, CapabilityResource, ResourceContent
from .models import CreateCatalogNodeRequest
from .editor import CreateEditorSessionRequest
from .models import CreateProjectRequest
from .assets import CreateProjectVersionRequest
from .agents.provider_settings import CreateProviderConnection
from f8pysdk.decision import DecisionRequest
from .models import DeployProjectRequest
from .local_integration import DetectModdingTargetRequest
from .editor import EditorPositionRequest
from f8studio_core.graph import HistoryRequest
from f8media_protocol.models import MediaSessionOffer
from f8media_protocol.models import OverlayResult
from f8studio_core.graph import PatchRequest
from .local_integration import PreviewUnityInstallRequest
from .agents.provider_probe import ProbeProviderRequest
from .local_integration import RegisterHotkeyRequest
from .agents import RenameAgentSessionRequest
from .agents import ResolveAgentApprovalRequest
from .models import RuntimeStateReadRequest
from .agents import SelectAgentModelRequest
from .models import ServiceActiveRequest
from .models import ServiceCommandRequest
from .models import ServiceStartRequest
from .models import ServiceStateRequest
from .agents import StartAgentRunRequest
from .assets import UpdateAssetRequest
from .editor import UpdateEditorDocumentRequest
from .models import UpdateProjectRequest
from .agents.provider_settings import UpdateProviderSettings
from .models import ValidateDocumentRequest
from .local_integration import VerifySkeletonUdpRequest


class CapabilitiesResponse(msgspec.Struct, frozen=True):
    protocol_version: Literal["f8studio-api/1"]
    capabilities: ServerCapabilities


class AcceptedResponse(msgspec.Struct, frozen=True):
    accepted: bool


class ValidationResponse(msgspec.Struct, frozen=True, rename="camel"):
    valid: bool
    graph_revision: int
    layout_revision: int


@dataclass(frozen=True)
class RouteContract:
    method: str
    path: str
    request: Any
    response: Any
    status: int = 200
    response_media_type: str = "application/json"


ROUTES = (
    RouteContract('get', '/api/extension-tools', None, tuple[ToolView, ...]),
    RouteContract('post', '/api/extension-tools/{extension_id}/{tool_id}/run', ToolRunRequest, ToolJob, 202),
    RouteContract('get', '/api/tool-jobs', None, tuple[ToolJob, ...]),
    RouteContract('get', '/api/tool-jobs/{job_id}', None, ToolJob),
    RouteContract('post', '/api/tool-jobs/{job_id}/cancel', None, ToolJob),
    RouteContract('get', '/api/extension-resources/{extension_id}/{resource_id}/file', None, None, 200, 'application/octet-stream'),
    RouteContract('get', '/api/extension-resources', None, tuple[CapabilityResource, ...]),
    RouteContract('get', '/api/extension-resources/{extension_id}/{resource_id}', None, ResourceContent),
    RouteContract("get", "/api/health", None, HealthStatus, 200),
    RouteContract("get", "/api/logs", None, tuple[EventEnvelope, ...], 200),
    RouteContract("get", "/api/capabilities", None, CapabilitiesResponse, 200),
    RouteContract("get", "/api/media/rtc-configuration", None, models.BrowserRtcConfiguration, 200),
    RouteContract("get", "/api/catalog", None, CatalogSnapshot, 200),
    RouteContract("post", "/api/catalog/refresh", None, CatalogSnapshot, 200),
    RouteContract("get", "/api/extensions", None, tuple[ExtensionStatus, ...], 200),
    RouteContract("post", "/api/extensions/import", ExtensionImportRequest, tuple[ExtensionStatus, ...], 200),
    RouteContract("get", "/api/extensions/{extension_id}/plan", None, ExtensionInstallPlan, 200),
    RouteContract("get", "/api/environments", None, tuple[EnvironmentStatus, ...], 200),
    RouteContract("get", "/api/environments/presets", None, tuple[PresetEnvironmentStatus, ...], 200),
    RouteContract("post", "/api/extensions/{extension_id}/install", None, ExtensionStatus, 200),
    RouteContract("post", "/api/extensions/{extension_id}/cancel", None, ExtensionStatus, 200),
    RouteContract("put", "/api/extensions/{extension_id}/enabled", ExtensionToggleRequest, ExtensionStatus, 200),
    RouteContract("delete", "/api/extensions/{extension_id}", None, ExtensionStatus, 200),
    RouteContract("post", "/api/catalog/nodes", CreateCatalogNodeRequest, GraphNode, 200),
    RouteContract("get", "/api/assets", None, tuple[assets.AssetSummary, ...], 200),
    RouteContract("post", "/api/assets", CreateAssetRequest, assets.AssetRecord, 201),
    RouteContract("post", "/api/assets/import", AssetExport, assets.AssetRecord, 201),
    RouteContract("get", "/api/assets/{asset_id}", None, assets.AssetRecord, 200),
    RouteContract("put", "/api/assets/{asset_id}", UpdateAssetRequest, assets.AssetRecord, 200),
    RouteContract("delete", "/api/assets/{asset_id}", None, None, 204),
    RouteContract("get", "/api/assets/{asset_id}/versions", None, tuple[assets.AssetVersion, ...], 200),
    RouteContract("get", "/api/assets/{asset_id}/export", None, assets.AssetExport, 200),
    RouteContract("get", "/api/runtime/monitors", None, tuple[F8MonitorSnapshot, ...], 200),
    RouteContract("get", "/api/presentation", None, tuple[models.PresentationCommand, ...], 200),
    RouteContract("post", "/api/media/sessions", MediaSessionOffer, media.MediaSessionAnswer, 201),
    RouteContract("post", "/api/audio/sessions", AudioSessionOffer, media.AudioSessionAnswer, 201),
    RouteContract("delete", "/api/audio/sessions/{session_id}", None, None, 204),
    RouteContract("post", "/api/media/overlays", OverlayResult, AcceptedResponse, 202),
    RouteContract("get", "/api/media/sessions/{session_id}/media-timestamps/{media_timestamp}", None, media.MediaFrameMapping, 200),
    RouteContract("get", "/api/media/metrics", None, media.MediaMetrics, 200),
    RouteContract("get", "/api/media/gateway", None, media.MediaGatewayHealth, 200),
    RouteContract("get", "/api/media/sample", None, media.MediaSample, 200),
    RouteContract("delete", "/api/media/sessions/{session_id}", None, None, 204),
    RouteContract("get", "/api/projects", None, tuple[models.ProjectSummary, ...], 200),
    RouteContract("post", "/api/projects", CreateProjectRequest, models.ProjectRecord, 201),
    RouteContract("get", "/api/projects/{project_id}", None, models.ProjectRecord, 200),
    RouteContract("delete", "/api/projects/{project_id}", None, None, 204),
    RouteContract("get", "/api/projects/{project_id}/graph/export", None, GraphExchange, 200),
    RouteContract("post", "/api/projects/{project_id}/graph/import", GraphExchange, models.ProjectRecord, 200),
    RouteContract("put", "/api/projects/{project_id}", UpdateProjectRequest, models.ProjectRecord, 200),
    RouteContract("get", "/api/projects/{project_id}/versions", None, tuple[assets.ProjectVersion, ...], 200),
    RouteContract("post", "/api/projects/{project_id}/versions", CreateProjectVersionRequest, assets.ProjectVersion, 201),
    RouteContract("post", "/api/projects/{project_id}/versions/{version_id}/restore", None, models.ProjectRecord, 200),
    RouteContract("post", "/api/editor/sessions", CreateEditorSessionRequest, editor.EditorSessionRecord, 201),
    RouteContract("get", "/api/editor/sessions/{session_id}", None, editor.EditorSessionRecord, 200),
    RouteContract("put", "/api/editor/sessions/{session_id}", UpdateEditorDocumentRequest, editor.EditorSessionRecord, 200),
    RouteContract("post", "/api/editor/sessions/{session_id}/analyze", None, editor.EditorAnalysis, 200),
    RouteContract("post", "/api/editor/sessions/{session_id}/completion", EditorPositionRequest, editor.EditorLanguageResult, 200),
    RouteContract("post", "/api/editor/sessions/{session_id}/hover", EditorPositionRequest, editor.EditorLanguageResult, 200),
    RouteContract("post", "/api/editor/sessions/{session_id}/signature-help", EditorPositionRequest, editor.EditorLanguageResult, 200),
    RouteContract("delete", "/api/editor/sessions/{session_id}", None, None, 204),
    RouteContract("get", "/api/local/capabilities", None, tuple[local.LocalCapability, ...], 200),
    RouteContract("get", "/api/local/serial-ports", None, tuple[local.SerialPortInfo, ...], 200),
    RouteContract("post", "/api/local/modding/detect", DetectModdingTargetRequest, F8JsonValue, 200),
    RouteContract("post", "/api/local/modding/unity/preview", PreviewUnityInstallRequest, local.UnityInstallPlan, 200),
    RouteContract("post", "/api/local/modding/unity/apply", ApplyUnityInstallRequest, F8JsonValue, 200),
    RouteContract("post", "/api/local/modding/verify-udp", VerifySkeletonUdpRequest, local.SkeletonUdpVerification, 200),
    RouteContract("get", "/api/local/hotkeys", None, tuple[local.HotkeyBinding, ...], 200),
    RouteContract("post", "/api/local/hotkeys", RegisterHotkeyRequest, local.HotkeyBinding, 201),
    RouteContract("delete", "/api/local/hotkeys/{binding_id}", None, None, 204),
    RouteContract("post", "/api/projects/{project_id}/validate", ValidateDocumentRequest, ValidationResponse, 200),
    RouteContract("post", "/api/projects/{project_id}/patch", PatchRequest, PatchResult, 200),
    RouteContract("post", "/api/projects/{project_id}/patch:preview", PatchRequest, PatchResult, 200),
    RouteContract("post", "/api/projects/{project_id}/undo", HistoryRequest, PatchResult, 200),
    RouteContract("post", "/api/projects/{project_id}/redo", HistoryRequest, PatchResult, 200),
    RouteContract("post", "/api/projects/{project_id}/deploy", DeployProjectRequest, models.DeployJob, 202),
    RouteContract("get", "/api/agents/providers", None, tuple[agents.AgentProviderSummary, ...], 200),
    RouteContract("get", "/api/agents/providers/settings", None, tuple[settings.ProviderSettingsView, ...], 200),
    RouteContract("post", "/api/decisions/evaluate", DecisionRequest, DecisionResult, 200),
    RouteContract("put", "/api/agents/providers/{provider_id}/settings", UpdateProviderSettings, settings.ProviderSettingsView, 200),
    RouteContract("post", "/api/agents/connections", CreateProviderConnection, settings.ProviderSettingsView, 201),
    RouteContract("delete", "/api/agents/connections/{provider_id}", None, None, 204),
    RouteContract("post", "/api/agents/connections/probe", ProbeProviderRequest, ProviderProbeResult, 200),
    RouteContract("get", "/api/agents/sessions", None, tuple[agents.AgentSessionSummary, ...], 200),
    RouteContract("post", "/api/agents/sessions", CreateAgentSessionRequest, agents.AgentSessionRecord, 201),
    RouteContract("get", "/api/agents/sessions/{session_id}", None, agents.AgentSessionRecord, 200),
    RouteContract("put", "/api/agents/sessions/{session_id}", RenameAgentSessionRequest, agents.AgentSessionRecord, 200),
    RouteContract("delete", "/api/agents/sessions/{session_id}", None, None, 204),
    RouteContract("put", "/api/agents/sessions/{session_id}/model", SelectAgentModelRequest, agents.AgentSessionRecord, 200),
    RouteContract("post", "/api/agents/sessions/{session_id}/runs", StartAgentRunRequest, agents.AgentSessionRecord, 202),
    RouteContract("post", "/api/agents/sessions/{session_id}/approvals/{approval_id}", ResolveAgentApprovalRequest, agents.AgentSessionRecord, 200),
    RouteContract("delete", "/api/agents/sessions/{session_id}/runs/current", None, agents.AgentSessionRecord, 200),
    RouteContract("get", "/api/projects/{project_id}/deployments/latest", None, models.DeployJob | None, 200),
    RouteContract("get", "/api/jobs/{job_id}", None, models.DeployJob, 200),
    RouteContract("delete", "/api/jobs/{job_id}", None, models.DeployJob, 200),
    RouteContract("post", "/api/runtime/services/{service_id}/start", ServiceStartRequest, ManagedProcessResult, 200),
    RouteContract("post", "/api/projects/{project_id}/stop", None, None, 204),
    RouteContract("post", "/api/runtime/services/{service_id}/stop", None, (models.RuntimeActionResult, ManagedProcessResult), 200),
    RouteContract("post", "/api/projects/{project_id}/services/{service_id}/restart", None, models.DeployJob, 202),
    RouteContract("get", "/api/runtime/services/{service_id}/status", None, models.ServiceRuntimeStatus, 200),
    RouteContract("post", "/api/runtime/services/{service_id}/active", ServiceActiveRequest, models.RuntimeActionResult, 200),
    RouteContract("post", "/api/runtime/services/{service_id}/state", ServiceStateRequest, models.RuntimeActionResult, 200),
    RouteContract("post", "/api/runtime/services/{service_id}/nodes/{node_id}/state:read", RuntimeStateReadRequest, models.RuntimeNodeState, 200),
    RouteContract("post", "/api/runtime/services/{service_id}/commands", ServiceCommandRequest, models.RuntimeActionResult, 200),
)


def contract_types() -> tuple[Any, ...]:
    return tuple(item for route in ROUTES for value in (route.request, route.response)
                 if value is not None for item in (cast(tuple[Any, ...], value) if isinstance(value, tuple) else (value,)))


def contract_schemas() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    types = contract_types()
    inputs, outputs, components = model_schemas(types, ref_template="#/components/schemas/{name}")
    cursor = iter(zip(inputs, outputs, strict=True))
    def schema_for(value: Any, *, request: bool) -> dict[str, Any]:
        if isinstance(value, tuple):
            return {"anyOf": [next(cursor)[0 if request else 1] for _ in cast(tuple[Any, ...], value)]}
        return next(cursor)[0 if request else 1]

    paths: dict[tuple[str, str], dict[str, Any]] = {}
    for route in ROUTES:
        operation: dict[str, Any] = {}
        if route.request is not None:
            operation["requestBody"] = {"required": True, "content": {"application/json": {"schema": schema_for(route.request, request=True)}}}
        response: dict[str, Any] = {"description": "Success"}
        if route.response_media_type != "application/json":
            response["content"] = {route.response_media_type: {"schema": {"type": "string", "format": "binary"}}}
        if route.response is not None:
            response["content"] = {"application/json": {"schema": schema_for(route.response, request=False)}}
        operation["responses"] = {str(route.status): response}
        paths[(route.path, route.method)] = operation
    return paths, components


def install_openapi(app: FastAPI) -> None:
    def openapi() -> dict[str, Any]:
        if app.openapi_schema is not None:
            return app.openapi_schema
        result = get_openapi(title=app.title, version=app.version, routes=app.routes)
        paths, components = contract_schemas()
        result.setdefault("components", {}).setdefault("schemas", {}).update(components)
        for (path, method), contract in paths.items():
            operation = result["paths"][path][method]
            operation["responses"].update(contract["responses"])
            if "requestBody" in contract:
                operation["requestBody"] = contract["requestBody"]
        app.openapi_schema = result
        return result
    app.openapi = openapi
