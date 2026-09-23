from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Collection
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypeVar, cast
from urllib.parse import urlparse

import msgspec
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.types import ASGIApp

from f8pysdk.specs import F8JsonValue
from f8media_protocol.client import MediaGatewayRequestError, MediaGatewayUnavailable
from f8media_protocol.contracts import MediaGateway
from f8media_protocol.models import AudioSessionOffer, MediaSessionOffer, OverlayResult
from f8studio_core import API_PROTOCOL_VERSION, HealthStatus, ServerCapabilities
from f8studio_core.graph import (
    GraphValidationError,
    HistoryRequest,
    IdempotencyConflictError,
    OperationTargetError,
    PatchRequest,
    PatchResult,
    RevisionConflictError,
)

from .application import StudioApplication
from .agents import (
    CreateAgentSessionRequest,
    ResolveAgentApprovalRequest,
    StartAgentRunRequest,
)
from .assets import (
    AssetExport,
    AssetKind,
    CreateAssetRequest,
    CreateProjectVersionRequest,
    UpdateAssetRequest,
)
from .editor import CreateEditorSessionRequest, EditorPositionRequest, UpdateEditorDocumentRequest
from .local_integration import (
    ApplyUnityInstallRequest,
    DetectModdingTargetRequest,
    PreviewUnityInstallRequest,
    RegisterHotkeyRequest,
    VerifySkeletonUdpRequest,
)
from .models import (
    BrowserRtcConfiguration,
    CreateCatalogNodeRequest,
    CreateProjectRequest,
    DeployProjectRequest,
    RuntimeNodeState,
    RuntimeStateReadRequest,
    ServiceActiveRequest,
    ServiceCommandRequest,
    ServiceStartRequest,
    ServiceStateRequest,
    UpdateProjectRequest,
    ValidateDocumentRequest,
)
from .runtime import RuntimeConfig, RuntimeGateway


logger = logging.getLogger(__name__)
SERVER_VERSION = "0.1.0"
T = TypeVar("T")


def default_web_dist() -> Path:
    package_bundle = Path(__file__).resolve().parent / "web_dist"
    if (package_bundle / "index.html").is_file():
        return package_bundle
    return Path(__file__).resolve().parents[2] / "f8studio_web" / "dist"


def default_data_dir() -> Path:
    configured = os.environ.get("F8STUDIO_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".local" / "share" / "f8studio-web").resolve()


DEFAULT_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "testserver"})


def _origin_allowed(origin: str | None, allowed_hosts: Collection[str]) -> bool:
    if origin is None:
        return True
    parsed = urlparse(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in allowed_hosts


class LoopbackOriginMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, allowed_hosts: Collection[str]) -> None:
        super().__init__(app)
        self._allowed_hosts = frozenset(allowed_hosts)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path.startswith("/api/") and request.method not in {"GET", "HEAD", "OPTIONS"}:
            if not _origin_allowed(request.headers.get("origin"), self._allowed_hosts):
                return JSONResponse(status_code=403, content={"detail": "request origin is not allowed"})
        return await call_next(request)


def _json_value(value: object) -> F8JsonValue:
    return cast(F8JsonValue, msgspec.to_builtins(value, str_keys=True))


async def _decode_body(request: Request, value_type: type[T]) -> T:
    raw = await request.body()
    try:
        return msgspec.json.decode(raw, type=value_type)
    except msgspec.DecodeError as exc:
        raise HTTPException(status_code=422, detail=f"invalid request body: {exc}") from exc


def _patch_payload(result: PatchResult) -> F8JsonValue:
    return _json_value(
        {
            "requestId": result.request_id,
            "graphChanged": result.graph_changed,
            "layoutChanged": result.layout_changed,
            "document": result.document,
        }
    )


def create_app(
    *,
    web_dist: Path | None = None,
    data_dir: Path | None = None,
    runtime: RuntimeGateway | None = None,
    runtime_config: RuntimeConfig | None = None,
    service_roots: tuple[Path, ...] | None = None,
    application: StudioApplication | None = None,
    media_gateway: MediaGateway | None = None,
    allowed_hosts: tuple[str, ...] | None = None,
    rtc_configuration: BrowserRtcConfiguration | None = None,
) -> FastAPI:
    resolved_web_dist = (web_dist or default_web_dist()).resolve()
    index_path = resolved_web_dist / "index.html"
    has_web_assets = index_path.is_file()
    studio = application or StudioApplication(
        data_dir=data_dir or default_data_dir(), runtime=runtime,
        runtime_config=runtime_config, service_roots=service_roots, media_gateway=media_gateway,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        try:
            await studio.start()
            yield
        finally:
            await studio.close()

    app = FastAPI(
        title="Feel8 Web Studio API",
        version=SERVER_VERSION,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    resolved_allowed_hosts = frozenset(
        host.strip().lower() for host in (allowed_hosts or tuple(DEFAULT_ALLOWED_HOSTS)) if host.strip()
    )
    if not resolved_allowed_hosts:
        raise ValueError("at least one allowed host is required")
    app.add_middleware(LoopbackOriginMiddleware, allowed_hosts=resolved_allowed_hosts)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=sorted(resolved_allowed_hosts),
    )

    @app.exception_handler(GraphValidationError)
    async def graph_validation_error(_request: Request, exc: GraphValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": {"code": exc.code, "message": str(exc)}})

    @app.exception_handler(RevisionConflictError)
    async def revision_conflict(_request: Request, exc: RevisionConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": {"code": exc.code, "message": str(exc)}})

    @app.exception_handler(IdempotencyConflictError)
    async def idempotency_conflict(_request: Request, exc: IdempotencyConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": {"code": exc.code, "message": str(exc)}})

    @app.exception_handler(OperationTargetError)
    async def operation_target_error(_request: Request, exc: OperationTargetError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": {"code": exc.code, "message": str(exc)}})

    @app.exception_handler(FileNotFoundError)
    async def not_found(_request: Request, exc: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(FileExistsError)
    async def already_exists(_request: Request, exc: FileExistsError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid_value(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(MediaGatewayRequestError)
    async def media_gateway_request_error(_request: Request, exc: MediaGatewayRequestError) -> JSONResponse:
        await studio.events.publish(
            event_type="media.error",
            scope="server",
            payload={"operation": "Media gateway", "message": str(exc.detail)},
        )
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(MediaGatewayUnavailable)
    async def media_gateway_unavailable(_request: Request, exc: MediaGatewayUnavailable) -> JSONResponse:
        await studio.events.publish(
            event_type="media.error",
            scope="server",
            payload={"operation": "Media gateway", "message": str(exc)},
        )
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled Web Studio API error", exc_info=exc)
        await studio.events.publish(
            event_type="server.error",
            scope="server",
            payload={"message": f"{type(exc).__name__}: {exc}"},
        )
        return JSONResponse(
            status_code=500,
            content={"detail": {"code": "internal_error", "message": "An internal server error occurred"}},
        )

    @app.get("/api/health")
    async def health() -> F8JsonValue:
        status = HealthStatus(
            status="ok",
            service="f8studio-server",
            version=SERVER_VERSION,
            protocol_version=API_PROTOCOL_VERSION,
            server_epoch=studio.server_epoch,
        )
        return status.to_json_object()

    @app.get("/api/logs")
    async def recent_logs(limit: int = 500) -> F8JsonValue:
        if limit < 1 or limit > 1000:
            raise ValueError("log limit must be between 1 and 1000")
        return _json_value(await studio.events.recent_logs(limit=limit))

    @app.get("/api/capabilities")
    async def capabilities() -> F8JsonValue:
        report = ServerCapabilities(
            graph_editing=True,
            runtime_control=True,
            web_assets=has_web_assets,
            web_rtc_video=True,
            web_rtc_audio=True,
            three_d=True,
            agent_tools=True,
        )
        return {"protocol_version": API_PROTOCOL_VERSION, "capabilities": report.to_json_object()}

    @app.get("/api/media/rtc-configuration")
    async def media_rtc_configuration() -> F8JsonValue:
        return _json_value(rtc_configuration or BrowserRtcConfiguration())

    @app.get("/api/catalog")
    async def catalog() -> F8JsonValue:
        return _json_value(studio.catalog.snapshot())

    @app.post("/api/catalog/nodes")
    async def create_catalog_node(request: Request) -> F8JsonValue:
        payload = await _decode_body(request, CreateCatalogNodeRequest)
        return _json_value(await asyncio.to_thread(studio.catalog.create_node, payload))

    @app.get("/api/assets")
    async def list_assets(kind: AssetKind | None = None) -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.assets.list_assets, kind))

    @app.post("/api/assets", status_code=201)
    async def create_asset(request: Request) -> F8JsonValue:
        payload = await _decode_body(request, CreateAssetRequest)
        record = await asyncio.to_thread(studio.assets.create, payload)
        await studio.events.publish(event_type="asset.created", scope=f"asset:{record.asset_id}", payload=_json_value(record))
        return _json_value(record)

    @app.post("/api/assets/import", status_code=201)
    async def import_asset(request: Request) -> F8JsonValue:
        payload = await _decode_body(request, AssetExport)
        record = await asyncio.to_thread(studio.assets.import_asset, payload)
        await studio.events.publish(event_type="asset.created", scope=f"asset:{record.asset_id}", payload=_json_value(record))
        return _json_value(record)

    @app.get("/api/assets/{asset_id}")
    async def get_asset(asset_id: str) -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.assets.get, asset_id))

    @app.put("/api/assets/{asset_id}")
    async def update_asset(asset_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, UpdateAssetRequest)
        record = await asyncio.to_thread(studio.assets.update, asset_id, payload)
        await studio.events.publish(event_type="asset.updated", scope=f"asset:{record.asset_id}", payload=_json_value(record))
        return _json_value(record)

    @app.delete("/api/assets/{asset_id}", status_code=204)
    async def delete_asset(asset_id: str) -> Response:
        await asyncio.to_thread(studio.assets.delete, asset_id)
        await studio.events.publish(event_type="asset.deleted", scope=f"asset:{asset_id}", payload={"assetId": asset_id})
        return Response(status_code=204)

    @app.get("/api/assets/{asset_id}/versions")
    async def list_asset_versions(asset_id: str) -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.assets.versions, asset_id))

    @app.get("/api/assets/{asset_id}/export")
    async def export_asset(asset_id: str) -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.assets.export, asset_id))

    @app.get("/api/runtime/monitors")
    async def runtime_monitors(project_id: str | None = None) -> F8JsonValue:
        return await studio.tools.monitor_snapshot(project_id)

    @app.get("/api/presentation")
    async def presentation_snapshot() -> F8JsonValue:
        return _json_value(studio.presentation.snapshot())

    @app.post("/api/media/sessions", status_code=201)
    async def create_media_session(request: Request) -> F8JsonValue:
        offer = await _decode_body(request, MediaSessionOffer)
        return _json_value(await studio.media_gateway.create_video_session(offer))

    @app.post("/api/audio/sessions", status_code=201)
    async def create_audio_session(request: Request) -> F8JsonValue:
        offer = await _decode_body(request, AudioSessionOffer)
        return _json_value(await studio.media_gateway.create_audio_session(offer))

    @app.delete("/api/audio/sessions/{session_id}", status_code=204)
    async def close_audio_session(session_id: str) -> Response:
        closed = await studio.media_gateway.close_audio_session(session_id)
        if not closed:
            raise HTTPException(status_code=404, detail="audio session not found")
        return Response(status_code=204)

    @app.post("/api/media/overlays", status_code=202)
    async def publish_media_overlay(request: Request) -> F8JsonValue:
        result = await _decode_body(request, OverlayResult)
        await studio.media_gateway.publish_overlay(result)
        return {"accepted": True}

    @app.get("/api/media/sessions/{session_id}/media-timestamps/{media_timestamp}")
    async def media_frame_mapping(session_id: str, media_timestamp: int) -> F8JsonValue:
        mapping = await studio.media_gateway.frame_mapping(session_id, media_timestamp)
        if mapping is None:
            raise HTTPException(status_code=404, detail="media frame mapping not found")
        return _json_value(mapping)

    @app.get("/api/media/metrics")
    async def media_metrics() -> F8JsonValue:
        return _json_value(await studio.media_gateway.metrics())

    @app.get("/api/media/gateway")
    async def media_gateway_health() -> F8JsonValue:
        return _json_value(await studio.media_gateway.health())

    @app.get("/api/media/sample")
    async def sample_media(source: str, x: int, y: int) -> F8JsonValue:
        try:
            return _json_value(await studio.media_gateway.sample(source, x=x, y=y))
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=f"media source did not produce a frame: {source}") from exc

    @app.delete("/api/media/sessions/{session_id}", status_code=204)
    async def close_media_session(session_id: str) -> Response:
        closed = await studio.media_gateway.close_video_session(session_id)
        if not closed:
            raise HTTPException(status_code=404, detail="media session not found")
        return Response(status_code=204)

    @app.get("/api/projects")
    async def list_projects() -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.projects.list))

    @app.post("/api/projects", status_code=201)
    async def create_project(request: Request) -> F8JsonValue:
        payload = await _decode_body(request, CreateProjectRequest)
        record = await asyncio.to_thread(studio.projects.create, payload)
        await studio.events.publish(
            event_type="project.created",
            scope=f"project:{record.project_id}",
            payload=_json_value(record),
        )
        return _json_value(record)

    @app.get("/api/projects/{project_id}")
    async def get_project(project_id: str) -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.projects.get, project_id))

    @app.put("/api/projects/{project_id}")
    async def update_project(project_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, UpdateProjectRequest)
        record = await asyncio.to_thread(studio.projects.update, project_id, payload)
        await studio.events.publish(
            event_type="project.updated",
            scope=f"project:{project_id}",
            payload=_json_value(record),
        )
        return _json_value(record)

    @app.get("/api/projects/{project_id}/versions")
    async def list_project_versions(project_id: str) -> F8JsonValue:
        await asyncio.to_thread(studio.projects.get, project_id)
        return _json_value(await asyncio.to_thread(studio.assets.list_project_versions, project_id))

    @app.post("/api/projects/{project_id}/versions", status_code=201)
    async def create_project_version(project_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, CreateProjectVersionRequest)
        record = await asyncio.to_thread(studio.projects.get, project_id)
        version = await asyncio.to_thread(
            studio.assets.create_project_version,
            project_id,
            payload.name,
            record.document,
        )
        return _json_value(version)

    @app.post("/api/projects/{project_id}/versions/{version_id}/restore")
    async def restore_project_version(project_id: str, version_id: str) -> F8JsonValue:
        version = await asyncio.to_thread(studio.assets.get_project_version, project_id, version_id)
        record = await asyncio.to_thread(studio.projects.restore, project_id, version.document)
        await asyncio.to_thread(studio.local.refresh_hotkeys)
        await studio.events.publish(
            event_type="graph.committed",
            scope=f"project:{project_id}",
            payload={"requestId": f"restore:{version_id}", "graphChanged": True, "layoutChanged": True, "document": _json_value(record.document)},
        )
        return _json_value(record)

    @app.post("/api/editor/sessions", status_code=201)
    async def create_editor_session(request: Request) -> F8JsonValue:
        payload = await _decode_body(request, CreateEditorSessionRequest)
        return _json_value(await asyncio.to_thread(studio.editor.create, payload))

    @app.get("/api/editor/sessions/{session_id}")
    async def get_editor_session(session_id: str) -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.editor.get, session_id))

    @app.put("/api/editor/sessions/{session_id}")
    async def update_editor_session(session_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, UpdateEditorDocumentRequest)
        return _json_value(await asyncio.to_thread(studio.editor.update, session_id, payload))

    @app.post("/api/editor/sessions/{session_id}/analyze")
    async def analyze_editor_session(session_id: str) -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.editor.analyze, session_id))

    @app.post("/api/editor/sessions/{session_id}/completion")
    async def editor_completion(session_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, EditorPositionRequest)
        return _json_value(await asyncio.to_thread(studio.editor.completion, session_id, payload))

    @app.post("/api/editor/sessions/{session_id}/hover")
    async def editor_hover(session_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, EditorPositionRequest)
        return _json_value(await asyncio.to_thread(studio.editor.hover, session_id, payload))

    @app.delete("/api/editor/sessions/{session_id}", status_code=204)
    async def close_editor_session(session_id: str) -> Response:
        await asyncio.to_thread(studio.editor.close_session, session_id)
        return Response(status_code=204)

    @app.get("/api/local/capabilities")
    async def local_capabilities() -> F8JsonValue:
        return _json_value(studio.local.capabilities())

    @app.get("/api/local/serial-ports")
    async def serial_ports() -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.local.serial_ports))

    @app.post("/api/local/modding/detect")
    async def detect_modding_target(request: Request) -> F8JsonValue:
        payload = await _decode_body(request, DetectModdingTargetRequest)
        return _json_value(await asyncio.to_thread(studio.local.detect_modding_target, payload))

    @app.post("/api/local/modding/unity/preview")
    async def preview_unity_install(request: Request) -> F8JsonValue:
        payload = await _decode_body(request, PreviewUnityInstallRequest)
        return _json_value(await asyncio.to_thread(studio.local.preview_unity_install, payload))

    @app.post("/api/local/modding/unity/apply")
    async def apply_unity_install(request: Request) -> F8JsonValue:
        payload = await _decode_body(request, ApplyUnityInstallRequest)
        return _json_value(await asyncio.to_thread(studio.local.apply_unity_install, payload))

    @app.post("/api/local/modding/verify-udp")
    async def verify_skeleton_udp(request: Request) -> F8JsonValue:
        payload = await _decode_body(request, VerifySkeletonUdpRequest)
        return _json_value(await studio.local.verify_skeleton_udp(payload))

    @app.get("/api/local/hotkeys")
    async def list_hotkeys(project_id: str | None = None) -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.local.list_hotkeys, project_id))

    @app.post("/api/local/hotkeys", status_code=201)
    async def register_hotkey(request: Request) -> F8JsonValue:
        payload = await _decode_body(request, RegisterHotkeyRequest)
        return _json_value(await asyncio.to_thread(studio.local.register_hotkey, payload))

    @app.delete("/api/local/hotkeys/{binding_id}", status_code=204)
    async def unregister_hotkey(binding_id: str) -> Response:
        await asyncio.to_thread(studio.local.unregister_hotkey, binding_id)
        return Response(status_code=204)

    @app.post("/api/projects/{project_id}/validate")
    async def validate_project(project_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, ValidateDocumentRequest)
        if payload.document.project_id != project_id:
            raise ValueError("document projectId does not match route project id")
        await asyncio.to_thread(studio.tools.validate_document, payload.document)
        return {
            "valid": True,
            "graphRevision": payload.document.graph_revision,
            "layoutRevision": payload.document.layout_revision,
        }

    async def runtime_result(operation: str, call: Awaitable[object]) -> F8JsonValue:
        try:
            return _json_value(await call)
        except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("runtime request failed operation=%s", operation, exc_info=exc)
            await studio.events.publish(
                event_type="runtime.error",
                scope="server",
                payload={"operation": operation, "message": f"{type(exc).__name__}: {exc}"},
            )
            raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}") from exc

    @app.post("/api/projects/{project_id}/patch")
    async def patch_project(project_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, PatchRequest)
        return _patch_payload(await studio.tools.apply_patch(project_id, payload))

    @app.post("/api/projects/{project_id}/patch:preview")
    async def preview_project_patch(project_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, PatchRequest)
        result = await asyncio.to_thread(studio.tools.preview_patch, project_id, payload)
        return _patch_payload(result)

    @app.post("/api/projects/{project_id}/undo")
    async def undo_project(project_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, HistoryRequest)
        return _patch_payload(await studio.tools.undo(project_id, payload))

    @app.post("/api/projects/{project_id}/redo")
    async def redo_project(project_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, HistoryRequest)
        return _patch_payload(await studio.tools.redo(project_id, payload))

    @app.post("/api/projects/{project_id}/deploy", status_code=202)
    async def deploy_project(project_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, DeployProjectRequest)
        return _json_value(await studio.tools.deploy(project_id, payload))

    @app.get("/api/agents/providers")
    async def agent_providers() -> F8JsonValue:
        return _json_value(studio.agents.providers())

    @app.get("/api/agents/sessions")
    async def agent_sessions(project_id: str | None = None) -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.agents.list, project_id))

    @app.post("/api/agents/sessions", status_code=201)
    async def create_agent_session(request: Request) -> F8JsonValue:
        payload = await _decode_body(request, CreateAgentSessionRequest)
        return _json_value(await asyncio.to_thread(studio.agents.create, payload))

    @app.get("/api/agents/sessions/{session_id}")
    async def get_agent_session(session_id: str) -> F8JsonValue:
        return _json_value(await asyncio.to_thread(studio.agents.get, session_id))

    @app.post("/api/agents/sessions/{session_id}/runs", status_code=202)
    async def start_agent_run(session_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, StartAgentRunRequest)
        return _json_value(await studio.agents.start_run(session_id, payload))

    @app.post("/api/agents/sessions/{session_id}/approvals/{approval_id}")
    async def resolve_agent_approval(session_id: str, approval_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, ResolveAgentApprovalRequest)
        return _json_value(await studio.agents.resolve_approval(session_id, approval_id, payload))

    @app.delete("/api/agents/sessions/{session_id}/runs/current")
    async def cancel_agent_run(session_id: str) -> F8JsonValue:
        return _json_value(await studio.agents.cancel(session_id))

    @app.get("/api/projects/{project_id}/deployments/latest")
    async def latest_deployment(project_id: str) -> F8JsonValue:
        await asyncio.to_thread(studio.projects.get, project_id)
        return _json_value(await studio.jobs.latest(project_id))

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> F8JsonValue:
        return _json_value(await studio.jobs.get(job_id))

    @app.delete("/api/jobs/{job_id}")
    async def cancel_job(job_id: str) -> F8JsonValue:
        return _json_value(await studio.jobs.cancel(job_id))

    @app.post("/api/runtime/services/{service_id}/start")
    async def start_service(service_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, ServiceStartRequest)
        return _json_value(await studio.processes.start(service_id, service_class=payload.service_class))

    @app.post("/api/runtime/services/{service_id}/stop")
    async def stop_service(service_id: str) -> F8JsonValue:
        try:
            await studio.runtime.terminate(service_id)
        except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
            logger.info(
                "runtime terminate unavailable; stopping managed process service_id=%s",
                service_id,
                exc_info=exc,
            )
        return _json_value(await studio.processes.stop(service_id))

    @app.get("/api/runtime/services/{service_id}/status")
    async def service_status(service_id: str) -> F8JsonValue:
        return await runtime_result(f"status:{service_id}", studio.runtime.status(service_id))

    @app.post("/api/runtime/services/{service_id}/active")
    async def set_service_active(service_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, ServiceActiveRequest)
        return await runtime_result(
            f"active:{service_id}",
            studio.runtime.set_active(service_id, active=payload.active),
        )

    @app.post("/api/runtime/services/{service_id}/state")
    async def set_service_state(service_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, ServiceStateRequest)
        return await runtime_result(
            f"state:{service_id}",
            studio.runtime.set_state(
                service_id,
                node_id=payload.node_id,
                field=payload.field,
                value=payload.value,
            ),
        )

    @app.post("/api/runtime/services/{service_id}/nodes/{node_id}/state:read")
    async def read_node_state(service_id: str, node_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, RuntimeStateReadRequest)
        normalized_fields = tuple(dict.fromkeys(field.strip() for field in payload.fields if field.strip()))
        if len(normalized_fields) > 128:
            raise HTTPException(status_code=422, detail="at most 128 state fields may be read at once")
        fields = await asyncio.gather(
            *(studio.runtime.read_state(service_id, node_id=node_id, field=field) for field in normalized_fields)
        )
        return _json_value(RuntimeNodeState(service_id=service_id, node_id=node_id, fields=tuple(fields)))

    @app.post("/api/runtime/services/{service_id}/commands")
    async def invoke_service_command(service_id: str, request: Request) -> F8JsonValue:
        payload = await _decode_body(request, ServiceCommandRequest)
        return await runtime_result(
            f"command:{service_id}",
            studio.runtime.invoke_command(
                service_id,
                call=payload.call,
                params=payload.params,
            ),
        )

    @app.websocket("/api/events")
    async def events(websocket: WebSocket) -> None:
        if not _origin_allowed(websocket.headers.get("origin"), resolved_allowed_hosts):
            await websocket.close(code=1008, reason="websocket origin is not allowed")
            return
        after_text = websocket.query_params.get("after")
        try:
            after_sequence = None if after_text is None else int(after_text)
        except ValueError:
            await websocket.close(code=1008, reason="after must be an integer")
            return
        stream = await studio.events.open_stream(
            client_epoch=websocket.query_params.get("epoch"),
            after_sequence=after_sequence,
        )
        await websocket.accept()
        try:
            if stream.snapshot_required:
                projects = await asyncio.to_thread(studio.projects.list)
                await websocket.send_json(
                    {
                        "eventId": "snapshot",
                        "serverEpoch": studio.server_epoch,
                        "sequence": stream.current_sequence,
                        "type": "stream.snapshot",
                        "scope": "server",
                        "payload": {
                            "projects": _json_value(projects),
                            "oldestSequence": stream.oldest_sequence,
                        },
                    }
                )
            for event in stream.replay:
                await websocket.send_json(_json_value(event))
            while True:
                event = await stream.queue.get()
                await websocket.send_json(_json_value(event))
                if event.type == "stream.resync_required":
                    await websocket.close(code=1013, reason="event stream resynchronization required")
                    return
        except WebSocketDisconnect:
            return
        finally:
            await studio.events.close_stream(stream.subscription_id)

    if has_web_assets:
        assets_path = resolved_web_dist / "assets"
        if assets_path.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_path), name="web-assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def web_app(path: str) -> FileResponse:
            if path == "api" or path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API route not found")
            requested_path = (resolved_web_dist / path).resolve()
            if path and requested_path.is_relative_to(resolved_web_dist) and requested_path.is_file():
                return FileResponse(requested_path)
            return FileResponse(index_path)

    return app
