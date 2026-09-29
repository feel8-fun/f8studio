from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, TypeVar, cast
from uuid import uuid4

import msgspec

from f8pysdk.generated import F8StateAccess
from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import (
    CreateNodeOp,
    GraphNode,
    NodeCatalog,
    NodeLayout,
    OperatorNode,
    PatchRequest,
    PatchResult,
    RevisionConflictError,
    SetNodeStateOp,
    StudioDocument,
)

from ..automation_tools import StudioAutomationTools
from ..catalog import CatalogSnapshot
from ..editor import CreateEditorSessionRequest, EditorSessionService
from ..editor_context import editor_support_files
from ..events import EventJournal
from ..local_integration import (
    ApplyUnityInstallRequest,
    DetectModdingTargetRequest,
    LocalIntegrationService,
    PreviewUnityInstallRequest,
    VerifySkeletonUdpRequest,
)
from ..models import DeployJob, DeployProjectRequest, JobStatus
from ..project_repository import utc_now_text
from .models import (
    AgentApproval,
    AgentArtifact,
    AgentImage,
    AgentMessage,
    AgentProviderSummary,
    AgentRunStatus,
    AgentSessionRecord,
    AgentSessionSummary,
    AgentToolCall,
    ApprovalStatus,
    CreateAgentSessionRequest,
    ResolveAgentApprovalRequest,
    RenameAgentSessionRequest,
    SelectAgentModelRequest,
    StartAgentRunRequest,
    ToolCallStatus,
)
from .providers import AgentProviderRegistry
from .decisions import SystemOneDecisionClient
from .provider_settings import CreateProviderConnection, ProviderSettingsView, UpdateProviderSettings
from .provider_probe import ProbeProviderRequest, ProviderProbeResult
from .repository import AgentRepository
from .skills import AgentSkillLibrary
from .graph_edits import GraphChanges, build_patch


logger = logging.getLogger(__name__)
T = TypeVar("T")
_TERMINAL_JOBS = {JobStatus.succeeded, JobStatus.partially_failed, JobStatus.failed, JobStatus.cancelled}
_APPROVAL_TTL = timedelta(minutes=5)
_MAX_MODEL_TOOL_CALLS = 48
_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_IMAGES = 3


def _validate_images(images: tuple[AgentImage, ...]) -> None:
    if len(images) > _MAX_IMAGES:
        raise ValueError(f"at most {_MAX_IMAGES} images are allowed per message")
    signatures = {
        "image/png": b"\x89PNG\r\n\x1a\n",
        "image/jpeg": b"\xff\xd8\xff",
        "image/webp": b"RIFF",
        "image/gif": b"GIF8",
    }
    for image in images:
        if not image.name.strip() or len(image.name) > 200:
            raise ValueError("image name must be between 1 and 200 characters")
        header, separator, encoded = image.data_url.partition(",")
        media_type = header.removeprefix("data:").removesuffix(";base64")
        if not separator or header != f"data:{media_type};base64" or media_type not in signatures:
            raise ValueError("image must be a base64 PNG, JPEG, WebP, or GIF data URL")
        if len(encoded) > (_MAX_IMAGE_BYTES + 2) * 4 // 3 + 4:
            raise ValueError("image exceeds the 4 MB limit")
        try:
            data = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise ValueError("image has invalid base64 data") from exc
        if not data or len(data) > _MAX_IMAGE_BYTES or not data.startswith(signatures[media_type]):
            raise ValueError("image format does not match its content or exceeds 4 MB")
        if media_type == "image/webp" and data[8:12] != b"WEBP":
            raise ValueError("image format does not match its content")


def _title_from_prompt(prompt: str) -> str:
    first_line = prompt.strip().splitlines()[0]
    title = " ".join(first_line.split())
    return title[:56].rstrip() + ("..." if len(title) > 56 else "")


class ApprovalDeniedError(RuntimeError):
    pass


@dataclass(frozen=True)
class _PendingApproval:
    session_id: str
    future: asyncio.Future[bool]


def _json_value(value: object) -> F8JsonValue:
    return cast(F8JsonValue, msgspec.to_builtins(value, str_keys=True))


def _tool_text(value: object) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False)


def _arguments_hash(arguments: dict[str, F8JsonValue]) -> str:
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _future_timestamp(delta: timedelta) -> str:
    return (datetime.now(UTC) + delta).isoformat(timespec="milliseconds")


def _catalog_evidence(catalog: CatalogSnapshot) -> F8JsonValue:
    return {"serviceCount": len(catalog.services), "operatorCount": len(catalog.operators)}


def _catalog_index(catalog: CatalogSnapshot) -> F8JsonValue:
    return {
        "services": [{"serviceClass": str(spec.serviceClass), "label": str(spec.label)} for spec in catalog.services],
        "operators": [{"serviceClass": str(spec.serviceClass), "operatorClass": str(spec.operatorClass),
                       "label": str(spec.label)} for spec in catalog.operators],
    }


def _catalog_search(catalog: CatalogSnapshot, query: str) -> F8JsonValue:
    needle = query.strip().casefold()
    if len(needle) < 2:
        raise ValueError("Catalog search needs at least two characters")
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


def _catalog_operator(catalog: CatalogSnapshot, service_class: str, operator_class: str) -> F8JsonValue:
    spec = next((item for item in catalog.operators
                 if str(item.serviceClass) == service_class and str(item.operatorClass) == operator_class), None)
    if spec is None:
        raise ValueError(f"Unknown operator: {service_class}/{operator_class}")
    return _json_value(spec)


def _graph_outline(document: StudioDocument) -> F8JsonValue:
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
        "edges": _json_value(document.edges),
        "layout": _json_value(document.layout),
    }


def _document_evidence(document: StudioDocument) -> F8JsonValue:
    return {
        "projectId": document.project_id,
        "graphRevision": document.graph_revision,
        "layoutRevision": document.layout_revision,
        "nodeCount": len(document.nodes),
        "edgeCount": len(document.edges),
    }


def _patch_evidence(result: PatchResult) -> F8JsonValue:
    return {
        "requestId": result.request_id,
        "graphChanged": result.graph_changed,
        "layoutChanged": result.layout_changed,
        "graphRevision": result.document.graph_revision,
        "layoutRevision": result.document.layout_revision,
        "runtimeErrors": list(result.runtime_errors),
    }


def _deploy_evidence(job: DeployJob) -> F8JsonValue:
    return {
        "jobId": job.job_id,
        "status": job.status.value,
        "sourceGraphRevision": job.source_graph_revision,
        "serviceCount": len(job.service_results),
    }


def _monitor_evidence(snapshot: F8JsonValue) -> F8JsonValue:
    return {"sampleCount": len(snapshot) if isinstance(snapshot, list) else 0}


class AgentService:
    def __init__(
        self,
        *,
        database_path: Path,
        tools: StudioAutomationTools,
        editor: EditorSessionService,
        local: LocalIntegrationService,
        skills: AgentSkillLibrary,
        events: EventJournal,
        providers: AgentProviderRegistry | None = None,
    ) -> None:
        self._repository = AgentRepository(database_path)
        self._tools = tools
        self._editor = editor
        self._local = local
        self._skills = skills
        self._events = events
        self._providers = providers or AgentProviderRegistry(database_path.with_name("agent-providers.json"))
        self.decisions = SystemOneDecisionClient(self._providers)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._approvals: dict[str, _PendingApproval] = {}
        self._lock = asyncio.Lock()
        self._mark_interrupted_sessions()

    def providers(self) -> tuple[AgentProviderSummary, ...]:
        return self._providers.summaries()

    def provider_settings(self) -> tuple[ProviderSettingsView, ...]:
        return self._providers.settings()

    async def update_provider_settings(self, provider_id: str, request: UpdateProviderSettings) -> ProviderSettingsView:
        async with self._lock:
            if any(not task.done() for task in self._tasks.values()):
                raise ValueError("Wait for active agent runs to finish or cancel them before changing provider settings")
            return await asyncio.to_thread(self._providers.update_settings, provider_id, request)

    async def create_provider_connection(self, request: CreateProviderConnection) -> ProviderSettingsView:
        async with self._lock:
            if any(not task.done() for task in self._tasks.values()):
                raise ValueError("Wait for active agent runs before changing provider connections")
            return await asyncio.to_thread(self._providers.create_connection, request)

    async def delete_provider_connection(self, provider_id: str) -> None:
        async with self._lock:
            if any(not task.done() for task in self._tasks.values()):
                raise ValueError("Wait for active agent runs before changing provider connections")
            await asyncio.to_thread(self._providers.delete_connection, provider_id)

    async def probe_provider(self, request: ProbeProviderRequest) -> ProviderProbeResult:
        return await self._providers.probe(request)

    def create(self, request: CreateAgentSessionRequest) -> AgentSessionRecord:
        self._providers.validate_selection(request.provider_id, request.model_id)
        self._tools.project_summary(request.project_id)
        timestamp = utc_now_text()
        title = request.title.strip() or "New agent session"
        record = AgentSessionRecord(
            session_id=uuid4().hex,
            project_id=request.project_id,
            title=title,
            provider_id=request.provider_id,
            model_id=request.model_id,
            status=AgentRunStatus.idle,
            created_at=timestamp,
            updated_at=timestamp,
            auto_title_pending=title in {"Studio agent", "New agent session"},
        )
        return self._repository.save(record)

    def list(self, project_id: str | None = None) -> tuple[AgentSessionSummary, ...]:
        if project_id is not None:
            self._tools.project_summary(project_id)
        return self._repository.list(project_id)

    def get(self, session_id: str) -> AgentSessionRecord:
        record = self._repository.get(session_id)
        if record is None:
            raise FileNotFoundError(f"agent session not found: {session_id}")
        return record

    async def rename(self, session_id: str, request: RenameAgentSessionRequest) -> AgentSessionRecord:
        title = " ".join(request.title.split())
        if not title or len(title) > 120:
            raise ValueError("agent session title must be between 1 and 120 characters")
        async with self._lock:
            record = await asyncio.to_thread(self.get, session_id)
            if record.status in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                raise ValueError("stop the active agent run before renaming its session")
            updated = msgspec.structs.replace(record, title=title, auto_title_pending=False, updated_at=utc_now_text())
            await asyncio.to_thread(self._repository.save, updated)
        await self._publish(updated)
        return updated

    async def select_model(self, session_id: str, request: SelectAgentModelRequest) -> AgentSessionRecord:
        async with self._lock:
            record = await asyncio.to_thread(self.get, session_id)
            if record.status in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                raise ValueError("stop the active agent run before changing its model")
            self._providers.validate_selection(request.provider_id, request.model_id)
            updated = msgspec.structs.replace(
                record, provider_id=request.provider_id, model_id=request.model_id,
                updated_at=utc_now_text(),
            )
            await asyncio.to_thread(self._repository.save, updated)
        await self._publish(updated)
        return updated

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            record = await asyncio.to_thread(self.get, session_id)
            if record.status in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                raise ValueError("stop the active agent run before deleting its session")
            await asyncio.to_thread(self._repository.delete, session_id)
        await self._events.publish(
            event_type="agent.session.deleted",
            scope=f"project:{record.project_id}",
            payload={"sessionId": session_id},
        )

    async def start_run(self, session_id: str, request: StartAgentRunRequest) -> AgentSessionRecord:
        prompt = request.prompt.strip()
        if not prompt and not request.images:
            raise ValueError("agent prompt or image must be provided")
        _validate_images(request.images)
        async with self._lock:
            record = await asyncio.to_thread(self.get, session_id)
            if record.status in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                raise ValueError("agent session already has an active run")
            self._providers.validate_selection(record.provider_id, record.model_id)
            if request.images and not self._providers.supports_image(record.provider_id, record.model_id):
                raise ValueError(f"selected agent model does not support image input: {record.model_id}")
            timestamp = utc_now_text()
            started = msgspec.structs.replace(
                record,
                title=_title_from_prompt(prompt or "Image") if record.auto_title_pending and not record.messages else record.title,
                auto_title_pending=False,
                status=AgentRunStatus.running,
                updated_at=timestamp,
                messages=record.messages
                + (
                    AgentMessage(
                        message_id=uuid4().hex,
                        role="user",
                        content=prompt,
                        created_at=timestamp,
                        images=request.images,
                        provider_id=record.provider_id,
                        model_id=record.model_id,
                    ),
                ),
                approval=None,
                error_message="",
                traceback_id="",
            )
            await asyncio.to_thread(self._repository.save, started)
            task = asyncio.create_task(self._run(started.session_id, prompt, request.reasoning_effort), name=f"agent:{started.session_id}")
            self._tasks[started.session_id] = task
            task.add_done_callback(lambda _task, key=started.session_id: self._tasks.pop(key, None))
        await self._publish(started)
        return started

    async def resolve_approval(
        self,
        session_id: str,
        approval_id: str,
        request: ResolveAgentApprovalRequest,
    ) -> AgentSessionRecord:
        async with self._lock:
            record = await asyncio.to_thread(self.get, session_id)
            approval = record.approval
            if approval is None or approval.approval_id != approval_id:
                raise FileNotFoundError(f"pending agent approval not found: {approval_id}")
            if approval.status is not ApprovalStatus.pending:
                raise ValueError(f"agent approval is already {approval.status.value}")
            if request.arguments_hash != approval.arguments_hash:
                raise ValueError("agent approval argumentsHash does not match the pending tool call")
            pending = self._approvals.get(approval_id)
            if pending is None or pending.session_id != session_id:
                raise ValueError("agent approval is no longer active")

            now = datetime.now(UTC)
            if now >= datetime.fromisoformat(approval.expires_at):
                updated = self._resolve_record_approval(record, ApprovalStatus.expired)
                await asyncio.to_thread(self._repository.save, updated)
                if not pending.future.done():
                    pending.future.set_exception(TimeoutError("agent approval expired"))
                raise ValueError("agent approval expired")

            current = await asyncio.to_thread(self._tools.document, record.project_id)
            if current.graph_revision != approval.target_graph_revision:
                updated = self._resolve_record_approval(record, ApprovalStatus.invalidated)
                await asyncio.to_thread(self._repository.save, updated)
                conflict = RevisionConflictError(
                    f"approval revision conflict: expected {approval.target_graph_revision}, "
                    f"current {current.graph_revision}"
                )
                if not pending.future.done():
                    pending.future.set_exception(conflict)
                raise conflict

            status = ApprovalStatus.approved if request.approved else ApprovalStatus.denied
            updated = self._resolve_record_approval(record, status)
            await asyncio.to_thread(self._repository.save, updated)
            if not pending.future.done():
                pending.future.set_result(request.approved)
        await self._publish(updated)
        return updated

    async def cancel(self, session_id: str) -> AgentSessionRecord:
        async with self._lock:
            record = await asyncio.to_thread(self.get, session_id)
            if record.status not in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                return record
            task = self._tasks.get(session_id)
            if task is not None:
                task.cancel()
            approval = record.approval
            if approval is not None and approval.status is ApprovalStatus.pending:
                pending = self._approvals.get(approval.approval_id)
                if pending is not None and not pending.future.done():
                    pending.future.cancel()
                record = self._resolve_record_approval(record, ApprovalStatus.cancelled)
            tool_calls = tuple(
                msgspec.structs.replace(
                    call,
                    status=ToolCallStatus.cancelled,
                    updated_at=utc_now_text(),
                    error_message="agent run cancelled before tool completion",
                )
                if call.status in {
                    ToolCallStatus.queued,
                    ToolCallStatus.running,
                    ToolCallStatus.waiting_for_approval,
                }
                else call
                for call in record.tool_calls
            )
            cancelled = msgspec.structs.replace(
                record,
                status=AgentRunStatus.cancelled,
                updated_at=utc_now_text(),
                tool_calls=tool_calls,
                error_message="agent run cancelled; completed side effects were not rolled back",
            )
            await asyncio.to_thread(self._repository.save, cancelled)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await self._publish(cancelled)
        return cancelled

    async def close(self) -> None:
        session_ids = tuple(self._tasks)
        for session_id in session_ids:
            await self.cancel(session_id)
        self._tasks.clear()
        self._approvals.clear()
        await self.decisions.close()

    async def _run(self, session_id: str, prompt: str, reasoning_effort: Literal["low", "medium", "high"] | None = None) -> None:
        try:
            record = await asyncio.to_thread(self.get, session_id)
            if record.provider_id != "deterministic":
                supports_images = self._providers.supports_image(record.provider_id, record.model_id)
                images = tuple(image for message in record.messages[-9:] for image in message.images)[-_MAX_IMAGES:] if supports_images else ()
                provider_input = self._conversation_prompt(record, prompt)
                if images:
                    provider_input += "\nAttached images are ordered by their appearance in the conversation."
                provider_run = self._providers.run_with_tools(
                    provider_id=record.provider_id, model_id=record.model_id,
                    prompt=provider_input, tools=self._model_tools(record),
                    images=images, reasoning_effort=reasoning_effort,
                )
                response = await asyncio.wait_for(provider_run, timeout=300)
            else:
                catalog = await self._tool(
                    record,
                    tool_name="catalog.read",
                    arguments={},
                    target_graph_revision=None,
                    operation=lambda: asyncio.to_thread(self._tools.catalog),
                    result_encoder=_catalog_evidence,
                )
                record = await asyncio.to_thread(self.get, session_id)
                document = await self._tool(
                    record,
                    tool_name="graph.read",
                    arguments={"projectId": record.project_id},
                    target_graph_revision=None,
                    operation=lambda: asyncio.to_thread(self._tools.document, record.project_id),
                    result_encoder=_document_evidence,
                )
                if "diagnos" in prompt.lower() or "诊断" in prompt:
                    await self._run_diagnostics(record, document)
                else:
                    await self._run_graph_build(record, catalog, document)
                finished = await asyncio.to_thread(self.get, session_id)
                response = await self._providers.complete(
                    provider_id=finished.provider_id,
                    model_id=finished.model_id,
                    prompt=self._evidence_prompt(finished),
                )
            finished = await asyncio.to_thread(self.get, session_id)
            completed = self._append_message(finished, role="assistant", content=response)
            completed = msgspec.structs.replace(
                completed,
                status=AgentRunStatus.succeeded,
                updated_at=utc_now_text(),
                error_message="",
                traceback_id="",
            )
            await asyncio.to_thread(self._repository.save, completed)
            await self._publish(completed)
        except asyncio.CancelledError:
            raise
        except ApprovalDeniedError as exc:
            await self._finish_stopped(session_id, AgentRunStatus.cancelled, str(exc))
        except Exception as exc:
            traceback_id = uuid4().hex
            logger.exception("agent run failed session_id=%s traceback_id=%s", session_id, traceback_id)
            await self._finish_stopped(
                session_id,
                AgentRunStatus.failed,
                f"{type(exc).__name__}: {exc}",
                traceback_id=traceback_id,
            )

    def _code_target(self, project_id: str, node_id: str) -> tuple[StudioDocument, GraphNode, str]:
        document = self._tools.document(project_id)
        node = next((item for item in document.nodes if item.node_id == node_id), None)
        if node is None:
            raise FileNotFoundError(f"code node not found: {node_id}")
        editor_support_files(node, "code")
        state_fields = node.spec.stateFields
        fields = () if isinstance(state_fields, msgspec.UnsetType) else state_fields
        field = next(item for item in fields if item.name == "code")
        control = field.control
        if isinstance(control, msgspec.UnsetType) or control.language != "python":
            raise ValueError(f"code field is not Python: {node_id}")
        if field.access is not F8StateAccess.rw:
            raise ValueError(f"code field is not writable: {node_id}")
        incoming = {port.port_id for port in node.ports if port.kind.value == "state" and port.name == "code"}
        if any(edge.to_node_id == node_id and edge.to_port_id in incoming for edge in document.edges):
            raise ValueError(f"code field is driven by an incoming state connection: {node_id}")
        value = node.state_values.get("code", "")
        if not isinstance(value, str):
            raise TypeError(f"code field is not text: {node_id}")
        return document, node, value

    @staticmethod
    def _conversation_prompt(record: AgentSessionRecord, prompt: str) -> str:
        earlier = record.messages[:-1][-8:]
        if not earlier:
            return prompt
        history = [{"role": message.role, "content": message.content[:4000],
                    "images": [image.name for image in message.images]} for message in earlier]
        return f"Previous conversation:\n{json.dumps(history, ensure_ascii=False)}\nCurrent request:\n{prompt}"

    async def _check_model_tool_budget(self, record: AgentSessionRecord) -> None:
        if record.provider_id == "deterministic":
            return
        latest = await asyncio.to_thread(self.get, record.session_id)
        if len(latest.tool_calls) - len(record.tool_calls) >= _MAX_MODEL_TOOL_CALLS:
            raise RuntimeError(f"agent run exceeded {_MAX_MODEL_TOOL_CALLS} tool calls")

    def _analyze_code(self, project_id: str, node_id: str, code: str) -> object:
        _, node, _ = self._code_target(project_id, node_id)
        session = self._editor.create(
            CreateEditorSessionRequest(
                language="python",
                text=code,
                filename="state.py",
                support_files=editor_support_files(node, "code"),
                project_id=project_id,
                node_id=node_id,
                field_name="code",
            )
        )
        try:
            return self._editor.analyze(session.session_id)
        finally:
            self._editor.close_session(session.session_id)

    def _model_tools(self, record: AgentSessionRecord) -> tuple[Callable[..., Awaitable[str]], ...]:
        project_id = record.project_id
        previewed_patches: set[str] = set()
        proposals: dict[str, PatchRequest] = {}
        previewed_unity_plans: set[str] = set()

        async def catalog_read() -> str:
            """List installed service and operator IDs/labels. Use catalog_search and catalog_operator to inspect relevant nodes."""
            result = await self._tool(
                record, tool_name="catalog.read", arguments={}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._tools.catalog), result_encoder=_catalog_evidence,
            )
            return _tool_text(_catalog_index(result))

        async def catalog_search(query: str) -> str:
            """Find operators by name, class, or description; returns matching IDs for catalog_operator."""
            result = await self._tool(
                record, tool_name="catalog.search", arguments={"query": query}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(_catalog_search, self._tools.catalog(), query),
            )
            return _tool_text(result)

        async def catalog_operator(service_class: str, operator_class: str) -> str:
            """Read exact operator specification: behavior, ports, state fields, defaults, and constraints."""
            result = await self._tool(
                record, tool_name="catalog.operator",
                arguments={"serviceClass": service_class, "operatorClass": operator_class}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(_catalog_operator, self._tools.catalog(), service_class, operator_class),
            )
            return _tool_text(result)

        async def catalog_create_node(node_id: str, service_class: str, operator_class: str,
                                      service_id: str, name: str) -> str:
            """Create the exact operator-node JSON for a createNode patch; does not edit the graph."""
            def create() -> GraphNode:
                snapshot = self._tools.catalog()
                catalog = NodeCatalog(services=snapshot.services, operators=snapshot.operators)
                document = self._tools.document(project_id)
                if any(node.node_id == node_id for node in document.nodes):
                    raise ValueError(f"Node ID already exists: {node_id}")
                if not any(node.service_id == service_id and node.service_class == service_class
                           and not isinstance(node, OperatorNode) for node in document.nodes):
                    raise ValueError(f"Service instance not found: {service_id} ({service_class})")
                return catalog.create_operator_node(
                    node_id=node_id, service_id=service_id, service_class=service_class,
                    operator_class=operator_class, name=name,
                )
            result = await self._tool(
                record, tool_name="catalog.create_node",
                arguments={"nodeId": node_id, "serviceClass": service_class,
                           "operatorClass": operator_class, "serviceId": service_id, "name": name},
                target_graph_revision=None, operation=lambda: asyncio.to_thread(create),
                result_encoder=lambda node: {"nodeId": node.node_id, "operatorClass": operator_class},
            )
            return _tool_text(result)

        async def skills_list() -> str:
            """List available Studio workflow skills, including locally installed game skills."""
            result = await self._tool(
                record, tool_name="skills.list", arguments={}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._skills.list),
            )
            return _tool_text(result)

        async def skill_read(skill_id: str) -> str:
            """Read an available Studio skill by ID before using its workflow guidance."""
            result = await self._tool(
                record, tool_name="skills.read", arguments={"skillId": skill_id}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._skills.read, skill_id),
            )
            return result

        async def graph_read() -> str:
            """Read current graph revisions, node IDs, state values, ports, edges, and layout; use graph_node for full node JSON."""
            result = await self._tool(
                record, tool_name="graph.read", arguments={"projectId": project_id}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._tools.document, project_id), result_encoder=_document_evidence,
            )
            return _tool_text(_graph_outline(result))

        async def graph_node(node_id: str) -> str:
            """Read one existing graph node with its full spec and port IDs for patch construction."""
            def read() -> GraphNode:
                document = self._tools.document(project_id)
                node = next((item for item in document.nodes if item.node_id == node_id), None)
                if node is None:
                    raise ValueError(f"Node not found: {node_id}")
                return node
            result = await self._tool(
                record, tool_name="graph.node", arguments={"nodeId": node_id}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(read),
                result_encoder=lambda node: {"nodeId": node.node_id},
            )
            return _tool_text(result)

        async def graph_preview_patch(patch_json: str) -> str:
            """Validate a JSON PatchRequest against the current project without changing it."""
            patch = msgspec.json.decode(patch_json, type=PatchRequest)
            result = await self._tool(
                record, tool_name="graph.preview_patch", arguments={"projectId": project_id, "patch": _json_value(patch)},
                target_graph_revision=patch.expected_graph_revision,
                operation=lambda: asyncio.to_thread(self._tools.preview_patch, project_id, patch),
                result_encoder=_patch_evidence,
            )
            previewed_patches.add(_arguments_hash({"patch": _json_value(patch)}))
            await self._append_artifact(record.session_id, AgentArtifact(
                artifact_id=uuid4().hex, kind="graph_patch", title="Proposed graph patch",
                payload={"patch": _json_value(patch), "afterGraphRevision": result.document.graph_revision},
                created_at=utc_now_text(),
            ))
            return _tool_text(_patch_evidence(result))

        async def graph_apply_patch(patch_json: str) -> str:
            """Apply a previously previewed JSON PatchRequest after human approval."""
            patch = msgspec.json.decode(patch_json, type=PatchRequest)
            if any(isinstance(operation, SetNodeStateOp) and operation.field == "code" for operation in patch.operations):
                raise ValueError("use code_read, code_analyze, and code_write to edit Python node code")
            fingerprint = _arguments_hash({"patch": _json_value(patch)})
            if fingerprint not in previewed_patches:
                raise ValueError("graph patch must be previewed in this run before applying")
            result = await self._approved_tool(
                record, tool_name="graph.apply_patch", arguments={"projectId": project_id, "patch": _json_value(patch)},
                target_graph_revision=patch.expected_graph_revision,
                operation=lambda: self._tools.apply_patch(project_id, patch), result_encoder=_patch_evidence,
            )
            previewed_patches.discard(fingerprint)
            return _tool_text(_patch_evidence(result))

        async def graph_propose_changes(changes_json: str) -> str:
            """Build and preview a graph patch from compact JSON; does not change the graph.

            Prefer this over copying full specs into graph_preview_patch. JSON shape:
            {"expectedGraphRevision": 0, "expectedLayoutRevision": 0,
             "nodes": [{"nodeId": "phase", "serviceClass": "f8.pyengine", "serviceId": "engine",
                        "operatorClass": "f8.phase", "name": "Phase", "stateValues": {"hz": 1}, "x": 100, "y": 100}],
             "connections": [{"fromNodeId": "phase", "fromPort": "phase", "toNodeId": "cosine", "toPort": "phase", "kind": "data"}],
             "stateUpdates": [{"nodeId": "wave", "field": "upstreamSampleIntervalMs", "value": 20}]}
            Nodes, connections, and stateUpdates are optional arrays. Connections use port names, not IDs.
            To create a service omit operatorClass/serviceId. Reuse existing services and nodes.
            Returns proposalId; immediately call graph_apply_proposal to show the approval UI.
            """
            changes = msgspec.json.decode(changes_json, type=GraphChanges)
            proposal_id = uuid4().hex

            def prepare() -> tuple[PatchRequest, PatchResult]:
                document = self._tools.document(project_id)
                patch = build_patch(document, self._tools.catalog(), changes, request_id=f"agent:{proposal_id}")
                preview = self._tools.preview_patch(project_id, patch)
                self._tools.validate_document(preview.document)
                return patch, preview

            patch, preview = await self._tool(
                record, tool_name="graph.propose_changes", arguments={"changes": _json_value(changes)},
                target_graph_revision=changes.expected_graph_revision,
                operation=lambda: asyncio.to_thread(prepare),
                result_encoder=lambda result: {"proposalId": proposal_id, "operationCount": len(result[0].operations),
                                               "preview": _patch_evidence(result[1])},
            )
            proposals[proposal_id] = patch
            await self._append_artifact(record.session_id, AgentArtifact(
                artifact_id=proposal_id, kind="graph_patch", title="Proposed graph changes",
                payload={"patch": _json_value(patch), "changes": _json_value(changes),
                         "afterGraphRevision": preview.document.graph_revision}, created_at=utc_now_text(),
            ))
            return _tool_text({"proposalId": proposal_id, "operationCount": len(patch.operations),
                               "nextAction": "Call graph_apply_proposal with this proposalId to request approval."})

        async def graph_apply_proposal(proposal_id: str) -> str:
            """Request user approval and apply the exact patch prepared by graph_propose_changes.

            Calling this tool opens the approval UI and waits for the user's decision. Do not ask
            for approval in chat instead. No graph mutation happens before approval.
            """
            patch = proposals.get(proposal_id)
            if patch is None:
                raise ValueError("Unknown proposalId; call graph_propose_changes in this run first")
            result = await self._approved_tool(
                record, tool_name="graph.apply_patch", arguments={"projectId": project_id, "patch": _json_value(patch)},
                target_graph_revision=patch.expected_graph_revision,
                operation=lambda: self._tools.apply_patch(project_id, patch), result_encoder=_patch_evidence,
            )
            proposals.pop(proposal_id)
            return _tool_text(_patch_evidence(result))

        async def code_read(node_id: str) -> str:
            """Read the Python code in one node, its graph revision, and its SHA-256 content hash."""
            document, node, code = await self._tool(
                record, tool_name="code.read", arguments={"nodeId": node_id}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._code_target, project_id, node_id),
                result_encoder=lambda result: {"nodeId": result[1].node_id, "graphRevision": result[0].graph_revision},
            )
            return _tool_text({
                "nodeId": node.node_id, "nodeName": node.name, "graphRevision": document.graph_revision,
                "codeSha256": hashlib.sha256(code.encode("utf-8")).hexdigest(), "code": code,
            })

        async def code_analyze(node_id: str, code: str) -> str:
            """Analyze proposed Python node code with its generated Studio API support files."""
            result = await self._tool(
                record, tool_name="code.analyze", arguments={"nodeId": node_id, "code": code},
                target_graph_revision=None,
                operation=lambda: asyncio.to_thread(self._analyze_code, project_id, node_id, code),
            )
            return _tool_text(result)

        async def code_write(node_id: str, expected_graph_revision: int, expected_code_sha256: str, code: str) -> str:
            """Write Python code to one node after human approval; requires the read revision and content hash."""
            compile(code, f"{node_id}.py", "exec")
            document, _, current = await asyncio.to_thread(self._code_target, project_id, node_id)
            digest = hashlib.sha256(current.encode("utf-8")).hexdigest()
            if document.graph_revision != expected_graph_revision or digest != expected_code_sha256:
                raise RevisionConflictError(f"code changed since read: {node_id}; read the node again")
            await self._append_artifact(record.session_id, AgentArtifact(
                artifact_id=uuid4().hex, kind="text", title=f"Proposed code change: {node_id}",
                payload={"nodeId": node_id, "beforeSha256": digest, "before": current, "after": code},
                created_at=utc_now_text(),
            ))
            patch = PatchRequest(
                request_id=f"agent-code:{record.session_id}:{uuid4().hex}",
                expected_graph_revision=expected_graph_revision,
                expected_layout_revision=document.layout_revision,
                operations=(SetNodeStateOp(node_id=node_id, field="code", value=code),),
            )
            async def apply_code() -> PatchResult:
                _, _, latest = await asyncio.to_thread(self._code_target, project_id, node_id)
                if hashlib.sha256(latest.encode("utf-8")).hexdigest() != expected_code_sha256:
                    raise RevisionConflictError(f"code changed during approval: {node_id}")
                return await self._tools.apply_patch(project_id, patch)

            result = await self._approved_tool(
                record, tool_name="code.write",
                arguments={"nodeId": node_id, "expectedGraphRevision": expected_graph_revision,
                           "expectedCodeSha256": expected_code_sha256, "code": code},
                target_graph_revision=expected_graph_revision, operation=apply_code, result_encoder=_patch_evidence,
            )
            return _tool_text({
                "graphRevision": result.document.graph_revision,
                "layoutRevision": result.document.layout_revision,
                "runtimeErrors": list(result.runtime_errors),
            })

        async def graph_validate() -> str:
            """Validate the current graph after edits."""
            document = await asyncio.to_thread(self._tools.document, project_id)
            await self._tool(
                record, tool_name="graph.validate", arguments={"graphRevision": document.graph_revision},
                target_graph_revision=document.graph_revision,
                operation=lambda: asyncio.to_thread(self._tools.validate_document, document),
            )
            return _tool_text({"valid": True, "graphRevision": document.graph_revision})

        async def project_deploy() -> str:
            """Deploy the current graph after human approval and wait for the deployment result."""
            document = await asyncio.to_thread(self._tools.document, project_id)
            request = DeployProjectRequest(
                request_id=f"agent-deploy:{record.session_id}:{uuid4().hex}",
                expected_graph_revision=document.graph_revision,
            )
            result = await self._approved_tool(
                record, tool_name="project.deploy", arguments={"projectId": project_id, "request": _json_value(request)},
                target_graph_revision=document.graph_revision,
                operation=lambda: self._deploy_and_wait(project_id, request), result_encoder=_deploy_evidence,
            )
            return _tool_text(result)

        async def runtime_observe() -> str:
            """Read the project's current runtime monitor samples after deployment."""
            result = await self._tool(
                record, tool_name="runtime.observe", arguments={"projectId": project_id}, target_graph_revision=None,
                operation=lambda: self._tools.monitor_snapshot(project_id), result_encoder=_monitor_evidence,
            )
            return _tool_text(result)

        async def logs_read(limit: int = 50) -> str:
            """Read recent Studio logs for debugging; limit must be between 1 and 100."""
            if not 1 <= limit <= 100:
                raise ValueError("log limit must be between 1 and 100")
            result = await self._tool(
                record, tool_name="logs.read", arguments={"limit": limit}, target_graph_revision=None,
                operation=lambda: self._events.recent_logs(limit=limit),
            )
            return _tool_text(result)

        async def modding_detect_target(target_path: str) -> str:
            """Detect the game engine and whether Studio has a supported installer for this local target."""
            result = await self._tool(
                record, tool_name="modding.detect_target", arguments={"targetPath": target_path}, target_graph_revision=None,
                operation=lambda: asyncio.to_thread(
                    self._local.detect_modding_target, DetectModdingTargetRequest(target_path=target_path)
                ),
            )
            return _tool_text(result)

        async def modding_preview_unity_install(target_path: str) -> str:
            """Preview exact Unity exporter installation actions and files; does not write to the game."""
            result = await self._tool(
                record, tool_name="modding.preview_unity_install", arguments={"targetPath": target_path},
                target_graph_revision=None,
                operation=lambda: asyncio.to_thread(
                    self._local.preview_unity_install, PreviewUnityInstallRequest(target_path=target_path)
                ),
            )
            previewed_unity_plans.add(result.plan_id)
            await self._append_artifact(record.session_id, AgentArtifact(
                artifact_id=uuid4().hex, kind="text", title="Unity installation preview",
                payload=_json_value(result), created_at=utc_now_text(),
            ))
            return _tool_text(result)

        async def modding_apply_unity_install(plan_id: str) -> str:
            """Install the exact previewed Unity plan after human approval; never use for Unreal."""
            if plan_id not in previewed_unity_plans:
                raise ValueError("Unity installation plan must be previewed in this run before applying")
            document = await asyncio.to_thread(self._tools.document, project_id)
            result = await self._approved_tool(
                record, tool_name="modding.apply_unity_install", arguments={"planId": plan_id},
                target_graph_revision=document.graph_revision,
                operation=lambda: asyncio.to_thread(
                    self._local.apply_unity_install, ApplyUnityInstallRequest(plan_id=plan_id, confirm=True)
                ),
            )
            previewed_unity_plans.discard(plan_id)
            return _tool_text(result)

        async def modding_verify_udp(port: int = 39540) -> str:
            """Verify a complete decoded skeleton frame from the game exporter on a UDP port."""
            result = await self._tool(
                record, tool_name="modding.verify_udp", arguments={"port": port}, target_graph_revision=None,
                operation=lambda: self._local.verify_skeleton_udp(VerifySkeletonUdpRequest(port=port)),
            )
            return _tool_text(result)

        return (
            skills_list, skill_read, catalog_read, catalog_search, catalog_operator, catalog_create_node,
            graph_read, graph_node, graph_preview_patch, graph_apply_patch,
            graph_propose_changes, graph_apply_proposal,
            code_read, code_analyze, code_write, graph_validate, project_deploy,
            runtime_observe, logs_read, modding_detect_target, modding_preview_unity_install,
            modding_apply_unity_install, modding_verify_udp,
        )

    async def _run_diagnostics(self, record: AgentSessionRecord, document: object) -> None:
        if not isinstance(document, StudioDocument):
            raise TypeError("graph.read returned an invalid document")
        await self._tool(
            record,
            tool_name="graph.validate",
            arguments={"projectId": record.project_id, "graphRevision": document.graph_revision},
            target_graph_revision=document.graph_revision,
            operation=lambda: asyncio.to_thread(self._tools.validate_document, document),
        )
        latest = await asyncio.to_thread(self.get, record.session_id)
        monitors = await self._tool(
            latest,
            tool_name="runtime.observe",
            arguments={"projectId": record.project_id},
            target_graph_revision=document.graph_revision,
            operation=lambda: self._tools.monitor_snapshot(record.project_id),
            result_encoder=_monitor_evidence,
        )
        artifact = AgentArtifact(
            artifact_id=uuid4().hex,
            kind="diagnostics",
            title="Graph diagnostics",
            payload={
                "valid": True,
                "graphRevision": document.graph_revision,
                "nodeCount": len(document.nodes),
                "edgeCount": len(document.edges),
                "monitors": monitors,
            },
            created_at=utc_now_text(),
        )
        await self._append_artifact(record.session_id, artifact)

    async def _run_graph_build(self, record: AgentSessionRecord, catalog: object, document: object) -> None:
        if not isinstance(catalog, CatalogSnapshot) or not isinstance(document, StudioDocument):
            raise TypeError("agent graph context has invalid catalog or document")
        operations = self._build_value_stepper_operations(catalog, document)
        if not operations:
            artifact = AgentArtifact(
                artifact_id=uuid4().hex,
                kind="diagnostics",
                title="Graph already satisfies goal",
                payload={"graphRevision": document.graph_revision, "changed": False},
                created_at=utc_now_text(),
            )
            await self._append_artifact(record.session_id, artifact)
            return
        patch = PatchRequest(
            request_id=f"agent:{record.session_id}:{uuid4().hex}",
            expected_graph_revision=document.graph_revision,
            expected_layout_revision=document.layout_revision,
            operations=operations,
        )
        patch_json = cast(dict[str, F8JsonValue], _json_value(patch))
        latest = await asyncio.to_thread(self.get, record.session_id)
        preview = await self._tool(
            latest,
            tool_name="graph.preview_patch",
            arguments={"projectId": record.project_id, "patch": patch_json},
            target_graph_revision=document.graph_revision,
            operation=lambda: asyncio.to_thread(self._tools.preview_patch, record.project_id, patch),
            result_encoder=_patch_evidence,
        )
        artifact = AgentArtifact(
            artifact_id=uuid4().hex,
            kind="graph_patch",
            title="Proposed graph patch",
            payload={
                "patch": patch_json,
                "beforeGraphRevision": document.graph_revision,
                "afterGraphRevision": preview.document.graph_revision,
                "operationCount": len(operations),
            },
            created_at=utc_now_text(),
        )
        await self._append_artifact(record.session_id, artifact)
        latest = await asyncio.to_thread(self.get, record.session_id)
        applied = await self._approved_tool(
            latest,
            tool_name="graph.apply_patch",
            arguments={"projectId": record.project_id, "patch": patch_json},
            target_graph_revision=document.graph_revision,
            operation=lambda: self._tools.apply_patch(record.project_id, patch),
            result_encoder=_patch_evidence,
        )
        latest = await asyncio.to_thread(self.get, record.session_id)
        await self._tool(
            latest,
            tool_name="graph.validate",
            arguments={"projectId": record.project_id, "graphRevision": applied.document.graph_revision},
            target_graph_revision=applied.document.graph_revision,
            operation=lambda: asyncio.to_thread(self._tools.validate_document, applied.document),
        )
        deploy_request = DeployProjectRequest(
            request_id=f"agent-deploy:{record.session_id}:{uuid4().hex}",
            expected_graph_revision=applied.document.graph_revision,
        )
        deploy_json = cast(dict[str, F8JsonValue], _json_value(deploy_request))
        latest = await asyncio.to_thread(self.get, record.session_id)
        job = await self._approved_tool(
            latest,
            tool_name="project.deploy",
            arguments={"projectId": record.project_id, "request": deploy_json},
            target_graph_revision=applied.document.graph_revision,
            operation=lambda: self._deploy_and_wait(record.project_id, deploy_request),
            result_encoder=_deploy_evidence,
        )
        await self._append_artifact(
            record.session_id,
            AgentArtifact(
                artifact_id=uuid4().hex,
                kind="deployment",
                title="Deployment result",
                payload=_json_value(job),
                created_at=utc_now_text(),
            ),
        )
        latest = await asyncio.to_thread(self.get, record.session_id)
        monitors = await self._tool(
            latest,
            tool_name="runtime.observe",
            arguments={"projectId": record.project_id},
            target_graph_revision=applied.document.graph_revision,
            operation=lambda: self._tools.monitor_snapshot(record.project_id),
            result_encoder=_monitor_evidence,
        )
        await self._append_artifact(
            record.session_id,
            AgentArtifact(
                artifact_id=uuid4().hex,
                kind="monitor",
                title="Runtime monitor evidence",
                payload=monitors,
                created_at=utc_now_text(),
            ),
        )

    @staticmethod
    def _build_value_stepper_operations(catalog: object, document: object) -> tuple[CreateNodeOp, ...]:
        if not isinstance(catalog, CatalogSnapshot) or not isinstance(document, StudioDocument):
            raise TypeError("invalid graph builder inputs")
        node_catalog = NodeCatalog(services=catalog.services, operators=catalog.operators)
        service = next(
            (node for node in document.nodes if not isinstance(node, OperatorNode) and node.service_class == "f8.pystudio"),
            None,
        )
        operations: list[CreateNodeOp] = []
        if service is None:
            service = node_catalog.create_service_node(node_id="studio", service_class="f8.pystudio", name="Studio")
            operations.append(
                CreateNodeOp(node=service, layout=NodeLayout(node_id=service.node_id, x=80.0, y=80.0, width=524.0, height=300.0))
            )
        existing_stepper = next(
            (
                node
                for node in document.nodes
                if isinstance(node, OperatorNode) and node.operator_class == "f8.value_stepper"
            ),
            None,
        )
        if existing_stepper is None:
            node_id = "agent_value_stepper"
            occupied = {node.node_id for node in document.nodes}
            suffix = 2
            while node_id in occupied:
                node_id = f"agent_value_stepper_{suffix}"
                suffix += 1
            operator = node_catalog.create_operator_node(
                node_id=node_id,
                service_id=service.service_id,
                service_class="f8.pystudio",
                operator_class="f8.value_stepper",
                name="AI Value Stepper",
            )
            operations.append(
                CreateNodeOp(node=operator, layout=NodeLayout(node_id=node_id, x=120.0, y=150.0, width=240.0))
            )
        return tuple(operations)

    async def _deploy_and_wait(self, project_id: str, request: DeployProjectRequest) -> DeployJob:
        job = await self._tools.deploy(project_id, request)
        while job.status not in _TERMINAL_JOBS:
            await asyncio.sleep(0.02)
            job = await self._tools.deployment(job.job_id)
        if job.status is not JobStatus.succeeded:
            detail = job.error_message or "deployment did not succeed"
            raise RuntimeError(f"deployment {job.job_id} finished with {job.status.value}: {detail}")
        return job

    async def _tool(
        self,
        record: AgentSessionRecord,
        *,
        tool_name: str,
        arguments: dict[str, F8JsonValue],
        target_graph_revision: int | None,
        operation: Callable[[], Awaitable[T]],
        result_encoder: Callable[[T], F8JsonValue] | None = None,
    ) -> T:
        await self._check_model_tool_budget(record)
        call = AgentToolCall(
            tool_call_id=uuid4().hex,
            tool_name=tool_name,
            arguments=arguments,
            arguments_hash=_arguments_hash(arguments),
            target_graph_revision=target_graph_revision,
            status=ToolCallStatus.running,
            created_at=utc_now_text(),
            updated_at=utc_now_text(),
        )
        await self._append_tool_call(record.session_id, call)
        try:
            result = await operation()
        except Exception as exc:
            traceback_id = uuid4().hex
            logger.exception(
                "agent tool failed session_id=%s tool_call_id=%s tool=%s traceback_id=%s",
                record.session_id,
                call.tool_call_id,
                tool_name,
                traceback_id,
            )
            await self._update_tool_call(
                record.session_id,
                call.tool_call_id,
                status=ToolCallStatus.failed,
                error_message=f"{type(exc).__name__}: {exc}",
                traceback_id=traceback_id,
            )
            raise
        await self._update_tool_call(
            record.session_id,
            call.tool_call_id,
            status=ToolCallStatus.succeeded,
            result=_json_value(result) if result_encoder is None else result_encoder(result),
        )
        return result

    async def _approved_tool(
        self,
        record: AgentSessionRecord,
        *,
        tool_name: str,
        arguments: dict[str, F8JsonValue],
        target_graph_revision: int,
        operation: Callable[[], Awaitable[T]],
        result_encoder: Callable[[T], F8JsonValue] | None = None,
    ) -> T:
        await self._check_model_tool_budget(record)
        arguments_hash = _arguments_hash(arguments)
        timestamp = utc_now_text()
        call = AgentToolCall(
            tool_call_id=uuid4().hex,
            tool_name=tool_name,
            arguments=arguments,
            arguments_hash=arguments_hash,
            target_graph_revision=target_graph_revision,
            status=ToolCallStatus.waiting_for_approval,
            created_at=timestamp,
            updated_at=timestamp,
        )
        approval = AgentApproval(
            approval_id=uuid4().hex,
            tool_call_id=call.tool_call_id,
            tool_name=tool_name,
            arguments_hash=arguments_hash,
            target_graph_revision=target_graph_revision,
            expires_at=_future_timestamp(_APPROVAL_TTL),
            status=ApprovalStatus.pending,
        )
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        async with self._lock:
            latest = await asyncio.to_thread(self.get, record.session_id)
            if latest.approval is not None and latest.approval.status is ApprovalStatus.pending:
                raise ValueError("Another tool is awaiting approval in this session; wait for it to finish")
            waiting = msgspec.structs.replace(
                latest,
                status=AgentRunStatus.waiting_for_approval,
                updated_at=timestamp,
                tool_calls=latest.tool_calls + (call,),
                approval=approval,
            )
            await asyncio.to_thread(self._repository.save, waiting)
            self._approvals[approval.approval_id] = _PendingApproval(session_id=record.session_id, future=future)
        await self._publish(waiting)
        try:
            approved = await asyncio.wait_for(future, timeout=_APPROVAL_TTL.total_seconds())
        except TimeoutError:
            await self._expire_approval(record.session_id, approval.approval_id, call.tool_call_id)
            raise TimeoutError(f"approval timed out for tool {tool_name}") from None
        finally:
            self._approvals.pop(approval.approval_id, None)
        if not approved:
            await self._update_tool_call(record.session_id, call.tool_call_id, status=ToolCallStatus.denied)
            raise ApprovalDeniedError(f"approval denied for tool {tool_name}")
        await self._update_tool_call(record.session_id, call.tool_call_id, status=ToolCallStatus.running)
        try:
            result = await operation()
        except Exception as exc:
            traceback_id = uuid4().hex
            logger.exception(
                "approved agent tool failed session_id=%s tool_call_id=%s tool=%s traceback_id=%s",
                record.session_id,
                call.tool_call_id,
                tool_name,
                traceback_id,
            )
            await self._update_tool_call(
                record.session_id,
                call.tool_call_id,
                status=ToolCallStatus.failed,
                error_message=f"{type(exc).__name__}: {exc}",
                traceback_id=traceback_id,
            )
            raise
        await self._update_tool_call(
            record.session_id,
            call.tool_call_id,
            status=ToolCallStatus.succeeded,
            result=_json_value(result) if result_encoder is None else result_encoder(result),
        )
        return result

    async def _append_tool_call(self, session_id: str, call: AgentToolCall) -> None:
        async with self._lock:
            record = await asyncio.to_thread(self.get, session_id)
            updated = msgspec.structs.replace(
                record,
                tool_calls=record.tool_calls + (call,),
                updated_at=utc_now_text(),
            )
            await asyncio.to_thread(self._repository.save, updated)
            await self._publish(updated)

    async def _update_tool_call(
        self,
        session_id: str,
        tool_call_id: str,
        *,
        status: ToolCallStatus,
        result: F8JsonValue = None,
        error_message: str = "",
        traceback_id: str = "",
    ) -> None:
        async with self._lock:
            record = await asyncio.to_thread(self.get, session_id)
            calls: list[AgentToolCall] = []
            found = False
            for call in record.tool_calls:
                if call.tool_call_id != tool_call_id:
                    calls.append(call)
                    continue
                found = True
                calls.append(
                    msgspec.structs.replace(
                        call,
                        status=status,
                        updated_at=utc_now_text(),
                        result=result,
                        error_message=error_message,
                        traceback_id=traceback_id,
                    )
                )
            if not found:
                raise FileNotFoundError(f"agent tool call not found: {tool_call_id}")
            updated = msgspec.structs.replace(
                record,
                status=AgentRunStatus.running if status is ToolCallStatus.running else record.status,
                tool_calls=tuple(calls),
                updated_at=utc_now_text(),
            )
            await asyncio.to_thread(self._repository.save, updated)
            await self._publish(updated)

    async def _append_artifact(self, session_id: str, artifact: AgentArtifact) -> None:
        async with self._lock:
            record = await asyncio.to_thread(self.get, session_id)
            updated = msgspec.structs.replace(
                record,
                artifacts=record.artifacts + (artifact,),
                updated_at=utc_now_text(),
            )
            await asyncio.to_thread(self._repository.save, updated)
            await self._publish(updated)

    async def _expire_approval(self, session_id: str, approval_id: str, tool_call_id: str) -> None:
        record = await asyncio.to_thread(self.get, session_id)
        if record.approval is not None and record.approval.approval_id == approval_id:
            record = self._resolve_record_approval(record, ApprovalStatus.expired)
            await asyncio.to_thread(self._repository.save, record)
        await self._update_tool_call(
            session_id,
            tool_call_id,
            status=ToolCallStatus.failed,
            error_message="agent approval expired",
        )

    async def _finish_stopped(
        self,
        session_id: str,
        status: AgentRunStatus,
        error_message: str,
        *,
        traceback_id: str = "",
    ) -> None:
        record = await asyncio.to_thread(self.get, session_id)
        stopped = msgspec.structs.replace(
            record,
            status=status,
            updated_at=utc_now_text(),
            error_message=error_message,
            traceback_id=traceback_id,
        )
        await asyncio.to_thread(self._repository.save, stopped)
        await self._publish(stopped)

    async def _publish(self, record: AgentSessionRecord) -> None:
        approval_id = record.approval.approval_id if record.approval is not None else None
        await self._events.publish(
            event_type="agent.session.updated",
            scope=f"project:{record.project_id}",
            payload={
                "sessionId": record.session_id,
                "status": record.status.value,
                "approvalId": approval_id,
                "updatedAt": record.updated_at,
            },
        )

    @staticmethod
    def _resolve_record_approval(record: AgentSessionRecord, status: ApprovalStatus) -> AgentSessionRecord:
        approval = record.approval
        if approval is None:
            raise ValueError("agent session has no approval to resolve")
        return msgspec.structs.replace(
            record,
            status=AgentRunStatus.running if status is ApprovalStatus.approved else record.status,
            updated_at=utc_now_text(),
            approval=msgspec.structs.replace(approval, status=status, resolved_at=utc_now_text()),
        )

    @staticmethod
    def _append_message(
        record: AgentSessionRecord,
        *,
        role: Literal["user", "assistant", "system"],
        content: str,
    ) -> AgentSessionRecord:
        return msgspec.structs.replace(
            record,
            messages=record.messages
            + (
                AgentMessage(
                    message_id=uuid4().hex,
                    role=role,
                    content=content,
                    created_at=utc_now_text(),
                    provider_id=record.provider_id,
                    model_id=record.model_id,
                ),
            ),
        )

    @staticmethod
    def _evidence_prompt(record: AgentSessionRecord) -> str:
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

    def _mark_interrupted_sessions(self) -> None:
        for record in self._repository.interrupted():
            timestamp = utc_now_text()
            tool_calls = tuple(
                msgspec.structs.replace(
                    call,
                    status=ToolCallStatus.failed,
                    updated_at=timestamp,
                    error_message="tool interrupted by server restart",
                )
                if call.status in {
                    ToolCallStatus.queued,
                    ToolCallStatus.running,
                    ToolCallStatus.waiting_for_approval,
                }
                else call
                for call in record.tool_calls
            )
            approval = record.approval
            if approval is not None and approval.status is ApprovalStatus.pending:
                approval = msgspec.structs.replace(
                    approval,
                    status=ApprovalStatus.invalidated,
                    resolved_at=timestamp,
                )
            interrupted = msgspec.structs.replace(
                record,
                status=AgentRunStatus.failed,
                updated_at=timestamp,
                tool_calls=tool_calls,
                approval=approval,
                error_message="agent run interrupted by server restart",
                traceback_id="",
            )
            self._repository.save(interrupted)


__all__ = ["AgentService"]
