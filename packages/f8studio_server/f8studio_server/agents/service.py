from __future__ import annotations

import asyncio
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

from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import (
    CreateNodeOp,
    NodeCatalog,
    NodeLayout,
    OperatorNode,
    PatchRequest,
    PatchResult,
    RevisionConflictError,
    StudioDocument,
)

from ..automation_tools import StudioAutomationTools
from ..catalog import CatalogSnapshot
from ..events import EventJournal
from ..models import DeployJob, DeployProjectRequest, JobStatus
from ..project_repository import utc_now_text
from .models import (
    AgentApproval,
    AgentArtifact,
    AgentMessage,
    AgentProviderSummary,
    AgentRunStatus,
    AgentSessionRecord,
    AgentSessionSummary,
    AgentToolCall,
    ApprovalStatus,
    CreateAgentSessionRequest,
    ResolveAgentApprovalRequest,
    StartAgentRunRequest,
    ToolCallStatus,
)
from .providers import AgentProviderRegistry
from .repository import AgentRepository


logger = logging.getLogger(__name__)
T = TypeVar("T")
_TERMINAL_JOBS = {JobStatus.succeeded, JobStatus.partially_failed, JobStatus.failed, JobStatus.cancelled}
_APPROVAL_TTL = timedelta(minutes=5)


class ApprovalDeniedError(RuntimeError):
    pass


@dataclass(frozen=True)
class _PendingApproval:
    session_id: str
    future: asyncio.Future[bool]


def _json_value(value: object) -> F8JsonValue:
    return cast(F8JsonValue, msgspec.to_builtins(value, str_keys=True))


def _arguments_hash(arguments: dict[str, F8JsonValue]) -> str:
    encoded = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _future_timestamp(delta: timedelta) -> str:
    return (datetime.now(UTC) + delta).isoformat(timespec="milliseconds")


def _catalog_evidence(catalog: CatalogSnapshot) -> F8JsonValue:
    return {"serviceCount": len(catalog.services), "operatorCount": len(catalog.operators)}


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
        events: EventJournal,
        providers: AgentProviderRegistry | None = None,
    ) -> None:
        self._repository = AgentRepository(database_path)
        self._tools = tools
        self._events = events
        self._providers = providers or AgentProviderRegistry()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._approvals: dict[str, _PendingApproval] = {}
        self._lock = asyncio.Lock()
        self._mark_interrupted_sessions()

    def providers(self) -> tuple[AgentProviderSummary, ...]:
        return self._providers.summaries()

    def create(self, request: CreateAgentSessionRequest) -> AgentSessionRecord:
        self._providers.validate_selection(request.provider_id, request.model_id)
        self._tools.project(request.project_id)
        timestamp = utc_now_text()
        record = AgentSessionRecord(
            session_id=uuid4().hex,
            project_id=request.project_id,
            title=request.title.strip() or "New agent session",
            provider_id=request.provider_id,
            model_id=request.model_id,
            status=AgentRunStatus.idle,
            created_at=timestamp,
            updated_at=timestamp,
        )
        return self._repository.save(record)

    def list(self, project_id: str | None = None) -> tuple[AgentSessionSummary, ...]:
        if project_id is not None:
            self._tools.project(project_id)
        return self._repository.list(project_id)

    def get(self, session_id: str) -> AgentSessionRecord:
        record = self._repository.get(session_id)
        if record is None:
            raise FileNotFoundError(f"agent session not found: {session_id}")
        return record

    async def start_run(self, session_id: str, request: StartAgentRunRequest) -> AgentSessionRecord:
        prompt = request.prompt.strip()
        if not prompt:
            raise ValueError("agent prompt must be non-empty")
        async with self._lock:
            record = await asyncio.to_thread(self.get, session_id)
            if record.status in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                raise ValueError("agent session already has an active run")
            self._providers.validate_selection(record.provider_id, record.model_id)
            timestamp = utc_now_text()
            started = msgspec.structs.replace(
                record,
                status=AgentRunStatus.running,
                updated_at=timestamp,
                messages=record.messages
                + (
                    AgentMessage(
                        message_id=uuid4().hex,
                        role="user",
                        content=prompt,
                        created_at=timestamp,
                    ),
                ),
                approval=None,
                error_message="",
                traceback_id="",
            )
            await asyncio.to_thread(self._repository.save, started)
            task = asyncio.create_task(self._run(started.session_id, prompt), name=f"agent:{started.session_id}")
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

    async def _run(self, session_id: str, prompt: str) -> None:
        try:
            record = await asyncio.to_thread(self.get, session_id)
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
            evidence = self._evidence_prompt(finished)
            response = await self._providers.complete(
                provider_id=finished.provider_id,
                model_id=finished.model_id,
                prompt=evidence,
            )
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
