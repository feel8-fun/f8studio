from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TypeVar
from uuid import uuid4

import msgspec
from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import (
    RevisionConflictError,
)

from f8studio_server.errors import InvalidRequestError, NotFoundError

from ..automation_tools import StudioAutomationTools
from ..models import DeployJob, DeployProjectRequest, JobStatus
from ..project_repository import utc_now_text
from .evidence import arguments_hash, json_value
from .models import (
    AgentApproval,
    AgentRunStatus,
    AgentSessionRecord,
    AgentToolCall,
    ApprovalStatus,
    ResolveAgentApprovalRequest,
    ToolCallStatus,
)
from .sessions import AgentSessions

logger = logging.getLogger(__name__)
T = TypeVar("T")
_APPROVAL_TTL = timedelta(minutes=5)
_MAX_MODEL_TOOL_CALLS = 48

class ApprovalDeniedError(RuntimeError):
    pass


@dataclass(frozen=True)
class _PendingApproval:
    session_id: str
    future: asyncio.Future[bool]
    layout_revision: int


def _future_timestamp(delta: timedelta) -> str:
    return (datetime.now(UTC) + delta).isoformat(timespec="milliseconds")


class AgentToolExecution:
    """Audit operations and own approval futures shared by both run strategies."""

    def __init__(self, sessions: AgentSessions, tools: StudioAutomationTools) -> None:
        self._sessions = sessions
        self._tools = tools
        self._approvals: dict[str, _PendingApproval] = {}

    @property
    def pending_count(self) -> int:
        return len(self._approvals)

    def cancel_pending(self, approval_id: str) -> None:
        pending = self._approvals.pop(approval_id, None)
        if pending is not None and not pending.future.done():
            pending.future.cancel()

    async def resolve_approval(
        self,
        session_id: str,
        approval_id: str,
        request: ResolveAgentApprovalRequest,
    ) -> AgentSessionRecord:
        async with self._sessions.lock(session_id):
            record = await asyncio.to_thread(self._sessions.get, session_id)
            approval = record.approval
            if approval is None or approval.approval_id != approval_id:
                raise NotFoundError(f"pending agent approval not found: {approval_id}")
            if approval.status is not ApprovalStatus.pending:
                raise InvalidRequestError(f"agent approval is already {approval.status.value}")
            if request.arguments_hash != approval.arguments_hash:
                raise InvalidRequestError("agent approval argumentsHash does not match the pending tool call")
            pending = self._approvals.get(approval_id)
            if pending is None or pending.session_id != session_id:
                raise InvalidRequestError("agent approval is no longer active")

            now = datetime.now(UTC)
            if now >= datetime.fromisoformat(approval.expires_at):
                updated = self._sessions.resolve_record_approval(record, ApprovalStatus.expired)
                await asyncio.to_thread(self._sessions.repository.save, updated)
                if not pending.future.done():
                    pending.future.set_exception(TimeoutError("agent approval expired"))
                raise InvalidRequestError("agent approval expired")

            current = await asyncio.to_thread(self._tools.document, record.project_id)
            if approval.target_graph_revision is not None and (
                    current.graph_revision != approval.target_graph_revision
                    or current.layout_revision != pending.layout_revision):
                updated = self._sessions.resolve_record_approval(record, ApprovalStatus.invalidated)
                await asyncio.to_thread(self._sessions.repository.save, updated)
                conflict = RevisionConflictError(
                    f"approval revision conflict: expected {approval.target_graph_revision}, "
                    f"current {current.graph_revision}; layout expected {pending.layout_revision}, "
                    f"current {current.layout_revision}"
                )
                if not pending.future.done():
                    pending.future.set_exception(conflict)
                raise conflict

            status = ApprovalStatus.approved if request.approved else ApprovalStatus.denied
            updated = self._sessions.resolve_record_approval(record, status)
            await asyncio.to_thread(self._sessions.repository.save, updated)
            if not pending.future.done():
                pending.future.set_result(request.approved)
        await self._sessions.publish(updated)
        return updated

    async def _check_model_tool_budget(self, record: AgentSessionRecord) -> None:
        if record.provider_id == "deterministic":
            return
        latest = await asyncio.to_thread(self._sessions.get, record.session_id)
        if len(latest.tool_calls) - len(record.tool_calls) >= _MAX_MODEL_TOOL_CALLS:
            raise RuntimeError(f"agent run exceeded {_MAX_MODEL_TOOL_CALLS} tool calls")

    async def run(
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
            arguments_hash=arguments_hash(arguments),
            target_graph_revision=target_graph_revision,
            status=ToolCallStatus.running,
            created_at=utc_now_text(),
            updated_at=utc_now_text(),
        )
        await self._sessions.append_tool_call(record.session_id, call)
        return await self._invoke(record, call, operation, result_encoder)

    async def approved(
        self,
        record: AgentSessionRecord,
        *,
        tool_name: str,
        arguments: dict[str, F8JsonValue],
        target_graph_revision: int | None,
        target_layout_revision: int | None = None,
        operation: Callable[[], Awaitable[T]],
        result_encoder: Callable[[T], F8JsonValue] | None = None,
    ) -> T:
        await self._check_model_tool_budget(record)
        digest = arguments_hash(arguments)
        timestamp = utc_now_text()
        call = AgentToolCall(
            tool_call_id=uuid4().hex,
            tool_name=tool_name,
            arguments=arguments,
            arguments_hash=digest,
            target_graph_revision=target_graph_revision,
            status=ToolCallStatus.waiting_for_approval,
            created_at=timestamp,
            updated_at=timestamp,
        )
        approval = AgentApproval(
            approval_id=uuid4().hex,
            tool_call_id=call.tool_call_id,
            tool_name=tool_name,
            arguments_hash=digest,
            target_graph_revision=target_graph_revision,
            expires_at=_future_timestamp(_APPROVAL_TTL),
            status=ApprovalStatus.pending,
        )
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        async with self._sessions.lock(record.session_id):
            latest = await asyncio.to_thread(self._sessions.get, record.session_id)
            if latest.status in {AgentRunStatus.cancelled, AgentRunStatus.failed, AgentRunStatus.succeeded}:
                raise asyncio.CancelledError("agent run already ended")
            document = await asyncio.to_thread(self._tools.document, record.project_id)
            if target_graph_revision is not None and (document.graph_revision != target_graph_revision or (
                target_layout_revision is not None and document.layout_revision != target_layout_revision
            )):
                raise RevisionConflictError("project changed before approval; preview the proposed change again")
            if latest.approval is not None and latest.approval.status is ApprovalStatus.pending:
                raise InvalidRequestError("Another tool is awaiting approval in this session; wait for it to finish")
            waiting = msgspec.structs.replace(
                latest,
                status=AgentRunStatus.waiting_for_approval,
                updated_at=timestamp,
                tool_calls=latest.tool_calls + (call,),
                approval=approval,
            )
            await asyncio.to_thread(self._sessions.repository.save, waiting)
            self._approvals[approval.approval_id] = _PendingApproval(
                session_id=record.session_id, future=future, layout_revision=document.layout_revision,
            )
        await self._sessions.publish(waiting)
        try:
            approved = await asyncio.wait_for(future, timeout=_APPROVAL_TTL.total_seconds())
        except TimeoutError:
            await self._expire_approval(record.session_id, approval.approval_id, call.tool_call_id)
            raise TimeoutError(f"approval timed out for tool {tool_name}") from None
        finally:
            self._approvals.pop(approval.approval_id, None)
        if not approved:
            await self._sessions.update_tool_call(record.session_id, call.tool_call_id, status=ToolCallStatus.denied)
            raise ApprovalDeniedError(f"approval denied for tool {tool_name}")
        await self._sessions.update_tool_call(record.session_id, call.tool_call_id, status=ToolCallStatus.running)
        return await self._invoke(record, call, operation, result_encoder)

    async def _invoke(self, record: AgentSessionRecord, call: AgentToolCall,
                      operation: Callable[[], Awaitable[T]], result_encoder: Callable[[T], F8JsonValue] | None) -> T:
        try:
            result = await operation()
        except Exception as exc:
            traceback_id = uuid4().hex
            logger.exception(
                "agent tool failed session_id=%s tool_call_id=%s tool=%s traceback_id=%s",
                record.session_id,
                call.tool_call_id,
                call.tool_name,
                traceback_id,
            )
            await self._sessions.update_tool_call(
                record.session_id,
                call.tool_call_id,
                status=ToolCallStatus.failed,
                error_message=f"{type(exc).__name__}: {exc}",
                traceback_id=traceback_id,
            )
            raise
        await self._sessions.update_tool_call(
            record.session_id,
            call.tool_call_id,
            status=ToolCallStatus.succeeded,
            result=json_value(result) if result_encoder is None else result_encoder(result),
        )
        return result

    async def _expire_approval(self, session_id: str, approval_id: str, tool_call_id: str) -> None:
        async with self._sessions.lock(session_id):
            record = await asyncio.to_thread(self._sessions.get, session_id)
            if (record.approval is not None and record.approval.approval_id == approval_id
                    and record.approval.status is ApprovalStatus.pending):
                record = self._sessions.resolve_record_approval(record, ApprovalStatus.expired)
                await asyncio.to_thread(self._sessions.repository.save, record)
        await self._sessions.update_tool_call(
            session_id,
            tool_call_id,
            status=ToolCallStatus.failed,
            error_message="agent approval expired",
        )

    async def finish_stopped(
        self,
        session_id: str,
        status: AgentRunStatus,
        error_message: str,
        *,
        traceback_id: str = "",
    ) -> None:
        async with self._sessions.lock(session_id):
            record = await asyncio.to_thread(self._sessions.get, session_id)
            if record.status not in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                return
            stopped = await self.stop_record(record, status, error_message, traceback_id=traceback_id)
            await self._sessions.publish(stopped)

    async def stop_record(self, record: AgentSessionRecord, status: AgentRunStatus, error_message: str,
                          *, traceback_id: str = "", tool_error_message: str = "") -> AgentSessionRecord:
        """Persist a terminal transition while the caller holds this session's lock."""
        timestamp = utc_now_text()
        approval = record.approval
        if approval is not None and approval.status is ApprovalStatus.pending:
            self.cancel_pending(approval.approval_id)
            approval = msgspec.structs.replace(
                approval, status=ApprovalStatus.cancelled if status is AgentRunStatus.cancelled else ApprovalStatus.expired,
                resolved_at=timestamp,
            )
        calls = tuple(
            msgspec.structs.replace(
                call, status=ToolCallStatus.cancelled if status is AgentRunStatus.cancelled else ToolCallStatus.failed,
                updated_at=timestamp, error_message=tool_error_message or error_message, traceback_id=traceback_id,
            ) if call.status in {ToolCallStatus.queued, ToolCallStatus.running, ToolCallStatus.waiting_for_approval}
            else call for call in record.tool_calls
        )
        stopped = msgspec.structs.replace(
            record, status=status, updated_at=timestamp, approval=approval, tool_calls=calls,
            error_message=error_message, traceback_id=traceback_id,
        )
        await asyncio.to_thread(self._sessions.repository.save, stopped)
        return stopped

    async def deploy_and_wait(self, project_id: str, request: DeployProjectRequest) -> DeployJob:
        job = await self._tools.deploy(project_id, request)
        while job.status not in _TERMINAL_JOBS:
            await asyncio.sleep(0.02)
            job = await self._tools.deployment(job.job_id)
        if job.status is not JobStatus.succeeded:
            detail = job.error_message or "deployment did not succeed"
            raise RuntimeError(f"deployment {job.job_id} finished with {job.status.value}: {detail}")
        return job


_TERMINAL_JOBS = {JobStatus.succeeded, JobStatus.partially_failed, JobStatus.failed, JobStatus.cancelled}
