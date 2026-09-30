from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal
from uuid import uuid4

import msgspec
from f8pysdk.specs import F8JsonValue

from f8studio_server.errors import InvalidRequestError, NotFoundError

from ..events import EventJournal
from ..project_repository import utc_now_text
from .models import (
    AgentArtifact,
    AgentMessage,
    AgentRunStatus,
    AgentSessionRecord,
    AgentToolCall,
    ApprovalStatus,
    ToolCallStatus,
)
from .repository import AgentRepository


class AgentSessions:
    """Own persisted records, per-session locks and session event publication."""

    def __init__(self, database_path: Path, events: EventJournal) -> None:
        self.repository = AgentRepository(database_path)
        self._events = events
        self._locks: dict[str, asyncio.Lock] = {}
        self._mark_interrupted_sessions()

    def get(self, session_id: str) -> AgentSessionRecord:
        record = self.repository.get(session_id)
        if record is None:
            raise NotFoundError(f"agent session not found: {session_id}")
        return record

    def lock(self, session_id: str) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    async def append_tool_call(self, session_id: str, call: AgentToolCall) -> None:
        async with self.lock(session_id):
            record = await asyncio.to_thread(self.get, session_id)
            updated = msgspec.structs.replace(
                record,
                tool_calls=record.tool_calls + (call,),
                updated_at=utc_now_text(),
            )
            await asyncio.to_thread(self.repository.save, updated)
            await self.publish(updated)

    async def update_tool_call(
        self,
        session_id: str,
        tool_call_id: str,
        *,
        status: ToolCallStatus,
        result: F8JsonValue = None,
        error_message: str = "",
        traceback_id: str = "",
    ) -> None:
        async with self.lock(session_id):
            record = await asyncio.to_thread(self.get, session_id)
            if record.status in {AgentRunStatus.succeeded, AgentRunStatus.failed, AgentRunStatus.cancelled}:
                return
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
                raise NotFoundError(f"agent tool call not found: {tool_call_id}")
            updated = msgspec.structs.replace(
                record,
                status=AgentRunStatus.running if status is ToolCallStatus.running else record.status,
                tool_calls=tuple(calls),
                updated_at=utc_now_text(),
            )
            await asyncio.to_thread(self.repository.save, updated)
            await self.publish(updated)

    async def append_artifact(self, session_id: str, artifact: AgentArtifact) -> None:
        async with self.lock(session_id):
            record = await asyncio.to_thread(self.get, session_id)
            updated = msgspec.structs.replace(
                record,
                artifacts=record.artifacts + (artifact,),
                updated_at=utc_now_text(),
            )
            await asyncio.to_thread(self.repository.save, updated)
            await self.publish(updated)

    async def publish(self, record: AgentSessionRecord) -> None:
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
    def resolve_record_approval(record: AgentSessionRecord, status: ApprovalStatus) -> AgentSessionRecord:
        approval = record.approval
        if approval is None:
            raise InvalidRequestError("agent session has no approval to resolve")
        return msgspec.structs.replace(
            record,
            status=AgentRunStatus.running if record.status is AgentRunStatus.waiting_for_approval else record.status,
            updated_at=utc_now_text(),
            approval=msgspec.structs.replace(approval, status=status, resolved_at=utc_now_text()),
        )

    @staticmethod
    def append_message(
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

    def _mark_interrupted_sessions(self) -> None:
        for record in self.repository.interrupted():
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
            self.repository.save(interrupted)
