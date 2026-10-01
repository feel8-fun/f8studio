from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from asyncio import timeout as run_timeout
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal
from uuid import uuid4

import msgspec

from f8studio_server.errors import InvalidRequestError

from ..automation_tools import StudioAutomationTools
from ..editor import EditorSessionService
from ..events import EventJournal
from ..local_integration import (
    LocalIntegrationService,
)
from ..project_repository import utc_now_text
from .evidence import conversation_prompt, evidence_prompt
from .execution import AgentToolExecution, ApprovalDeniedError
from ..extension_tools import ExtensionTools
from .model_tools import AgentModelTools
from .models import (
    AgentImage,
    AgentMessage,
    AgentProviderSummary,
    AgentRunStatus,
    AgentSessionRecord,
    AgentSessionSummary,
    ApprovalStatus,
    CreateAgentSessionRequest,
    RenameAgentSessionRequest,
    ResolveAgentApprovalRequest,
    SelectAgentModelRequest,
    StartAgentRunRequest,
)
from .provider_probe import ProbeProviderRequest, ProviderProbeResult
from .provider_settings import CreateProviderConnection, ProviderSettingsView, UpdateProviderSettings
from .providers import AgentProviderRegistry
from .sessions import AgentSessions
from .skills import AgentSkillLibrary
from .workflows import DeterministicWorkflow

logger = logging.getLogger(__name__)
_RUN_TIMEOUT_S = 300.0
_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_MAX_IMAGES = 3


def _validate_images(images: tuple[AgentImage, ...]) -> None:
    if len(images) > _MAX_IMAGES:
        raise InvalidRequestError(f"at most {_MAX_IMAGES} images are allowed per message")
    signatures = {
        "image/png": b"\x89PNG\r\n\x1a\n",
        "image/jpeg": b"\xff\xd8\xff",
        "image/webp": b"RIFF",
        "image/gif": b"GIF8",
    }
    for image in images:
        if not image.name.strip() or len(image.name) > 200:
            raise InvalidRequestError("image name must be between 1 and 200 characters")
        header, separator, encoded = image.data_url.partition(",")
        media_type = header.removeprefix("data:").removesuffix(";base64")
        if not separator or header != f"data:{media_type};base64" or media_type not in signatures:
            raise InvalidRequestError("image must be a base64 PNG, JPEG, WebP, or GIF data URL")
        if len(encoded) > (_MAX_IMAGE_BYTES + 2) * 4 // 3 + 4:
            raise InvalidRequestError("image exceeds the 4 MB limit")
        try:
            data = base64.b64decode(encoded, validate=True)
        except binascii.Error as exc:
            raise InvalidRequestError("image has invalid base64 data") from exc
        if not data or len(data) > _MAX_IMAGE_BYTES or not data.startswith(signatures[media_type]):
            raise InvalidRequestError("image format does not match its content or exceeds 4 MB")
        if media_type == "image/webp" and data[8:12] != b"WEBP":
            raise InvalidRequestError("image format does not match its content")


def _title_from_prompt(prompt: str) -> str:
    first_line = prompt.strip().splitlines()[0]
    title = " ".join(first_line.split())
    return title[:56].rstrip() + ("..." if len(title) > 56 else "")


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
        extension_tools: ExtensionTools | None = None,
    ) -> None:
        self._sessions = AgentSessions(database_path, events)
        self._execution = AgentToolExecution(self._sessions, tools)
        self._workflow = DeterministicWorkflow(tools, self._execution, self._sessions)
        self._model_tool_factory = AgentModelTools(tools=tools, execution=self._execution, sessions=self._sessions, editor=editor, local=local, skills=skills, events=events, extension_tools=extension_tools)
        self._tools = tools
        self._events = events
        self._providers = providers or AgentProviderRegistry(database_path.with_name("agent-providers.json"))
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    def providers(self) -> tuple[AgentProviderSummary, ...]:
        return self._providers.summaries()

    def provider_settings(self) -> tuple[ProviderSettingsView, ...]:
        return self._providers.settings()

    async def update_provider_settings(self, provider_id: str, request: UpdateProviderSettings) -> ProviderSettingsView:
        async with self._lock:
            if any(not task.done() for task in self._tasks.values()):
                raise InvalidRequestError("Wait for active agent runs to finish or cancel them before changing provider settings")
            return await asyncio.to_thread(self._providers.update_settings, provider_id, request)

    async def create_provider_connection(self, request: CreateProviderConnection) -> ProviderSettingsView:
        async with self._lock:
            if any(not task.done() for task in self._tasks.values()):
                raise InvalidRequestError("Wait for active agent runs before changing provider connections")
            return await asyncio.to_thread(self._providers.create_connection, request)

    async def delete_provider_connection(self, provider_id: str) -> None:
        async with self._lock:
            if any(not task.done() for task in self._tasks.values()):
                raise InvalidRequestError("Wait for active agent runs before changing provider connections")
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
        return self._sessions.repository.save(record)

    def list(self, project_id: str | None = None) -> tuple[AgentSessionSummary, ...]:
        if project_id is not None:
            self._tools.project_summary(project_id)
        return self._sessions.repository.list(project_id)

    def get(self, session_id: str) -> AgentSessionRecord:
        return self._sessions.get(session_id)

    async def rename(self, session_id: str, request: RenameAgentSessionRequest) -> AgentSessionRecord:
        title = " ".join(request.title.split())
        if not title or len(title) > 120:
            raise InvalidRequestError("agent session title must be between 1 and 120 characters")
        async with self._sessions.lock(session_id):
            record = await asyncio.to_thread(self.get, session_id)
            if record.status in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                raise InvalidRequestError("stop the active agent run before renaming its session")
            updated = msgspec.structs.replace(record, title=title, auto_title_pending=False, updated_at=utc_now_text())
            await asyncio.to_thread(self._sessions.repository.save, updated)
        await self._sessions.publish(updated)
        return updated

    async def select_model(self, session_id: str, request: SelectAgentModelRequest) -> AgentSessionRecord:
        async with self._sessions.lock(session_id):
            record = await asyncio.to_thread(self.get, session_id)
            if record.status in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                raise InvalidRequestError("stop the active agent run before changing its model")
            self._providers.validate_selection(request.provider_id, request.model_id)
            updated = msgspec.structs.replace(
                record, provider_id=request.provider_id, model_id=request.model_id,
                updated_at=utc_now_text(),
            )
            await asyncio.to_thread(self._sessions.repository.save, updated)
        await self._sessions.publish(updated)
        return updated

    async def delete(self, session_id: str) -> None:
        async with self._sessions.lock(session_id):
            record = await asyncio.to_thread(self.get, session_id)
            if record.status in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                raise InvalidRequestError("stop the active agent run before deleting its session")
            await asyncio.to_thread(self._sessions.repository.delete, session_id)
        await self._events.publish(
            event_type="agent.session.deleted",
            scope=f"project:{record.project_id}",
            payload={"sessionId": session_id},
        )

    async def start_run(self, session_id: str, request: StartAgentRunRequest) -> AgentSessionRecord:
        prompt = request.prompt.strip()
        if not prompt and not request.images:
            raise InvalidRequestError("agent prompt or image must be provided")
        _validate_images(request.images)
        async with self._lock, self._sessions.lock(session_id):
            record = await asyncio.to_thread(self.get, session_id)
            previous_task = self._tasks.get(session_id)
            if record.status in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval} or (
                previous_task is not None and not previous_task.done()
            ):
                raise InvalidRequestError("agent session already has an active run")
            self._providers.validate_selection(record.provider_id, record.model_id)
            if request.images and not self._providers.supports_image(record.provider_id, record.model_id):
                raise InvalidRequestError(f"selected agent model does not support image input: {record.model_id}")
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
            await asyncio.to_thread(self._sessions.repository.save, started)
            task = asyncio.create_task(self._run(started.session_id, prompt, request.reasoning_effort), name=f"agent:{started.session_id}")
            self._tasks[started.session_id] = task
            task.add_done_callback(lambda finished, key=started.session_id: self._run_finished(key, finished))
        await self._sessions.publish(started)
        return started

    def _run_finished(self, session_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(session_id) is task:
            self._tasks.pop(session_id, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error("agent task failed session_id=%s", session_id, exc_info=task.exception())

    async def resolve_approval(self, session_id: str, approval_id: str, request: ResolveAgentApprovalRequest) -> AgentSessionRecord:
        return await self._execution.resolve_approval(session_id, approval_id, request)

    async def cancel(self, session_id: str) -> AgentSessionRecord:
        async with self._sessions.lock(session_id):
            record = await asyncio.to_thread(self.get, session_id)
            if record.status not in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                return record
            task = self._tasks.get(session_id)
            if task is not None:
                task.cancel()
            cancelled = await self._execution.stop_record(
                record, AgentRunStatus.cancelled,
                "agent run cancelled; completed side effects were not rolled back",
                tool_error_message="agent run cancelled before tool completion",
            )
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await self._sessions.publish(cancelled)
        return cancelled

    async def close(self) -> None:
        session_ids = tuple(self._tasks)
        for session_id in session_ids:
            await self.cancel(session_id)
        self._tasks.clear()

    async def _run(self, session_id: str, prompt: str, reasoning_effort: Literal["low", "medium", "high"] | None = None) -> None:
        try:
            async with run_timeout(_RUN_TIMEOUT_S):
                await self._execute_run(session_id, prompt, reasoning_effort)
        except asyncio.CancelledError:
            await self._execution.finish_stopped(session_id, AgentRunStatus.cancelled, "agent run cancelled")
            raise
        except ApprovalDeniedError as exc:
            await self._execution.finish_stopped(session_id, AgentRunStatus.cancelled, str(exc))
        except Exception as exc:
            traceback_id = uuid4().hex
            logger.exception("agent run failed session_id=%s traceback_id=%s", session_id, traceback_id)
            await self._execution.finish_stopped(
                session_id,
                AgentRunStatus.failed,
                f"{type(exc).__name__}: {exc}",
                traceback_id=traceback_id,
            )

    async def _execute_run(self, session_id: str, prompt: str, reasoning_effort: Literal["low", "medium", "high"] | None = None) -> None:
        record = await asyncio.to_thread(self.get, session_id)
        if record.provider_id != "deterministic":
            supports_images = self._providers.supports_image(record.provider_id, record.model_id)
            images = tuple(image for message in record.messages[-9:] for image in message.images)[-_MAX_IMAGES:] if supports_images else ()
            provider_input = conversation_prompt(record, prompt)
            if images:
                provider_input += "\nAttached images are ordered by their appearance in the conversation."
            provider_run = self._providers.run_with_tools(
                provider_id=record.provider_id, model_id=record.model_id,
                prompt=provider_input, tools=self._model_tools(record),
                images=images, reasoning_effort=reasoning_effort,
            )
            response = await provider_run
        else:
            await self._workflow.run(record, prompt)
            finished = await asyncio.to_thread(self.get, session_id)
            response = await self._providers.complete(
                provider_id=finished.provider_id,
                model_id=finished.model_id,
                prompt=evidence_prompt(finished),
            )
        async with self._sessions.lock(session_id):
            finished = await asyncio.to_thread(self.get, session_id)
            if finished.status not in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval}:
                return
            if finished.approval is not None and finished.approval.status is ApprovalStatus.pending:
                raise RuntimeError("agent provider returned with an unresolved approval")
            completed = self._sessions.append_message(finished, role="assistant", content=response)
            completed = msgspec.structs.replace(
                completed, status=AgentRunStatus.succeeded, updated_at=utc_now_text(),
                error_message="", traceback_id="",
            )
            await asyncio.to_thread(self._sessions.repository.save, completed)
            await self._sessions.publish(completed)

    def _model_tools(self, record: AgentSessionRecord) -> tuple[Callable[..., Awaitable[str]], ...]:
        return self._model_tool_factory.bind(record)


__all__ = ["AgentService"]
