"""One-shot extension tasks, isolated from the Studio process and graph runtime."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import math
import os
from pathlib import Path
import traceback
from typing import Literal
from uuid import uuid4

import msgspec

from f8pysdk.codec import copy_model
from f8pysdk.extension_spec import ExtensionToolField
from f8pysdk.specs import F8JsonValue

from .errors import ConflictError, InvalidRequestError, NotFoundError
from .extensions import ExtensionManager

logger = logging.getLogger(__name__)
_OUTPUT_LIMIT = 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ToolView(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    extension_id: str
    tool_id: str
    name: str
    description: str
    fields: tuple[ExtensionToolField, ...]
    requires_confirmation: bool


class ToolRunRequest(msgspec.Struct, frozen=True, kw_only=True, rename='camel', forbid_unknown_fields=True):
    arguments: dict[str, F8JsonValue]
    confirm: bool = False


class ToolResult(msgspec.Struct, frozen=True, kw_only=True, rename='camel', forbid_unknown_fields=True):
    schema_version: Literal['f8toolResult/1']
    success: bool
    message: str
    data: F8JsonValue = None


class ToolJob(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    job_id: str
    extension_id: str
    extension_version: str
    tool_id: str
    arguments: dict[str, F8JsonValue]
    status: Literal['queued', 'running', 'succeeded', 'failed', 'cancelled']
    created_at: str
    updated_at: str
    result: ToolResult | None = None
    error: str = ''
    log: str = ''


class CapabilityResource(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    extension_id: str
    resource_id: str
    description: str


class ResourceContent(msgspec.Struct, frozen=True, kw_only=True, rename='camel'):
    extension_id: str
    resource_id: str
    content: str


class ExtensionTools:
    def __init__(self, manager: ExtensionManager, root: Path) -> None:
        self._closing = False
        self.manager = manager
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, ToolJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        for path in sorted(root.glob('*.json')):
            job = msgspec.json.decode(path.read_bytes(), type=ToolJob)
            if job.status in {'queued', 'running'}:
                job = copy_model(job, update={'status': 'failed', 'error': 'Studio stopped before the task completed', 'updated_at': _now()})
            self._save(job)
        manager.set_tool_running_probe(self.running)

    def running(self, extension_id: str) -> bool:
        return any(job.extension_id == extension_id and (job.status in {'queued', 'running'}
                   or (job.job_id in self._tasks and not self._tasks[job.job_id].done())) for job in self._jobs.values())

    def list(self) -> tuple[ToolView, ...]:
        return tuple(ToolView(extension_id=manifest.extension_id, tool_id=tool.tool_id, name=tool.name,
                              description=tool.description, fields=tool.fields, requires_confirmation=tool.requires_confirmation)
                     for manifest in self.manager.active_manifests() for tool in manifest.tools)

    def resources(self) -> tuple[CapabilityResource, ...]:
        return tuple(CapabilityResource(extension_id=manifest.extension_id, resource_id=item.resource_id,
                                         description=item.description)
                     for manifest in self.manager.active_manifests() for item in manifest.resources)

    def resource_path(self, extension_id: str, resource_id: str) -> Path:
        manifest = next((item for item in self.manager.active_manifests() if item.extension_id == extension_id), None)
        resource = next((item for item in manifest.resources if item.resource_id == resource_id), None) if manifest else None
        if resource is None:
            raise NotFoundError('Extension resource is unavailable')
        return self.manager.capability_file(extension_id, resource.path)

    def read_resource(self, extension_id: str, resource_id: str) -> ResourceContent:
        path = self.resource_path(extension_id, resource_id)
        if path.stat().st_size > _OUTPUT_LIMIT:
            raise InvalidRequestError('Resource is too large for text access')
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeError as exc:
            raise InvalidRequestError('Resource is not UTF-8 text') from exc
        return ResourceContent(extension_id=extension_id, resource_id=resource_id, content=content)

    def jobs(self) -> tuple[ToolJob, ...]:
        return tuple(sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True))

    def get(self, job_id: str) -> ToolJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError(f'Unknown tool job: {job_id}')
        return job

    def _save(self, job: ToolJob) -> None:
        # Persist only validated IDs generated by Studio, never caller-supplied paths.
        if len(job.job_id) != 32 or any(char not in '0123456789abcdef' for char in job.job_id):
            raise ValueError('Invalid persisted tool job ID')
        target = self.root / f'{job.job_id}.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_bytes(msgspec.json.encode(job))
        temporary.replace(target)
        self._jobs[job.job_id] = job

    def submit(self, extension_id: str, tool_id: str, request: ToolRunRequest) -> ToolJob:
        if self._closing:
            raise InvalidRequestError('Studio is closing; tool submissions are stopped')
        tool, command, cwd, env = self.manager.tool_launcher(extension_id, tool_id)
        if tool.requires_confirmation and not request.confirm:
            raise InvalidRequestError('Explicit confirmation is required to execute this tool')
        if self.running(extension_id):
            raise ConflictError('This extension already has a running tool')
        fields = {field.name: field for field in tool.fields}
        if set(request.arguments) - fields.keys():
            raise InvalidRequestError('Unknown tool arguments')
        arguments: dict[str, F8JsonValue] = {}
        for field in tool.fields:
            value = request.arguments.get(field.name, field.default)
            if value is None or (field.required and value == ''):
                if field.required:
                    raise InvalidRequestError(f'Missing required argument: {field.name}')
                continue
            valid = ((field.kind == 'string' and isinstance(value, str) and len(value) <= 8192)
                     or (field.kind == 'boolean' and isinstance(value, bool))
                     or (field.kind == 'integer' and isinstance(value, int) and not isinstance(value, bool))
                     or (field.kind == 'number' and isinstance(value, (int, float)) and not isinstance(value, bool)
                         and math.isfinite(value)))
            if not valid or (field.choices and value not in field.choices):
                raise InvalidRequestError(f'Invalid argument: {field.name}')
            arguments[field.name] = value
        if len(msgspec.json.encode(arguments)) > 128 * 1024:
            raise InvalidRequestError('Tool arguments exceed the size limit')
        now = _now()
        job = ToolJob(job_id=uuid4().hex, extension_id=extension_id,
                      extension_version=self.manager.status(extension_id).version, tool_id=tool_id,
                      arguments=arguments, status='queued', created_at=now, updated_at=now)
        self._save(job)
        task = asyncio.create_task(self._run(job, command, cwd, env, tool.timeout_seconds))
        self._tasks[job.job_id] = task
        task.add_done_callback(self._task_finished)
        return job

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        for job_id, pending in tuple(self._tasks.items()):
            if pending is task:
                del self._tasks[job_id]
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error('Extension task failed outside the execution boundary',
                             exc_info=(type(error), error, error.__traceback__))

    @staticmethod
    async def _read(stream: asyncio.StreamReader | None) -> bytes:
        if stream is None:
            raise RuntimeError('Tool output pipe is unavailable')
        result = bytearray()
        while chunk := await stream.read(65536):
            result.extend(chunk)
            if len(result) > _OUTPUT_LIMIT:
                raise ValueError('Tool output exceeds the size limit')
        return bytes(result)

    async def _run(self, job: ToolJob, command: list[str], cwd: Path, env: dict[str, str], timeout: int) -> None:
        process: asyncio.subprocess.Process | None = None
        io_tasks: list[asyncio.Task[bytes]] = []
        try:
            self._save(copy_model(job, update={'status': 'running', 'updated_at': _now()}))
            async with asyncio.timeout(timeout):
                process = await asyncio.create_subprocess_exec(*command, cwd=cwd, env={**os.environ, **env},
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
                io_tasks = [asyncio.create_task(self._read(process.stdout)), asyncio.create_task(self._read(process.stderr))]
                if process.stdin is None:
                    raise RuntimeError('Tool input pipe is unavailable')
                process.stdin.write(msgspec.json.encode({'schemaVersion': 'f8toolInput/1', 'arguments': job.arguments}) + b'\n')
                await process.stdin.drain()
                process.stdin.close()
                stdout, stderr = await asyncio.gather(*io_tasks)
                await process.wait()
                log = stderr.decode('utf-8', errors='replace')[-65536:]
                if process.returncode != 0:
                    raise RuntimeError(f'Tool exited with code {process.returncode}: {log}')
                result = msgspec.json.decode(stdout, type=ToolResult)
                self._save(copy_model(job, update={'status': 'succeeded' if result.success else 'failed',
                    'result': result, 'error': '' if result.success else result.message, 'log': log, 'updated_at': _now()}))
        except asyncio.CancelledError:
            self._save(copy_model(job, update={'status': 'cancelled', 'updated_at': _now()}))
            raise
        except Exception as exc:
            # Background execution boundary: preserve the traceback and job context.
            logger.exception('Extension tool failed: %s/%s job=%s', job.extension_id, job.tool_id, job.job_id)
            self._save(copy_model(job, update={'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                'log': traceback.format_exc()[-65536:], 'updated_at': _now()}))
        finally:
            for task in io_tasks:
                if not task.done():
                    task.cancel()
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            if io_tasks:
                await asyncio.gather(*io_tasks, return_exceptions=True)

    async def cancel(self, job_id: str) -> ToolJob:
        self.get(job_id)
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self.get(job_id).status in {'queued', 'running'}:
            self._save(copy_model(self.get(job_id), update={'status': 'cancelled', 'updated_at': _now()}))
        return self.get(job_id)

    async def close(self) -> None:
        self._closing = True
        for job_id in tuple(self._tasks):
            await self.cancel(job_id)
