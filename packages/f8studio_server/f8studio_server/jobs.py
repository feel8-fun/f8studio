from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping
from typing import cast
from uuid import uuid4

import msgspec

from f8studio_core import compile_document, semantic_graph_revision
from f8studio_core.graph import IdempotencyConflictError, RevisionConflictError
from f8studio_core.graph.codec import canonical_json_bytes
from f8pysdk.specs import F8JsonValue, F8RuntimeGraph

from .events import EventJournal
from .job_repository import JobRepository
from .models import DeployJob, DeployProjectRequest, JobStatus, ServiceDeployResult
from .project_repository import utc_now_text
from .projects import ProjectService
from .runtime import RuntimeGateway


logger = logging.getLogger(__name__)


def _request_fingerprint(request: DeployProjectRequest) -> str:
    return hashlib.sha256(canonical_json_bytes(request)).hexdigest()


class DeployCoordinator:
    def __init__(
        self,
        *,
        projects: ProjectService,
        repository: JobRepository,
        runtime: RuntimeGateway,
        events: EventJournal,
    ) -> None:
        self._projects = projects
        self._repository = repository
        self._runtime = runtime
        self._events = events
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()
        self._repository.mark_interrupted_jobs_failed(timestamp=utc_now_text())

    async def submit(self, project_id: str, request: DeployProjectRequest) -> DeployJob:
        fingerprint = _request_fingerprint(request)
        async with self._lock:
            existing = await asyncio.to_thread(self._repository.find_request, project_id, request.request_id)
            if existing is not None:
                stored_fingerprint, job = existing
                if stored_fingerprint != fingerprint:
                    raise IdempotencyConflictError("deploy request id reused with different content")
                return job

            document = await asyncio.to_thread(self._projects.document, project_id)
            if document.graph_revision != request.expected_graph_revision:
                raise RevisionConflictError(
                    f"deploy revision conflict: expected {request.expected_graph_revision}, "
                    f"current {document.graph_revision}"
                )
            compiled = await asyncio.to_thread(compile_document, document)
            semantic_revision = semantic_graph_revision(document)
            timestamp = utc_now_text()
            duplicate = None
            if not request.force_apply:
                duplicate = await asyncio.to_thread(
                    self._repository.find_successful,
                    project_id,
                    semantic_revision,
                )
            job = DeployJob(
                job_id=uuid4().hex,
                request_id=request.request_id,
                project_id=project_id,
                source_graph_revision=document.graph_revision,
                source_semantic_revision=semantic_revision,
                status=JobStatus.succeeded if duplicate is not None else JobStatus.queued,
                created_at=timestamp,
                updated_at=timestamp,
                service_results=() if duplicate is None else duplicate.service_results,
            )
            await asyncio.to_thread(self._repository.create, job, request_fingerprint=fingerprint)
            if duplicate is not None:
                await self._publish_job("deploy.deduplicated", job)
                return job
            task = asyncio.create_task(
                self._run(job, compiled.per_service, force_apply=request.force_apply),
                name=f"deploy:{job.job_id}",
            )
            self._tasks[job.job_id] = task
            task.add_done_callback(lambda _task, job_id=job.job_id: self._tasks.pop(job_id, None))
            await self._publish_job("deploy.queued", job)
            return job

    async def get(self, job_id: str) -> DeployJob:
        job = await asyncio.to_thread(self._repository.get, job_id)
        if job is None:
            raise FileNotFoundError(f"deploy job not found: {job_id}")
        return job

    async def cancel(self, job_id: str) -> DeployJob:
        async with self._lock:
            job = await self.get(job_id)
            if job.status not in {JobStatus.queued, JobStatus.running}:
                return job
            task = self._tasks.get(job_id)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                latest = await self.get(job_id)
                if latest.status not in {JobStatus.queued, JobStatus.running}:
                    return latest
            cancelled = msgspec.structs.replace(
                job,
                status=JobStatus.cancelled,
                updated_at=utc_now_text(),
                error_message="deployment cancelled; already accepted service operations are not rolled back",
            )
            await asyncio.to_thread(self._repository.update, cancelled)
            await self._publish_job("deploy.cancelled", cancelled)
            return cancelled

    async def close(self) -> None:
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def _run(
        self,
        job: DeployJob,
        per_service: Mapping[str, F8RuntimeGraph],
        *,
        force_apply: bool,
    ) -> None:
        running = msgspec.structs.replace(job, status=JobStatus.running, updated_at=utc_now_text())
        await asyncio.to_thread(self._repository.update, running)
        await self._publish_job("deploy.running", running)
        try:
            calls: list[asyncio.Task[ServiceDeployResult]] = []
            for service_id, graph_value in sorted(per_service.items()):
                calls.append(
                    asyncio.create_task(
                        self._deploy_service(service_id, graph_value, force_apply=force_apply),
                        name=f"deploy:{job.job_id}:{service_id}",
                    )
                )
            results = tuple(await asyncio.gather(*calls))
            successes = sum(int(result.success) for result in results)
            if successes == len(results):
                status = JobStatus.succeeded
                error_message = ""
            elif successes:
                status = JobStatus.partially_failed
                error_message = "one or more services rejected the deployment"
            else:
                status = JobStatus.failed
                error_message = "all services rejected the deployment" if results else "document has no enabled services"
            finished = msgspec.structs.replace(
                running,
                status=status,
                updated_at=utc_now_text(),
                service_results=results,
                error_message=error_message,
            )
            await asyncio.to_thread(self._repository.update, finished)
            await self._publish_job("deploy.finished", finished)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("deploy job failed job_id=%s project_id=%s", job.job_id, job.project_id)
            failed = msgspec.structs.replace(
                running,
                status=JobStatus.failed,
                updated_at=utc_now_text(),
                error_message=f"{type(exc).__name__}: {exc}",
            )
            await asyncio.to_thread(self._repository.update, failed)
            await self._publish_job("deploy.finished", failed)

    async def _deploy_service(
        self,
        service_id: str,
        graph: F8RuntimeGraph,
        *,
        force_apply: bool,
    ) -> ServiceDeployResult:
        try:
            return await self._runtime.deploy(
                service_id=service_id,
                graph=graph,
                force_apply=force_apply,
            )
        except Exception as exc:
            logger.exception("service deployment failed service_id=%s", service_id)
            return ServiceDeployResult(
                service_id=service_id,
                success=False,
                error_message=f"{type(exc).__name__}: {exc}",
            )

    async def _publish_job(self, event_type: str, job: DeployJob) -> None:
        payload = cast(dict[str, F8JsonValue], msgspec.to_builtins(job, str_keys=True))
        await self._events.publish(
            event_type=event_type,
            scope=f"project:{job.project_id}",
            payload=payload,
        )


__all__ = ["DeployCoordinator"]
