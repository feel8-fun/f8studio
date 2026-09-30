from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from f8studio_core.graph import PatchResult, ServiceNode, import_graph

from .agents import AgentService
from .agents.models import AgentRunStatus
from .assets import AssetRepository
from .catalog import CatalogService
from .errors import ConflictError, InvalidRequestError, NotFoundError, ServiceUnavailableError
from .events import EventJournal
from .jobs import DeployCoordinator
from .local_integration import LocalIntegrationService
from .models import DeployJob, DeployProjectRequest, ProjectRecord, RuntimeActionResult
from .processes import ManagedProcessResult, ManagedServiceProcesses
from .projects import ProjectService
from .project_commits import ProjectCommits
from .runtime import RuntimeGateway
from .studio_runtime.identifiers import STUDIO_SERVICE_ID

logger = logging.getLogger(__name__)



class ProjectLifecycle:
    """Project lifecycle operations shared by HTTP and in-process automation."""

    def __init__(self, *, projects: ProjectService, jobs: DeployCoordinator,
                 agents: AgentService, processes: ManagedServiceProcesses,
                 local: LocalIntegrationService, events: EventJournal,
                 assets: AssetRepository, catalog: CatalogService, runtime: RuntimeGateway,
                 commits: ProjectCommits) -> None:
        self._projects = projects
        self._jobs = jobs
        self._agents = agents
        self._processes = processes
        self._local = local
        self._events = events
        self._assets = assets
        self._catalog = catalog
        self._runtime = runtime
        self._commits = commits

    async def delete(self, project_id: str) -> None:
        async with self._commits.lock(project_id):
            await asyncio.to_thread(self._projects.summary, project_id)
            if await self._jobs.has_active_project_job(project_id):
                raise ConflictError("Cancel the active deployment before deleting this project")
            sessions = await asyncio.to_thread(self._agents.list, project_id)
            if any(session.status in {AgentRunStatus.running, AgentRunStatus.waiting_for_approval} for session in sessions):
                raise ConflictError("Cancel active agent runs before deleting this project")
            service_ids = await asyncio.to_thread(self._projects.service_ids, project_id)
            if any(self._processes.is_running(service_id) for service_id in service_ids):
                raise ConflictError("Stop the project's services before deleting it")
            await asyncio.to_thread(self._projects.delete, project_id)
            self._local.forget_project_hotkeys(project_id)
            await self._events.publish(
                event_type="project.deleted", scope=f"project:{project_id}", payload={"projectId": project_id},
            )

    async def import_graph(self, project_id: str, payload: bytes, *,
                           expected_graph_revision: int | None = None,
                           expected_layout_revision: int | None = None) -> ProjectRecord:
        async with self._commits.lock(project_id):
            try:
                document = await asyncio.to_thread(import_graph, payload, project_id=project_id)
            except ValueError as exc:
                raise InvalidRequestError(str(exc)) from exc
            record = await asyncio.to_thread(self._projects.restore, project_id, document,
                                             expected_graph_revision=expected_graph_revision,
                                             expected_layout_revision=expected_layout_revision)
            await self._commits.publish(project_id, PatchResult(
                request_id="graph-import", graph_changed=True, layout_changed=True, document=record.document,
            ))
            return record

    async def restore_version(self, project_id: str, version_id: str) -> ProjectRecord:
        async with self._commits.lock(project_id):
            version = await asyncio.to_thread(self._assets.get_project_version, project_id, version_id)
            record = await asyncio.to_thread(self._projects.restore, project_id, version.document)
            await self._commits.publish(project_id, PatchResult(
                request_id=f"restore:{version_id}", graph_changed=True, layout_changed=True, document=record.document,
            ))
            return record

    async def stop_service(self, service_id: str) -> RuntimeActionResult | ManagedProcessResult:
        if service_id == STUDIO_SERVICE_ID:
            return await self._runtime.terminate(service_id)
        try:
            await self._runtime.terminate(service_id)
        except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
            logger.info(
                "runtime terminate unavailable; stopping managed process service_id=%s",
                service_id,
                exc_info=exc,
            )
        return await self._processes.stop(service_id)

    async def stop(self, project_id: str) -> None:
        async with self._commits.lock(project_id):
            document = await asyncio.to_thread(self._projects.document, project_id)
            await self._jobs.cancel_project(project_id)
            service_ids = {node.service_id for node in document.nodes if isinstance(node, ServiceNode)}
            for service_id in sorted(service_ids, key=lambda item: (item == STUDIO_SERVICE_ID, item)):
                await self.stop_service(service_id)

    async def restart_service(self, project_id: str, service_id: str) -> DeployJob:
        async with self._commits.lock(project_id):
            document = await asyncio.to_thread(self._projects.document, project_id)
            service = next((node for node in document.nodes if isinstance(node, ServiceNode) and node.service_id == service_id), None)
            if service is None:
                raise NotFoundError(f"Service {service_id} is not in project {project_id}")
            if service_id == STUDIO_SERVICE_ID or not self._processes.can_start(service.service_class):
                raise ConflictError(f"Service {service_id} cannot be restarted by Studio")
            if not self._processes.is_running(service_id):
                raise ConflictError(f"Service {service_id} is not a running Studio-managed process")
            await asyncio.to_thread(self._catalog.refresh, force_dynamic_service_classes=(service.service_class,))
            if not self._processes.can_start(service.service_class):
                raise ConflictError(f"Service {service.service_class} is unavailable after catalog refresh")
            await self._jobs.cancel_project(project_id)
            await self.stop_service(service_id)
            started = await self._processes.start(service_id, service_class=service.service_class)
            if not started.running:
                raise ServiceUnavailableError(f"Service {service_id} exited during startup")
            return await self._jobs.submit(
                project_id,
                DeployProjectRequest(request_id=uuid4().hex, expected_graph_revision=document.graph_revision),
            )
