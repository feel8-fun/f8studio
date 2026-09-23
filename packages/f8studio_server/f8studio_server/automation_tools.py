from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast

import msgspec

from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import GraphStore, HistoryRequest, PatchRequest, PatchResult, StudioDocument

from .catalog import CatalogService, CatalogSnapshot
from .events import EventJournal
from .jobs import DeployCoordinator
from .models import DeployJob, DeployProjectRequest, ProjectRecord
from .monitors import RuntimeMonitorStore
from .projects import ProjectMutationResult, ProjectService


def _json_value(value: object) -> F8JsonValue:
    return cast(F8JsonValue, msgspec.to_builtins(value, str_keys=True))


class StudioAutomationTools:
    """Single application-service boundary used by HTTP, agents, CLI, and MCP."""

    def __init__(
        self,
        *,
        catalog: CatalogService,
        projects: ProjectService,
        jobs: DeployCoordinator,
        monitors: RuntimeMonitorStore,
        events: EventJournal,
        refresh_hotkeys: Callable[[], None],
    ) -> None:
        self._catalog = catalog
        self._projects = projects
        self._jobs = jobs
        self._monitors = monitors
        self._events = events
        self._refresh_hotkeys = refresh_hotkeys

    def catalog(self) -> CatalogSnapshot:
        return self._catalog.snapshot()

    def project(self, project_id: str) -> ProjectRecord:
        return self._projects.get(project_id)

    def document(self, project_id: str) -> StudioDocument:
        return self._projects.document(project_id)

    def preview_patch(self, project_id: str, request: PatchRequest) -> PatchResult:
        document = self._projects.document(project_id)
        return GraphStore(document).apply(request)

    async def apply_patch(self, project_id: str, request: PatchRequest) -> PatchResult:
        mutation = await asyncio.to_thread(self._projects.patch, project_id, request)
        return await self._publish_mutation(project_id, mutation)

    async def undo(self, project_id: str, request: HistoryRequest) -> PatchResult:
        mutation = await asyncio.to_thread(self._projects.undo, project_id, request)
        return await self._publish_mutation(project_id, mutation)

    async def redo(self, project_id: str, request: HistoryRequest) -> PatchResult:
        mutation = await asyncio.to_thread(self._projects.redo, project_id, request)
        return await self._publish_mutation(project_id, mutation)

    def validate_document(self, document: StudioDocument) -> None:
        self._projects.validate(document)

    async def deploy(self, project_id: str, request: DeployProjectRequest) -> DeployJob:
        return await self._jobs.submit(project_id, request)

    async def deployment(self, job_id: str) -> DeployJob:
        return await self._jobs.get(job_id)

    async def monitor_snapshot(self, project_id: str | None = None) -> F8JsonValue:
        snapshots = await self._monitors.snapshot()
        if project_id is not None:
            document = await asyncio.to_thread(self._projects.document, project_id)
            service_ids = {node.service_id for node in document.nodes}
            snapshots = tuple(snapshot for snapshot in snapshots if str(snapshot.serviceId) in service_ids)
        return _json_value(snapshots)

    async def _publish_mutation(self, project_id: str, mutation: ProjectMutationResult) -> PatchResult:
        result = mutation.result
        if mutation.replayed:
            return result
        await asyncio.to_thread(self._refresh_hotkeys)
        await self._events.publish(
            event_type="graph.committed",
            scope=f"project:{project_id}",
            payload=_json_value(
                {
                    "requestId": result.request_id,
                    "graphChanged": result.graph_changed,
                    "layoutChanged": result.layout_changed,
                    "document": result.document,
                }
            ),
        )
        return result


__all__ = ["StudioAutomationTools"]
