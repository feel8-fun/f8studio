from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import cast

import msgspec
from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import GraphStore, HistoryRequest, PatchRequest, PatchResult, SetNodeStateOp, StudioDocument

from .catalog import CatalogService, CatalogSnapshot
from .events import EventJournal
from .jobs import DeployCoordinator
from .models import DeployJob, DeployProjectRequest, JobStatus, ProjectRecord
from .monitors import RuntimeMonitorStore
from .projects import ProjectMutationResult, ProjectService
from .runtime import RuntimeGateway

logger = logging.getLogger(__name__)


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
        runtime: RuntimeGateway,
    ) -> None:
        self._catalog = catalog
        self._projects = projects
        self._jobs = jobs
        self._monitors = monitors
        self._events = events
        self._refresh_hotkeys = refresh_hotkeys
        self._runtime = runtime

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
        result = await self._publish_mutation(project_id, mutation)
        if mutation.replayed:
            return result
        changes = [operation for operation in request.operations if isinstance(operation, SetNodeStateOp)]
        if not changes:
            return result
        deployment = await self._jobs.latest(project_id)
        if deployment is None or deployment.status not in {JobStatus.succeeded, JobStatus.partially_failed}:
            return result
        deployed_ids = {item.service_id for item in deployment.service_results if item.success}
        nodes = {node.node_id: node for node in result.document.nodes}
        errors: list[str] = []
        for change in changes:
            node = nodes.get(change.node_id)
            if node is None or node.service_id not in deployed_ids:
                continue
            try:
                response = await self._runtime.set_state(
                    node.service_id, node_id=node.node_id, field=change.field, value=change.value,
                )
                if not response.success:
                    errors.append(f"{node.node_id}.{change.field}: {response.error_message or 'runtime rejected state'}")
            except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
                logger.exception("runtime state sync failed project_id=%s node_id=%s field=%s", project_id, node.node_id, change.field)
                errors.append(f"{node.node_id}.{change.field}: {type(exc).__name__}: {exc}")
        return msgspec.structs.replace(result, runtime_errors=tuple(errors)) if errors else result

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
