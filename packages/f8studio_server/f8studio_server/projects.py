from __future__ import annotations

import hashlib
from dataclasses import dataclass
from collections.abc import Callable
from threading import RLock
from uuid import uuid4

import msgspec

from f8pysdk.f8_naming import ensure_token
from f8pysdk.specs import F8OperatorSpec, F8ServiceSpec
from f8studio_core.graph import (
    GraphNode,
    GraphStore,
    HistoryRequest,
    IdempotencyConflictError,
    PatchRequest,
    PatchResult,
    StudioDocument,
    new_document,
    validate_document,
)
from f8studio_core.graph.codec import canonical_json_bytes
from f8studio_core.graph.spec_edit import validate_spec_edit

from .models import CreateProjectRequest, ProjectRecord, ProjectSummary, UpdateProjectRequest
from .project_repository import ProjectRepository, StoredRequest, utc_now_text


@dataclass(frozen=True)
class ProjectMutationResult:
    result: PatchResult
    replayed: bool


def _request_fingerprint(action: str, request: object) -> str:
    return hashlib.sha256(canonical_json_bytes((action, request))).hexdigest()


class ProjectService:
    def __init__(
        self,
        repository: ProjectRepository,
        *,
        spec_resolver: Callable[[GraphNode], F8ServiceSpec | F8OperatorSpec] | None = None,
    ) -> None:
        self._repository = repository
        self._spec_resolver = spec_resolver
        self._stores: dict[str, GraphStore] = {}
        self._lock = RLock()

    def create(self, request: CreateProjectRequest) -> ProjectRecord:
        with self._lock:
            project_id = ensure_token(request.project_id or uuid4().hex, label="project_id")
            name = request.name.strip() or "Untitled Project"
            timestamp = utc_now_text()
            document = new_document(project_id=project_id, graph_id=project_id)
            record = ProjectRecord(
                project_id=project_id,
                name=name,
                description=request.description,
                created_at=timestamp,
                updated_at=timestamp,
                document=document,
            )
            saved = self._repository.create_project(record)
            self._stores[project_id] = GraphStore(document, spec_resolver=self._spec_resolver)
            return saved

    def list(self) -> tuple[ProjectSummary, ...]:
        return self._repository.list_projects()

    def get(self, project_id: str) -> ProjectRecord:
        project_id = ensure_token(project_id, label="project_id")
        record = self._repository.get_project(project_id)
        if record is None:
            raise FileNotFoundError(f"project not found: {project_id}")
        return record

    def update(self, project_id: str, request: UpdateProjectRequest) -> ProjectRecord:
        project_id = ensure_token(project_id, label="project_id")
        name = request.name.strip()
        if not name:
            raise ValueError("project name must be non-empty")
        return self._repository.update_metadata(
            project_id,
            name=name,
            description=request.description,
        )

    def delete(self, project_id: str) -> None:
        project_id = ensure_token(project_id, label="project_id")
        with self._lock:
            self._repository.delete_project(project_id)
            self._stores.pop(project_id, None)

    def document(self, project_id: str) -> StudioDocument:
        with self._lock:
            return self._store(project_id).snapshot()

    def validate(self, document: StudioDocument) -> None:
        validate_document(document)
        self._validate_installed_definitions(document)

    def _validate_installed_definitions(self, document: StudioDocument) -> None:
        if self._spec_resolver is None:
            return
        for node in document.nodes:
            try:
                validate_spec_edit(self._spec_resolver(node), node.spec)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"node {node.node_id} differs from installed definition: {exc}") from exc

    def restore(self, project_id: str, snapshot: StudioDocument) -> ProjectRecord:
        project_id = ensure_token(project_id, label="project_id")
        if snapshot.project_id != project_id:
            raise ValueError("snapshot projectId does not match target project")
        with self._lock:
            current = self.get(project_id).document
            restored = msgspec.structs.replace(
                snapshot,
                graph_id=current.graph_id,
                graph_revision=current.graph_revision + 1,
                layout_revision=current.layout_revision + 1,
            )
            self.validate(restored)
            record = self._repository.replace_document(project_id, restored)
            self._stores[project_id] = GraphStore(restored, spec_resolver=self._spec_resolver)
            return record

    def patch(self, project_id: str, request: PatchRequest) -> ProjectMutationResult:
        return self._mutate(project_id, action="patch", request=request)

    def undo(self, project_id: str, request: HistoryRequest) -> ProjectMutationResult:
        return self._mutate(project_id, action="undo", request=request)

    def redo(self, project_id: str, request: HistoryRequest) -> ProjectMutationResult:
        return self._mutate(project_id, action="redo", request=request)

    def _mutate(
        self,
        project_id: str,
        *,
        action: str,
        request: PatchRequest | HistoryRequest,
    ) -> ProjectMutationResult:
        project_id = ensure_token(project_id, label="project_id")
        fingerprint = _request_fingerprint(action, request)
        with self._lock:
            stored = self._repository.lookup_request(project_id, request.request_id)
            if stored is not None:
                return ProjectMutationResult(
                    result=self._stored_result(stored, action=action, fingerprint=fingerprint),
                    replayed=True,
                )
            store = self._store(project_id)

            def persist(result: PatchResult) -> None:
                self._repository.commit_result(
                    project_id,
                    action=action,
                    fingerprint=fingerprint,
                    result=result,
                )

            if action == "patch":
                if not isinstance(request, PatchRequest):
                    raise TypeError("patch action requires PatchRequest")
                result = store.apply_with_commit(request, before_commit=persist)
                return ProjectMutationResult(result=result, replayed=False)
            if not isinstance(request, HistoryRequest):
                raise TypeError(f"{action} action requires HistoryRequest")
            if action == "undo":
                result = store.undo_with_commit(request, before_commit=persist)
                return ProjectMutationResult(result=result, replayed=False)
            if action == "redo":
                result = store.redo_with_commit(request, before_commit=persist)
                return ProjectMutationResult(result=result, replayed=False)
            raise ValueError(f"unsupported project action: {action}")

    @staticmethod
    def _stored_result(stored: StoredRequest, *, action: str, fingerprint: str) -> PatchResult:
        if stored.action != action or stored.fingerprint != fingerprint:
            raise IdempotencyConflictError("request id reused with different content")
        return stored.result

    def _store(self, project_id: str) -> GraphStore:
        store = self._stores.get(project_id)
        if store is not None:
            return store
        record = self._repository.get_project(project_id)
        if record is None:
            raise FileNotFoundError(f"project not found: {project_id}")
        store = GraphStore(record.document, spec_resolver=self._spec_resolver)
        self._stores[project_id] = store
        return store


__all__ = ["ProjectMutationResult", "ProjectService"]
