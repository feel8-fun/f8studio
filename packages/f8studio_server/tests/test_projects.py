from pathlib import Path

import pytest

from f8pysdk.specs import F8ServiceSpec
from f8studio_core.graph import (
    CreateNodeOp,
    IdempotencyConflictError,
    NodeCatalog,
    PatchRequest,
)
from f8studio_server.models import CreateProjectRequest
from f8studio_server.project_repository import ProjectRepository
from f8studio_server.projects import ProjectService


def build_service(database_path: Path) -> ProjectService:
    return ProjectService(ProjectRepository(database_path))


def test_project_patch_is_persisted_and_idempotent_across_service_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "studio.sqlite3"
    service = build_service(database_path)
    project = service.create(CreateProjectRequest(project_id="project1", name="Example"))
    catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.pyengine", label="Engine")])
    engine = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")
    request = PatchRequest(
        request_id="create-engine",
        expected_graph_revision=0,
        expected_layout_revision=0,
        operations=(CreateNodeOp(node=engine),),
    )

    first = service.patch(project.project_id, request)
    restarted = build_service(database_path)
    replayed = restarted.patch(project.project_id, request)

    assert first.replayed is False
    assert replayed.replayed is True
    assert replayed.result == first.result
    assert restarted.get(project.project_id).document == first.result.document
    assert restarted.list()[0].graph_revision == 1


def test_project_request_id_conflict_survives_restart(tmp_path: Path) -> None:
    database_path = tmp_path / "studio.sqlite3"
    service = build_service(database_path)
    service.create(CreateProjectRequest(project_id="project1", name="Example"))
    catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.pyengine", label="Engine")])
    engine = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")
    service.patch(
        "project1",
        PatchRequest(
            request_id="request1",
            expected_graph_revision=0,
            expected_layout_revision=0,
            operations=(CreateNodeOp(node=engine),),
        ),
    )

    restarted = build_service(database_path)
    with pytest.raises(IdempotencyConflictError):
        restarted.patch(
            "project1",
            PatchRequest(
                request_id="request1",
                expected_graph_revision=1,
                expected_layout_revision=0,
                operations=(),
            ),
        )
