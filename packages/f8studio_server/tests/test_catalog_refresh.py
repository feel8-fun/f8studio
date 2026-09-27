from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from f8pysdk.service_runtime_tools.inventory import ServiceCatalog
from f8pysdk.specs import F8OperatorSchemaVersion, F8OperatorSpec, F8ServiceSchemaVersion, F8ServiceSpec
from f8studio_core.graph import NodeCatalog, StudioDocument
from f8studio_server.app import create_app
from f8studio_server.application import StudioApplication
from f8studio_server.catalog import CatalogService
from f8studio_server.models import DeployProjectRequest
from f8studio_server.processes import ManagedProcessResult


def test_catalog_refresh_replaces_stale_specs_in_shared_catalog() -> None:
    generation = 0

    def discover(*, roots: list[Path] | None, catalog: ServiceCatalog,
                 force_dynamic_service_classes: tuple[str, ...]) -> list[str]:
        nonlocal generation
        del roots, force_dynamic_service_classes
        generation += 1
        catalog.register_service(F8ServiceSpec(
            schemaVersion=F8ServiceSchemaVersion.f8service_1, serviceClass="test.engine", label="Engine",
        ))
        catalog.register_operator(F8OperatorSpec(
            schemaVersion=F8OperatorSchemaVersion.f8operator_1, serviceClass="test.engine",
            operatorClass=f"test.node_{generation}", label="Node",
        ))
        return ["test.engine"]

    with patch("f8studio_server.catalog.load_discovery_into_catalog", side_effect=discover):
        service = CatalogService(roots=())
        shared_catalog = service.sdk_catalog
        assert [spec.operatorClass for spec in service.snapshot().operators] == ["test.node_1"]
        updated = service.refresh()

    assert service.sdk_catalog is shared_catalog
    assert [spec.operatorClass for spec in updated.operators] == ["test.node_2"]
    assert not shared_catalog.operators.has("test.engine", "test.node_1")


def test_catalog_refresh_keeps_existing_specs_when_live_description_fails() -> None:
    attempts = 0

    def discover(*, roots: list[Path] | None, catalog: ServiceCatalog,
                 force_dynamic_service_classes: tuple[str, ...]) -> list[str]:
        nonlocal attempts
        del roots, force_dynamic_service_classes
        attempts += 1
        if attempts == 1:
            catalog.register_service(F8ServiceSpec(
                schemaVersion=F8ServiceSchemaVersion.f8service_1, serviceClass="test.engine", label="Engine",
            ))
            return ["test.engine"]
        return []

    with patch("f8studio_server.catalog.load_discovery_into_catalog", side_effect=discover):
        service = CatalogService(roots=())
        with pytest.raises(RuntimeError, match="test.engine"):
            service.refresh(force_dynamic_service_classes=("test.engine",))

    assert service.sdk_catalog.services.has("test.engine")


def test_catalog_refresh_api_returns_updated_snapshot(tmp_path: Path) -> None:
    application = StudioApplication(data_dir=tmp_path / "data", service_roots=())
    app = create_app(application=application, web_dist=tmp_path)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/api/catalog/refresh")
        assert response.status_code == 200
        assert any(item["serviceClass"] == "f8.pystudio" for item in response.json()["services"])

    asyncio.run(scenario())


def test_restart_service_refreshes_catalog_then_redeploys(tmp_path: Path) -> None:
    application = StudioApplication(data_dir=tmp_path / "data", service_roots=())
    service_spec = F8ServiceSpec(
        schemaVersion=F8ServiceSchemaVersion.f8service_1, serviceClass="test.engine", label="Engine",
    )
    service = NodeCatalog(services=(service_spec,), operators=()).create_service_node(
        node_id="engine", service_class="test.engine",
    )
    document = StudioDocument(
        schema_version="f8studio-document/2", project_id="project1", graph_id="graph1",
        graph_revision=2, layout_revision=0, nodes=(service,), edges=(),
    )
    app = create_app(application=application, web_dist=tmp_path)
    calls: list[str] = []

    async def cancel(_project_id: str) -> None:
        calls.append("cancel")

    async def terminate(_service_id: str) -> None:
        calls.append("terminate")

    async def stop(_service_id: str) -> ManagedProcessResult:
        calls.append("stop")
        return ManagedProcessResult(service_id="engine", running=False)

    async def start(_service_id: str, *, service_class: str) -> ManagedProcessResult:
        assert service_class == "test.engine"
        calls.append("start")
        return ManagedProcessResult(service_id="engine", running=True)

    async def deploy(_project_id: str, request: DeployProjectRequest) -> dict[str, str]:
        calls.append("deploy")
        assert request.expected_graph_revision == 2
        return {"jobId": "job1", "status": "queued"}

    def refresh(*, force_dynamic_service_classes: tuple[str, ...]) -> object:
        assert force_dynamic_service_classes == ("test.engine",)
        calls.append("refresh")
        return application.catalog.snapshot()

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/api/projects/project1/services/engine/restart")
        assert response.status_code == 202
        assert response.json()["jobId"] == "job1"

    with (
        patch.object(application.projects, "document", return_value=document),
        patch.object(application.processes, "can_start", return_value=True),
        patch.object(application.processes, "is_running", return_value=True),
        patch.object(application.catalog, "refresh", side_effect=refresh),
        patch.object(application.jobs, "cancel_project", new=AsyncMock(side_effect=cancel)),
        patch.object(application.runtime, "terminate", new=AsyncMock(side_effect=terminate)),
        patch.object(application.processes, "stop", new=AsyncMock(side_effect=stop)),
        patch.object(application.processes, "start", new=AsyncMock(side_effect=start)),
        patch.object(application.tools, "deploy", new=AsyncMock(side_effect=deploy)),
    ):
        asyncio.run(scenario())

    assert calls == ["refresh", "cancel", "terminate", "stop", "start", "deploy"]


def test_restart_rejects_external_process_without_stopping_it(tmp_path: Path) -> None:
    application = StudioApplication(data_dir=tmp_path / "data", service_roots=())
    service_spec = F8ServiceSpec(
        schemaVersion=F8ServiceSchemaVersion.f8service_1, serviceClass="test.engine", label="Engine",
    )
    service = NodeCatalog(services=(service_spec,), operators=()).create_service_node(
        node_id="engine", service_class="test.engine",
    )
    document = StudioDocument(
        schema_version="f8studio-document/2", project_id="project1", graph_id="graph1",
        graph_revision=2, layout_revision=0, nodes=(service,), edges=(),
    )
    app = create_app(application=application, web_dist=tmp_path)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/api/projects/project1/services/engine/restart")
        assert response.status_code == 409
        assert "not a running Studio-managed process" in response.json()["detail"]

    with (
        patch.object(application.projects, "document", return_value=document),
        patch.object(application.processes, "can_start", return_value=True),
        patch.object(application.processes, "is_running", return_value=False),
        patch.object(application.runtime, "terminate", new_callable=AsyncMock) as terminate,
    ):
        asyncio.run(scenario())
        terminate.assert_not_awaited()
