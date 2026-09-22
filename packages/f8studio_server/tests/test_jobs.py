import asyncio
from pathlib import Path

from f8pysdk.specs import F8JsonValue, F8RuntimeGraph, F8ServiceSpec
from f8studio_core.graph import CreateNodeOp, NodeCatalog, PatchRequest
from f8studio_server.events import EventJournal
from f8studio_server.job_repository import JobRepository
from f8studio_server.jobs import DeployCoordinator
from f8studio_server.models import (
    CreateProjectRequest,
    DeployProjectRequest,
    JobStatus,
    RuntimeActionResult,
    ServiceDeployResult,
    ServiceRuntimeStatus,
)
from f8studio_server.project_repository import ProjectRepository
from f8studio_server.projects import ProjectService
from f8studio_server.runtime import RuntimeMonitorCallback


class FakeRuntimeGateway:
    def __init__(self) -> None:
        self.deploy_calls: list[str] = []

    async def start_monitoring(self, callback: RuntimeMonitorCallback) -> None:
        del callback

    async def deploy(
        self,
        *,
        service_id: str,
        graph: F8RuntimeGraph,
        force_apply: bool,
    ) -> ServiceDeployResult:
        del graph, force_apply
        self.deploy_calls.append(service_id)
        return ServiceDeployResult(service_id=service_id, success=True)

    async def status(self, service_id: str) -> ServiceRuntimeStatus:
        return ServiceRuntimeStatus(
            service_id=service_id,
            service_class="f8.pyengine",
            runtime_instance_id="runtime1",
            active=True,
        )

    async def set_active(self, service_id: str, *, active: bool) -> RuntimeActionResult:
        del service_id, active
        return RuntimeActionResult(success=True)

    async def set_state(
        self,
        service_id: str,
        *,
        node_id: str,
        field: str,
        value: F8JsonValue,
    ) -> RuntimeActionResult:
        del service_id, node_id, field, value
        return RuntimeActionResult(success=True)

    async def invoke_command(
        self,
        service_id: str,
        *,
        call: str,
        params: dict[str, F8JsonValue],
    ) -> RuntimeActionResult:
        del service_id, call, params
        return RuntimeActionResult(success=True)

    async def terminate(self, service_id: str) -> RuntimeActionResult:
        del service_id
        return RuntimeActionResult(success=True)

    async def close(self) -> None:
        return


class BlockingRuntimeGateway(FakeRuntimeGateway):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def deploy(
        self,
        *,
        service_id: str,
        graph: F8RuntimeGraph,
        force_apply: bool,
    ) -> ServiceDeployResult:
        del graph, force_apply
        self.deploy_calls.append(service_id)
        self.started.set()
        await self.release.wait()
        return ServiceDeployResult(service_id=service_id, success=True)


class PartiallyFailingRuntimeGateway(FakeRuntimeGateway):
    async def deploy(
        self,
        *,
        service_id: str,
        graph: F8RuntimeGraph,
        force_apply: bool,
    ) -> ServiceDeployResult:
        del graph, force_apply
        self.deploy_calls.append(service_id)
        if service_id == "offline":
            raise OSError("Zenoh endpoint disconnected")
        return ServiceDeployResult(service_id=service_id, success=True)


async def wait_for_terminal_job(coordinator: DeployCoordinator, job_id: str) -> JobStatus:
    for _ in range(100):
        job = await coordinator.get(job_id)
        if job.status not in {JobStatus.queued, JobStatus.running}:
            return job.status
        await asyncio.sleep(0.01)
    raise AssertionError("deploy job did not finish")


def test_deploy_job_runs_and_semantic_duplicate_skips_runtime(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "studio.sqlite3"
        projects = ProjectService(ProjectRepository(database_path))
        projects.create(CreateProjectRequest(project_id="project1", name="Example"))
        catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.pyengine", label="Engine")])
        engine = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")
        projects.patch(
            "project1",
            PatchRequest(
                request_id="create-engine",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(CreateNodeOp(node=engine),),
            ),
        )
        runtime = FakeRuntimeGateway()
        coordinator = DeployCoordinator(
            projects=projects,
            repository=JobRepository(database_path),
            runtime=runtime,
            events=EventJournal(server_epoch="epoch1"),
        )

        first = await coordinator.submit(
            "project1",
            DeployProjectRequest(request_id="deploy1", expected_graph_revision=1),
        )
        assert await wait_for_terminal_job(coordinator, first.job_id) == JobStatus.succeeded
        duplicate = await coordinator.submit(
            "project1",
            DeployProjectRequest(request_id="deploy2", expected_graph_revision=1),
        )

        assert duplicate.status == JobStatus.succeeded
        assert runtime.deploy_calls == ["engine"]
        await coordinator.close()

    asyncio.run(scenario())


def test_cancelling_running_job_persists_cancelled_status(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "studio.sqlite3"
        projects = ProjectService(ProjectRepository(database_path))
        projects.create(CreateProjectRequest(project_id="project1", name="Example"))
        catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.pyengine", label="Engine")])
        engine = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")
        projects.patch(
            "project1",
            PatchRequest(
                request_id="create-engine",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(CreateNodeOp(node=engine),),
            ),
        )
        runtime = BlockingRuntimeGateway()
        coordinator = DeployCoordinator(
            projects=projects,
            repository=JobRepository(database_path),
            runtime=runtime,
            events=EventJournal(server_epoch="epoch1"),
        )

        submitted = await coordinator.submit(
            "project1",
            DeployProjectRequest(request_id="deploy1", expected_graph_revision=1),
        )
        await asyncio.wait_for(runtime.started.wait(), timeout=1.0)
        cancelled = await coordinator.cancel(submitted.job_id)

        assert cancelled.status == JobStatus.cancelled
        assert (await coordinator.get(submitted.job_id)).status == JobStatus.cancelled
        assert "not rolled back" in cancelled.error_message
        await coordinator.close()

    asyncio.run(scenario())


def test_multi_service_deploy_persists_partial_failure_results(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "studio.sqlite3"
        projects = ProjectService(ProjectRepository(database_path))
        projects.create(CreateProjectRequest(project_id="project1", name="Partial deployment"))
        catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.pyengine", label="Engine")])
        online = catalog.create_service_node(node_id="online", service_class="f8.pyengine")
        offline = catalog.create_service_node(node_id="offline", service_class="f8.pyengine")
        projects.patch(
            "project1",
            PatchRequest(
                request_id="create-services",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(CreateNodeOp(node=online), CreateNodeOp(node=offline)),
            ),
        )
        runtime = PartiallyFailingRuntimeGateway()
        coordinator = DeployCoordinator(
            projects=projects,
            repository=JobRepository(database_path),
            runtime=runtime,
            events=EventJournal(server_epoch="epoch1"),
        )

        submitted = await coordinator.submit(
            "project1",
            DeployProjectRequest(request_id="deploy-partial", expected_graph_revision=1),
        )
        assert await wait_for_terminal_job(coordinator, submitted.job_id) == JobStatus.partially_failed
        finished = await coordinator.get(submitted.job_id)
        results = {result.service_id: result for result in finished.service_results}
        assert results["online"].success is True
        assert results["offline"].success is False
        assert "Zenoh endpoint disconnected" in results["offline"].error_message
        assert finished.error_message == "one or more services rejected the deployment"
        await coordinator.close()

    asyncio.run(scenario())
