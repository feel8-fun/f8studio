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
    RuntimeStateField,
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

    async def read_state(self, service_id: str, *, node_id: str, field: str) -> RuntimeStateField:
        del service_id, node_id
        return RuntimeStateField(field=field, found=False)

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


class EndpointRuntimeGateway(FakeRuntimeGateway):
    def __init__(self, *, service_class: str, online: set[str] | None = None) -> None:
        super().__init__()
        self.service_class = service_class
        self.online = set() if online is None else online
        self.status_calls: list[str] = []

    async def status(self, service_id: str) -> ServiceRuntimeStatus:
        self.status_calls.append(service_id)
        if service_id not in self.online:
            raise TimeoutError(f"endpoint offline: {service_id}")
        return ServiceRuntimeStatus(
            service_id=service_id,
            service_class=self.service_class,
            runtime_instance_id="runtime1",
            active=True,
        )


class FakeServiceProcesses:
    def __init__(
        self,
        *,
        launchable: set[str],
        online: set[str],
        start_error: OSError | None = None,
    ) -> None:
        self.launchable = launchable
        self.online = online
        self.start_error = start_error
        self.running: set[str] = set()
        self.start_calls: list[tuple[str, str]] = []

    def can_start(self, service_class: str) -> bool:
        return service_class in self.launchable

    def is_running(self, service_id: str) -> bool:
        return service_id in self.running

    async def start(self, service_id: str, *, service_class: str) -> object:
        self.start_calls.append((service_id, service_class))
        if self.start_error is not None:
            raise self.start_error
        self.running.add(service_id)
        self.online.add(service_id)
        return object()

    def stop(self, service_id: str) -> None:
        self.running.discard(service_id)
        self.online.discard(service_id)


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


def test_deploy_starts_launchable_service_when_endpoint_is_offline(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "studio.sqlite3"
        projects = ProjectService(ProjectRepository(database_path))
        projects.create(CreateProjectRequest(project_id="project1", name="Screen capture"))
        catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.screencap", label="Screen Capture")])
        capture = catalog.create_service_node(node_id="capture", service_class="f8.screencap")
        projects.patch(
            "project1",
            PatchRequest(
                request_id="create-capture",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(CreateNodeOp(node=capture),),
            ),
        )
        online: set[str] = set()
        runtime = EndpointRuntimeGateway(service_class="f8.screencap", online=online)
        processes = FakeServiceProcesses(launchable={"f8.screencap"}, online=online)
        coordinator = DeployCoordinator(
            projects=projects,
            repository=JobRepository(database_path),
            runtime=runtime,
            events=EventJournal(server_epoch="epoch1"),
            processes=processes,
        )

        submitted = await coordinator.submit(
            "project1",
            DeployProjectRequest(request_id="deploy1", expected_graph_revision=1),
        )

        assert await wait_for_terminal_job(coordinator, submitted.job_id) == JobStatus.succeeded
        assert runtime.status_calls == ["capture"]
        assert processes.start_calls == [("capture", "f8.screencap")]
        assert runtime.deploy_calls == ["capture"]
        await coordinator.close()

    asyncio.run(scenario())


def test_deploy_reuses_online_endpoint_without_starting_process(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "studio.sqlite3"
        projects = ProjectService(ProjectRepository(database_path))
        projects.create(CreateProjectRequest(project_id="project1", name="Screen capture"))
        catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.screencap", label="Screen Capture")])
        capture = catalog.create_service_node(node_id="capture", service_class="f8.screencap")
        projects.patch(
            "project1",
            PatchRequest(
                request_id="create-capture",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(CreateNodeOp(node=capture),),
            ),
        )
        online = {"capture"}
        runtime = EndpointRuntimeGateway(service_class="f8.screencap", online=online)
        processes = FakeServiceProcesses(launchable={"f8.screencap"}, online=online)
        coordinator = DeployCoordinator(
            projects=projects,
            repository=JobRepository(database_path),
            runtime=runtime,
            events=EventJournal(server_epoch="epoch1"),
            processes=processes,
        )

        submitted = await coordinator.submit(
            "project1",
            DeployProjectRequest(request_id="deploy1", expected_graph_revision=1),
        )

        assert await wait_for_terminal_job(coordinator, submitted.job_id) == JobStatus.succeeded
        assert runtime.status_calls == ["capture"]
        assert processes.start_calls == []
        await coordinator.close()

    asyncio.run(scenario())


def test_deploy_does_not_launch_builtin_service(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "studio.sqlite3"
        projects = ProjectService(ProjectRepository(database_path))
        projects.create(CreateProjectRequest(project_id="project1", name="Studio"))
        catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.pystudio", label="Studio")])
        studio = catalog.create_service_node(node_id="studio", service_class="f8.pystudio")
        projects.patch(
            "project1",
            PatchRequest(
                request_id="create-studio",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(CreateNodeOp(node=studio),),
            ),
        )
        online = {"studio"}
        runtime = EndpointRuntimeGateway(service_class="f8.pystudio", online=online)
        processes = FakeServiceProcesses(launchable=set(), online=online)
        coordinator = DeployCoordinator(
            projects=projects,
            repository=JobRepository(database_path),
            runtime=runtime,
            events=EventJournal(server_epoch="epoch1"),
            processes=processes,
        )

        submitted = await coordinator.submit(
            "project1",
            DeployProjectRequest(request_id="deploy1", expected_graph_revision=1),
        )

        assert await wait_for_terminal_job(coordinator, submitted.job_id) == JobStatus.succeeded
        assert runtime.status_calls == []
        assert processes.start_calls == []
        await coordinator.close()

    asyncio.run(scenario())


def test_process_start_failure_is_reported_per_service(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "studio.sqlite3"
        projects = ProjectService(ProjectRepository(database_path))
        projects.create(CreateProjectRequest(project_id="project1", name="Screen capture"))
        catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.screencap", label="Screen Capture")])
        capture = catalog.create_service_node(node_id="capture", service_class="f8.screencap")
        projects.patch(
            "project1",
            PatchRequest(
                request_id="create-capture",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(CreateNodeOp(node=capture),),
            ),
        )
        online: set[str] = set()
        runtime = EndpointRuntimeGateway(service_class="f8.screencap", online=online)
        processes = FakeServiceProcesses(
            launchable={"f8.screencap"},
            online=online,
            start_error=OSError("executable denied"),
        )
        coordinator = DeployCoordinator(
            projects=projects,
            repository=JobRepository(database_path),
            runtime=runtime,
            events=EventJournal(server_epoch="epoch1"),
            processes=processes,
        )

        submitted = await coordinator.submit(
            "project1",
            DeployProjectRequest(request_id="deploy1", expected_graph_revision=1),
        )

        assert await wait_for_terminal_job(coordinator, submitted.job_id) == JobStatus.failed
        finished = await coordinator.get(submitted.job_id)
        result = finished.service_results[0]
        assert result.success is False
        assert "serviceId=capture serviceClass=f8.screencap" in result.error_message
        assert "OSError: executable denied" in result.error_message
        assert runtime.deploy_calls == []
        await coordinator.close()

    asyncio.run(scenario())


def test_force_deploy_restarts_launchable_service_after_stop(tmp_path: Path) -> None:
    async def scenario() -> None:
        database_path = tmp_path / "studio.sqlite3"
        projects = ProjectService(ProjectRepository(database_path))
        projects.create(CreateProjectRequest(project_id="project1", name="Screen capture"))
        catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.screencap", label="Screen Capture")])
        capture = catalog.create_service_node(node_id="capture", service_class="f8.screencap")
        projects.patch(
            "project1",
            PatchRequest(
                request_id="create-capture",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(CreateNodeOp(node=capture),),
            ),
        )
        online: set[str] = set()
        runtime = EndpointRuntimeGateway(service_class="f8.screencap", online=online)
        processes = FakeServiceProcesses(launchable={"f8.screencap"}, online=online)
        coordinator = DeployCoordinator(
            projects=projects,
            repository=JobRepository(database_path),
            runtime=runtime,
            events=EventJournal(server_epoch="epoch1"),
            processes=processes,
        )

        first = await coordinator.submit(
            "project1",
            DeployProjectRequest(request_id="deploy1", expected_graph_revision=1),
        )
        assert await wait_for_terminal_job(coordinator, first.job_id) == JobStatus.succeeded
        processes.stop("capture")
        second = await coordinator.submit(
            "project1",
            DeployProjectRequest(request_id="deploy2", expected_graph_revision=1, force_apply=True),
        )

        assert await wait_for_terminal_job(coordinator, second.job_id) == JobStatus.succeeded
        assert processes.start_calls == [
            ("capture", "f8.screencap"),
            ("capture", "f8.screencap"),
        ]
        assert runtime.deploy_calls == ["capture", "capture"]
        await coordinator.close()

    asyncio.run(scenario())
