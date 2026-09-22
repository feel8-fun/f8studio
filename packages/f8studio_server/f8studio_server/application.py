from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from f8media_protocol.client import RemoteMediaGateway
from f8media_protocol.contracts import MediaGateway

from .catalog import CatalogService
from .events import EventJournal
from .job_repository import JobRepository
from .jobs import DeployCoordinator
from .monitors import RuntimeMonitorStore
from .processes import ManagedServiceProcesses
from .project_repository import ProjectRepository
from .projects import ProjectService
from .runtime import RuntimeConfig, RuntimeGateway, ZenohRuntimeGateway
from .studio_runtime import EventPresentationOutlet, StudioRuntimeConfig, StudioRuntimeService


class StudioApplication:
    def __init__(
        self,
        *,
        data_dir: Path,
        server_epoch: str | None = None,
        runtime: RuntimeGateway | None = None,
        runtime_config: RuntimeConfig | None = None,
        service_roots: tuple[Path, ...] | None = None,
        media_gateway: MediaGateway | None = None,
    ) -> None:
        self.data_dir = data_dir.resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.server_epoch = server_epoch or uuid4().hex
        config = runtime_config or RuntimeConfig()
        self.events = EventJournal(server_epoch=self.server_epoch)
        self.presentation = EventPresentationOutlet(self.events)
        self.studio_runtime = StudioRuntimeService(
            StudioRuntimeConfig(
                bus_backend=config.bus_backend,
                zenoh_config_path=config.zenoh_config_path,
                zenoh_connect=config.zenoh_connect,
                zenoh_listen=config.zenoh_listen,
                zenoh_shm_pool_bytes=config.zenoh_shm_pool_bytes,
            ),
            presentation=self.presentation,
        )
        self.catalog = CatalogService(roots=service_roots, builtins=(self.studio_runtime.describe,))
        project_repository = ProjectRepository(self.data_dir / "studio.sqlite3")
        self.projects = ProjectService(project_repository)
        self._owns_runtime = runtime is None
        self.runtime = runtime or ZenohRuntimeGateway(config)
        self.monitors = RuntimeMonitorStore(self.events)
        if media_gateway is None:
            media_gateway = RemoteMediaGateway()
        self.media_gateway = media_gateway
        self.jobs = DeployCoordinator(
            projects=self.projects,
            repository=JobRepository(project_repository.database_path),
            runtime=self.runtime,
            events=self.events,
        )
        self.processes = ManagedServiceProcesses(
            catalog=self.catalog,
            runtime_config=config,
            events=self.events,
        )

    async def start(self) -> None:
        await self.media_gateway.start()
        if self._owns_runtime:
            await self.studio_runtime.start()
        await self.runtime.start_monitoring(self.monitors.ingest)

    async def close(self) -> None:
        await self.jobs.close()
        await self.media_gateway.close()
        await self.processes.close()
        await self.studio_runtime.stop()
        await self.runtime.close()
        await self.presentation.close()


__all__ = ["StudioApplication"]
