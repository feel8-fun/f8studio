from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import cast
from uuid import uuid4

import msgspec
from f8media_protocol.client import RemoteMediaGateway
from f8media_protocol.contracts import MediaGateway
from f8pysdk.generated import (
    F8BooleanTypeSchema,
    F8IntegerTypeSchema,
    F8NullTypeSchema,
    F8NumberTypeSchema,
    F8StateAccess,
    F8StateSpec,
    F8StringTypeSchema,
)
from f8pysdk.specs import F8JsonValue
from f8studio_core.graph import GraphNode, PatchRequest, RevisionConflictError, SetNodeStateOp, StudioDocument

from .catalog import CatalogService
from .assets import AssetRepository
from .editor import EditorSessionService
from .events import EventJournal
from .job_repository import JobRepository
from .jobs import DeployCoordinator
from .monitors import RuntimeMonitorStore
from .local_integration import HotkeyBinding, LocalIntegrationService
from .processes import ManagedServiceProcesses
from .project_repository import ProjectRepository
from .projects import ProjectService
from .runtime import RuntimeConfig, RuntimeGateway, ZenohRuntimeGateway
from .studio_runtime import EventPresentationOutlet, StudioRuntimeConfig, StudioRuntimeService


logger = logging.getLogger(__name__)
_HOTKEY_SELECT_CONTROLS = {"select", "dropdown", "dropbox", "combo", "combobox"}


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
        self.assets = AssetRepository(project_repository.database_path)
        self.editor = EditorSessionService(root=self.data_dir / "editor-sessions")
        self.local = LocalIntegrationService(
            database_path=project_repository.database_path,
            hotkey_activation=self._activate_hotkey,
            hotkey_validator=self._validate_hotkey,
        )
        self._owns_runtime = runtime is None
        self.runtime = runtime or ZenohRuntimeGateway(config)
        self.monitors = RuntimeMonitorStore(self.events)
        if media_gateway is None:
            media_gateway = RemoteMediaGateway()
        self.media_gateway = media_gateway
        self.processes = ManagedServiceProcesses(
            catalog=self.catalog,
            runtime_config=config,
            events=self.events,
        )
        self.jobs = DeployCoordinator(
            projects=self.projects,
            repository=JobRepository(project_repository.database_path),
            runtime=self.runtime,
            events=self.events,
            processes=self.processes,
        )

    async def start(self) -> None:
        await self.media_gateway.start()
        if self._owns_runtime:
            await self.studio_runtime.start()
        await self.runtime.start_monitoring(self.monitors.ingest)
        await self.local.start()

    async def close(self) -> None:
        await self.local.close()
        await self.jobs.close()
        await self.media_gateway.close()
        await self.processes.close()
        await self.studio_runtime.stop()
        await self.runtime.close()
        await self.presentation.close()
        self.editor.close()

    def _validate_hotkey(self, binding: HotkeyBinding) -> None:
        document, node, field = self._hotkey_target(binding)
        if field.access is not F8StateAccess.rw:
            raise ValueError("global hotkeys require a writable state field")
        control = self._state_control(field)
        is_numeric_button = control == "button" and isinstance(
            field.valueSchema,
            (F8IntegerTypeSchema, F8NumberTypeSchema),
        )
        is_select = control in _HOTKEY_SELECT_CONTROLS or bool(self._enum_values(field))
        if not is_numeric_button and not is_select:
            raise ValueError("global hotkeys support numeric button and select state controls")
        input_port_ids = {
            port.port_id
            for port in node.ports
            if port.kind.value == "state" and port.direction.value == "input" and port.runtime_name == field.name
        }
        if any(edge.to_node_id == node.node_id and edge.to_port_id in input_port_ids for edge in document.edges):
            raise ValueError("global hotkey target is driven by an upstream state connection")

    async def _activate_hotkey(self, binding: HotkeyBinding) -> None:
        mutation = None
        node: GraphNode | None = None
        next_value: F8JsonValue = None
        for _attempt in range(2):
            document, current_node, field = self._hotkey_target(binding)
            next_value = self._next_hotkey_value(current_node, field)
            request = PatchRequest(
                request_id=f"hotkey:{binding.binding_id}:{uuid4().hex}",
                expected_graph_revision=document.graph_revision,
                expected_layout_revision=document.layout_revision,
                operations=(SetNodeStateOp(node_id=current_node.node_id, field=field.name, value=next_value),),
            )
            try:
                mutation = await asyncio.to_thread(self.projects.patch, binding.project_id, request)
                node = current_node
                break
            except RevisionConflictError:
                continue
        if mutation is None or node is None:
            raise RevisionConflictError("global hotkey could not commit after a concurrent graph change")
        if not mutation.replayed:
            result = mutation.result
            payload = cast(
                F8JsonValue,
                msgspec.to_builtins(
                    {
                        "requestId": result.request_id,
                        "graphChanged": result.graph_changed,
                        "layoutChanged": result.layout_changed,
                        "document": result.document,
                    },
                    str_keys=True,
                ),
            )
            await self.events.publish(
                event_type="graph.committed",
                scope=f"project:{binding.project_id}",
                payload=payload,
            )
        try:
            await self.runtime.set_state(
                node.service_id,
                node_id=node.node_id,
                field=binding.field,
                value=next_value,
            )
        except (TimeoutError, OSError, RuntimeError, ValueError) as exc:
            logger.info(
                "global hotkey updated draft but runtime state sync was unavailable "
                "project_id=%s node_id=%s field=%s",
                binding.project_id,
                binding.node_id,
                binding.field,
                exc_info=exc,
            )

    def _hotkey_target(self, binding: HotkeyBinding) -> tuple[StudioDocument, GraphNode, F8StateSpec]:
        document = self.projects.document(binding.project_id)
        node = next((candidate for candidate in document.nodes if candidate.node_id == binding.node_id), None)
        if node is None:
            raise FileNotFoundError(f"global hotkey node not found: {binding.node_id}")
        state_fields = node.spec.stateFields
        fields = () if isinstance(state_fields, msgspec.UnsetType) else state_fields
        field = next((candidate for candidate in fields if candidate.name == binding.field), None)
        if field is None:
            raise FileNotFoundError(f"global hotkey state field not found: {binding.node_id}.{binding.field}")
        return document, node, field

    def _next_hotkey_value(self, node: GraphNode, field: F8StateSpec) -> F8JsonValue:
        current = node.state_values.get(field.name, self._schema_default(field))
        if self._state_control(field) == "button":
            if isinstance(field.valueSchema, F8IntegerTypeSchema):
                return int(current) + 1 if isinstance(current, (int, float)) and not isinstance(current, bool) else 1
            if isinstance(field.valueSchema, F8NumberTypeSchema):
                return float(current) + 1.0 if isinstance(current, (int, float)) and not isinstance(current, bool) else 1.0
        choices = self._enum_values(field) or self._pool_values(node, field)
        if not choices:
            raise ValueError(f"global hotkey select field has no choices: {node.node_id}.{field.name}")
        try:
            index = choices.index(current)
        except ValueError:
            return choices[0]
        return choices[(index + 1) % len(choices)]

    @staticmethod
    def _state_control(field: F8StateSpec) -> str:
        value = field.uiControl
        if isinstance(value, msgspec.UnsetType):
            return ""
        return value.split("[", 1)[0].strip().lower()

    @staticmethod
    def _enum_values(field: F8StateSpec) -> list[F8JsonValue]:
        schema = field.valueSchema
        if not isinstance(
            schema,
            (F8StringTypeSchema, F8NumberTypeSchema, F8IntegerTypeSchema, F8BooleanTypeSchema, F8NullTypeSchema),
        ):
            return []
        values = schema.enum
        if isinstance(values, msgspec.UnsetType):
            return []
        return cast(list[F8JsonValue], msgspec.to_builtins(values, str_keys=True))

    @staticmethod
    def _schema_default(field: F8StateSpec) -> F8JsonValue:
        value = field.valueSchema.default
        if isinstance(value, msgspec.UnsetType):
            return None
        return cast(F8JsonValue, msgspec.to_builtins(value, str_keys=True))

    @staticmethod
    def _pool_values(node: GraphNode, field: F8StateSpec) -> list[F8JsonValue]:
        value = field.uiControl
        if isinstance(value, msgspec.UnsetType):
            return []
        match = re.fullmatch(r"(?:select|dropdown|dropbox|combo|combobox)\[([A-Za-z_][A-Za-z0-9_]*)\]", value.strip())
        if match is None:
            return []
        pool_name = match.group(1)
        raw_pool = node.state_values.get(pool_name)
        if raw_pool is None:
            state_fields = node.spec.stateFields
            fields = () if isinstance(state_fields, msgspec.UnsetType) else state_fields
            pool_field = next((candidate for candidate in fields if candidate.name == pool_name), None)
            if pool_field is not None:
                raw_pool = StudioApplication._schema_default(pool_field)
        if not isinstance(raw_pool, list):
            return []
        return list(raw_pool)


__all__ = ["StudioApplication"]
