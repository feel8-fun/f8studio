from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path

import pytest

from f8pysdk.specs import F8JsonValue, F8RuntimeGraph, F8ServiceSpec
from f8studio_core.graph import CreateNodeOp, NodeCatalog, PatchRequest
from f8studio_server.application import StudioApplication
from f8studio_server.assets import (
    ASSET_SCHEMA_VERSION,
    AssetExport,
    AssetKind,
    AssetRepository,
    CreateAssetRequest,
    UpdateAssetRequest,
)
from f8studio_server.editor import (
    CreateEditorSessionRequest,
    EditorPositionRequest,
    EditorSessionService,
    EditorSupportFile,
    UpdateEditorDocumentRequest,
)
from f8studio_server.local_integration import LocalIntegrationService, RegisterHotkeyRequest, VerifySkeletonUdpRequest
from f8studio_server.models import (
    CreateCatalogNodeRequest,
    CreateProjectRequest,
    RuntimeActionResult,
    RuntimeStateField,
    ServiceDeployResult,
    ServiceRuntimeStatus,
)
from f8studio_server.project_repository import ProjectRepository
from f8studio_server.projects import ProjectService
from f8studio_server.runtime import RuntimeMonitorCallback


class HotkeyRuntimeGateway:
    def __init__(self) -> None:
        self.state_calls: list[tuple[str, str, str, F8JsonValue]] = []

    async def start_monitoring(self, callback: RuntimeMonitorCallback) -> None:
        del callback

    async def deploy(self, *, service_id: str, graph: F8RuntimeGraph, force_apply: bool) -> ServiceDeployResult:
        del graph, force_apply
        return ServiceDeployResult(service_id=service_id, success=True)

    async def status(self, service_id: str) -> ServiceRuntimeStatus:
        return ServiceRuntimeStatus(
            service_id=service_id,
            service_class="f8.pystudio",
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
        self.state_calls.append((service_id, node_id, field, value))
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


def _asset_repository(tmp_path: Path) -> AssetRepository:
    project_repository = ProjectRepository(tmp_path / "studio.sqlite3")
    return AssetRepository(project_repository.database_path)


def test_assets_are_validated_versioned_and_exportable(tmp_path: Path) -> None:
    repository = _asset_repository(tmp_path)
    created = repository.create(
        CreateAssetRequest(
            asset_id="component1",
            kind=AssetKind.component,
            name="Reusable fragment",
            tags=("vision", "vision", " local "),
            content={"schemaVersion": "f8studio-component/1", "nodes": [], "edges": [], "layout": []},
        )
    )
    updated = repository.update(
        created.asset_id,
        UpdateAssetRequest(
            name="Reusable fragment v2",
            description="tested",
            tags=("vision",),
            content={"schemaVersion": "f8studio-component/1", "nodes": [], "edges": [], "layout": []},
        ),
    )

    assert created.current_version == 1
    assert created.tags == ("vision", "local")
    assert updated.current_version == 2
    assert [version.version for version in repository.versions(created.asset_id)] == [2, 1]
    exported = repository.export(created.asset_id)
    assert exported.schema_version == ASSET_SCHEMA_VERSION
    assert exported.asset.name == "Reusable fragment v2"

    other = _asset_repository(tmp_path / "other")
    imported = other.import_asset(
        AssetExport(schema_version=exported.schema_version, asset=exported.asset, versions=exported.versions)
    )
    assert imported.content == updated.content
    assert imported.current_version == 2
    assert [version.version for version in other.versions(imported.asset_id)] == [2, 1]

    with pytest.raises(ValueError, match="layout must reference component nodes"):
        repository.create(
            CreateAssetRequest(
                kind=AssetKind.component,
                name="Invalid fragment",
                content={
                    "schemaVersion": "f8studio-component/1",
                    "nodes": [],
                    "edges": [],
                    "layout": [{"nodeId": "missing", "x": 0, "y": 0}],
                },
            )
        )


def test_project_versions_restore_as_a_new_revision(tmp_path: Path) -> None:
    project_repository = ProjectRepository(tmp_path / "studio.sqlite3")
    projects = ProjectService(project_repository)
    assets = AssetRepository(project_repository.database_path)
    original = projects.create(CreateProjectRequest(project_id="project1", name="Versioned"))
    version = assets.create_project_version("project1", "Empty", original.document)
    catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.pyengine", label="Engine")])
    engine = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")
    changed = projects.patch(
        "project1",
        PatchRequest(
            request_id="add-engine",
            expected_graph_revision=0,
            expected_layout_revision=0,
            operations=(CreateNodeOp(node=engine),),
        ),
    )

    restored = projects.restore("project1", assets.get_project_version("project1", version.version_id).document)

    assert len(changed.result.document.nodes) == 1
    assert restored.document.nodes == ()
    assert restored.document.graph_revision == 2
    assert restored.document.layout_revision == 1


def test_editor_sessions_enforce_versions_and_return_structured_diagnostics(tmp_path: Path) -> None:
    editor = EditorSessionService(root=tmp_path / "editor")
    json_session = editor.create(CreateEditorSessionRequest(language="json", filename="schema.json", text="{"))
    json_analysis = editor.analyze(json_session.session_id)
    assert json_analysis.engine == "json"
    assert json_analysis.diagnostics[0].severity == "error"

    python_session = editor.create(
        CreateEditorSessionRequest(language="python", filename="node.py", text="value: int = 'wrong'\n")
    )
    python_analysis = editor.analyze(python_session.session_id)
    assert python_analysis.engine == "basedpyright"
    assert any(item.severity == "error" for item in python_analysis.diagnostics)

    completion_session = editor.create(
        CreateEditorSessionRequest(language="python", filename="completion.py", text="from pathlib import Path\nPath.\n")
    )
    completion = editor.completion(
        completion_session.session_id,
        EditorPositionRequest(line=1, column=5),
    )
    assert completion.result is not None

    updated = editor.update(
        python_session.session_id,
        UpdateEditorDocumentRequest(version=2, text="value: int = 1\n"),
    )
    assert updated.version == 2
    assert editor.analyze(python_session.session_id).diagnostics == ()
    hover = editor.hover(python_session.session_id, EditorPositionRequest(line=0, column=2))
    assert hover.result is not None
    editor.close_session(python_session.session_id)
    editor.close_session(completion_session.session_id)

    with pytest.raises(ValueError, match="duplicate editor file path"):
        editor.create(
            CreateEditorSessionRequest(
                language="json",
                filename="schema.json",
                text="{}",
                support_files=(EditorSupportFile(path="schema.json", content="null"),),
            )
        )
    assert sorted(path.name for path in (tmp_path / "editor").iterdir()) == [json_session.session_id]


def test_udp_verifier_requires_a_decoded_complete_skeleton_frame() -> None:
    async def scenario() -> None:
        service = LocalIntegrationService()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        task = asyncio.create_task(
            service.verify_skeleton_udp(
                VerifySkeletonUdpRequest(port=port, timeout_ms=1000, minimum_frames=1)
            )
        )
        await asyncio.sleep(0.05)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.sendto(b"not-a-skeleton", ("127.0.0.1", port))
            sender.sendto(
                json.dumps({"type": "skeleton_binary", "modelName": "fixture", "bones": []}).encode(),
                ("127.0.0.1", port),
            )
        report = await task
        assert report.packet_count == 2
        assert report.decoded_frame_count == 1
        assert report.model_names == ("fixture",)
        assert report.verified is True
        assert report.decoder_errors

    asyncio.run(scenario())


def test_hotkey_contract_normalizes_accelerators() -> None:
    service = LocalIntegrationService()
    binding = service.register_hotkey(
        RegisterHotkeyRequest(
            accelerator="ctrl + shift + k",
            project_id="project1",
            node_id="controls",
            field="trigger",
        )
    )
    assert binding.accelerator == "Ctrl+Shift+K"
    assert service.list_hotkeys() == (binding,)
    with pytest.raises(ValueError, match="already registered"):
        service.register_hotkey(
            RegisterHotkeyRequest(
                accelerator="Ctrl+Shift+K",
                project_id="project1",
                node_id="controls",
                field="otherTrigger",
            )
        )
    service.unregister_hotkey(binding.binding_id)
    assert service.list_hotkeys() == ()


def test_hotkey_activation_commits_graph_state_and_syncs_runtime(tmp_path: Path) -> None:
    runtime = HotkeyRuntimeGateway()
    studio = StudioApplication(data_dir=tmp_path, runtime=runtime, service_roots=())
    project = studio.projects.create(CreateProjectRequest(project_id="project1", name="Hotkeys"))
    service_node = studio.catalog.create_node(
        CreateCatalogNodeRequest(kind="service", node_id="studio", service_class="f8.pystudio")
    )
    stepper = studio.catalog.create_node(
        CreateCatalogNodeRequest(
            kind="operator",
            node_id="stepper",
            service_class="f8.pystudio",
            service_id="studio",
            operator_class="f8.value_stepper",
        )
    )
    studio.projects.patch(
        project.project_id,
        PatchRequest(
            request_id="create-controls",
            expected_graph_revision=0,
            expected_layout_revision=0,
            operations=(CreateNodeOp(node=service_node), CreateNodeOp(node=stepper)),
        ),
    )
    binding = studio.local.register_hotkey(
        RegisterHotkeyRequest(
            accelerator="Ctrl+Alt+P",
            project_id=project.project_id,
            node_id="stepper",
            field="increaseTrigger",
        )
    )

    asyncio.run(studio._activate_hotkey(binding))

    updated = studio.projects.document(project.project_id)
    updated_stepper = next(node for node in updated.nodes if node.node_id == "stepper")
    assert updated_stepper.state_values["increaseTrigger"] == 1
    assert runtime.state_calls == [("studio", "stepper", "increaseTrigger", 1)]
    studio.editor.close()
