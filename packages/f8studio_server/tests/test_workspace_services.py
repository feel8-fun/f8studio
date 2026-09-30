from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from unittest.mock import AsyncMock, patch

import msgspec
import pytest

from f8pysdk.specs import F8JsonValue, F8RuntimeGraph, F8ServiceSpec
from f8studio_core.graph import CreateNodeOp, NodeCatalog, PatchRequest, SetNodeStateOp
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
    DeployJob,
    JobStatus,
    RuntimeActionResult,
    RuntimeStateField,
    ServiceDeployResult,
    ServiceRuntimeStatus,
)
from f8studio_server.project_repository import ProjectRepository
from f8studio_server.editor_context import editor_support_files
from f8studio_server.projects import ProjectService
from f8studio_server.runtime import RuntimeMonitorCallback


def test_python_script_editor_uses_injected_api_and_dynamic_bindings(tmp_path: Path) -> None:
    from f8pyengine.operators.python_script import PythonScriptRuntimeNode
    from f8studio_core.graph import OperatorNode

    node = OperatorNode(
        node_id="script", name="Script", service_id="engine", service_class="f8.pyengine",
        operator_class="f8.python_script", spec=PythonScriptRuntimeNode.SPEC,
    )
    support = editor_support_files(node, "code")
    files = {item.path: item.content for item in support}
    assert "class F8PyEngineContext:" in files["f8_script_api.pyi"]
    assert "msg: Any" in files["f8_dynamic_inputs.pyi"]
    assert "inputMode: Literal[" in files["f8_dynamic_states.pyi"]
    assert "def __getitem__(self, key: str) -> Any" in files["f8_dynamic_states.pyi"]

    editor = EditorSessionService(root=tmp_path / "editor")
    source = (
        "from f8_script_api import F8Inputs, F8PyEngineContext, F8States\n"
        "def onStart(ctx: F8PyEngineContext) -> None:\n"
        "    ctx.log(ctx.states['inputMode'])\n"
        "def onMsg(ctx: F8PyEngineContext, inputs: F8Inputs) -> None:\n"
        "    states: F8States = ctx.states\n"
        "    ctx.emit('out', inputs.msg if states.inputMode else None)\n"
    )
    session = editor.create(CreateEditorSessionRequest(
        language="python", filename="state.py", text=source, support_files=support,
    ))
    try:
        diagnostics = editor.analyze(session.session_id).diagnostics
        assert not any(item.severity == "error" for item in diagnostics), diagnostics
        completion = editor.completion(session.session_id, EditorPositionRequest(line=2, column=8))
        result = completion.result
        items = result if isinstance(result, list) else result.get("items") if isinstance(result, dict) else []
        assert any(isinstance(item, dict) and item.get("label") == "log" for item in items)
        signature = editor.signature_help(session.session_id, EditorPositionRequest(line=5, column=13)).result
        assert isinstance(signature, dict)
        signatures = signature.get("signatures")
        assert isinstance(signatures, list)
        assert any(isinstance(item, dict) and "port: str" in str(item.get("label")) for item in signatures), signature
        second_argument = editor.signature_help(session.session_id, EditorPositionRequest(line=5, column=20)).result
        assert isinstance(second_argument, dict)
        assert second_argument.get("activeParameter") == 1
    finally:
        editor.close_session(session.session_id)


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
    with pytest.raises(ValueError, match="signature help is only available for Python"):
        editor.signature_help(json_session.session_id, EditorPositionRequest(line=0, column=1))
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


def test_state_patch_syncs_deployed_runtime_and_reports_rejection(tmp_path: Path) -> None:
    runtime = HotkeyRuntimeGateway()
    studio = StudioApplication(data_dir=tmp_path, runtime=runtime, service_roots=())
    project = studio.projects.create(CreateProjectRequest(project_id="project1", name="State"))
    node = studio.catalog.create_node(
        CreateCatalogNodeRequest(kind="service", node_id="studio", service_class="f8.pystudio")
    )
    created = studio.projects.patch(project.project_id, PatchRequest(
        request_id="create", expected_graph_revision=0, expected_layout_revision=0,
        operations=(CreateNodeOp(node=node),),
    )).result.document

    async def run() -> None:
        first = await studio.tools.apply_patch(project.project_id, PatchRequest(
            request_id="offline", expected_graph_revision=created.graph_revision,
            expected_layout_revision=created.layout_revision,
            operations=(SetNodeStateOp(node_id="studio", field="tickMs", value=100),),
        ))
        assert first.runtime_errors == ()
        assert runtime.state_calls == []

        studio.jobs.latest = AsyncMock(return_value=DeployJob(
            job_id="job", request_id="deploy", project_id=project.project_id,
            source_graph_revision=first.document.graph_revision, source_semantic_revision="revision",
            status=JobStatus.succeeded, created_at="now", updated_at="now",
            service_results=(ServiceDeployResult(service_id="studio", success=True),),
        ))
        request = PatchRequest(
            request_id="online", expected_graph_revision=first.document.graph_revision,
            expected_layout_revision=first.document.layout_revision,
            operations=(SetNodeStateOp(node_id="studio", field="tickMs", value=200),),
        )
        second = await studio.tools.apply_patch(project.project_id, request)
        assert second.runtime_errors == ()
        assert runtime.state_calls == [("studio", "studio", "tickMs", 200)]
        await studio.tools.apply_patch(project.project_id, request)
        assert len(runtime.state_calls) == 1
        runtime.set_state = AsyncMock(return_value=RuntimeActionResult(success=False, error_message="rejected"))
        third = await studio.tools.apply_patch(project.project_id, PatchRequest(
            request_id="rejected", expected_graph_revision=second.document.graph_revision,
            expected_layout_revision=second.document.layout_revision,
            operations=(SetNodeStateOp(node_id="studio", field="tickMs", value=300),),
        ))
        assert third.runtime_errors == ("studio.tickMs: rejected",)
        stream = await studio.events.open_stream(client_epoch=studio.server_epoch, after_sequence=0)
        committed = [event for event in stream.replay if event.type == "graph.committed"]
        assert len(committed) == 3  # Idempotent retries never duplicate events.
        wire_payload = msgspec.json.decode(msgspec.json.encode(committed[-1].payload))
        assert wire_payload["runtimeErrors"] == ["studio.tickMs: rejected"]
        await studio.events.close_stream(stream.subscription_id)

    asyncio.run(run())
    studio.editor.close()


def test_editor_preserves_existing_files_and_bounds_idle_sessions(tmp_path: Path) -> None:
    root = tmp_path / "editor"
    root.mkdir()
    sentinel = root / "existing.txt"
    sentinel.write_text("owned by another session")
    editor = EditorSessionService(root=root, max_sessions=1, idle_timeout_s=10)
    request = CreateEditorSessionRequest(language="json", filename="test.json", text="{}")
    try:
        first = editor.create(request)
        assert sentinel.read_text() == "owned by another session"
        with pytest.raises(ValueError, match="session limit"):
            editor.create(request)
        with patch("f8studio_server.editor.monotonic", return_value=float("inf")):
            editor.reap_idle()
        with pytest.raises(FileNotFoundError):
            editor.get(first.session_id)
        assert not (root / first.session_id).exists()
        assert editor.create(request).session_id != first.session_id
    finally:
        editor.close()
    assert sentinel.exists()


def test_state_sync_cancellation_publishes_committed_document(tmp_path: Path) -> None:
    runtime = HotkeyRuntimeGateway()
    studio = StudioApplication(data_dir=tmp_path, runtime=runtime, service_roots=())
    project = studio.projects.create(CreateProjectRequest(project_id="cancel-sync", name="Cancel"))
    node = studio.catalog.create_node(
        CreateCatalogNodeRequest(kind="service", node_id="studio", service_class="f8.pystudio")
    )
    document = studio.projects.patch(project.project_id, PatchRequest(
        request_id="seed", expected_graph_revision=0, expected_layout_revision=0,
        operations=(CreateNodeOp(node=node),),
    )).result.document

    async def run() -> None:
        studio.jobs.latest = AsyncMock(return_value=DeployJob(
            job_id="deployed", request_id="deploy", project_id=project.project_id,
            source_graph_revision=document.graph_revision, source_semantic_revision="revision",
            status=JobStatus.succeeded, created_at="now", updated_at="now",
            service_results=(ServiceDeployResult(service_id="studio", success=True),),
        ))
        entered = asyncio.Event()
        hold = asyncio.Event()

        async def sync_state(*_args: object, **_kwargs: object) -> RuntimeActionResult:
            entered.set()
            await hold.wait()
            return RuntimeActionResult(success=True)

        runtime.set_state = AsyncMock(side_effect=sync_state)
        task = asyncio.create_task(studio.tools.apply_patch(project.project_id, PatchRequest(
            request_id="cancelled-sync", expected_graph_revision=document.graph_revision,
            expected_layout_revision=document.layout_revision,
            operations=(SetNodeStateOp(node_id="studio", field="tickMs", value=300),),
        )))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        stream = await studio.events.open_stream(client_epoch=studio.server_epoch, after_sequence=0)
        assert len(stream.replay) == 1
        event = msgspec.json.decode(msgspec.json.encode(stream.replay[0]))
        assert event['payload']['document']['graphRevision'] == document.graph_revision + 1
        assert event['payload']['runtimeErrors'] == ['Runtime state synchronization cancelled']
        assert studio.projects.document(project.project_id).nodes[0].state_values['tickMs'] == 300
        await studio.events.close_stream(stream.subscription_id)

    try:
        asyncio.run(run())
    finally:
        studio.editor.close()
