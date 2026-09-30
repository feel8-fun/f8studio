from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
import httpx
import msgspec
from agent_framework import Agent, Message
from agent_framework.openai import OpenAIChatClient
from fastapi.testclient import TestClient

from f8media_gateway.service import InProcessMediaGateway
from f8pysdk.specs import F8JsonValue, F8RuntimeGraph
from f8studio_core.graph import CreateNodeOp, OperatorNode, PatchRequest, RevisionConflictError, new_document, replace_node_spec
from f8studio_server import create_app
from f8studio_server.agents.models import AgentImage, AgentProviderSummary
from f8studio_server.agents.providers import AgentProviderRegistry
from f8studio_server.agents.provider_settings import CreateProviderConnection, ModelCapabilities, UpdateProviderSettings
from f8studio_server.agents.decisions import SystemOneDecisionClient
from f8studio_server.agents.graph_edits import GraphChanges, build_patch
from f8studio_server.agents.skills import AgentSkillLibrary
from f8studio_server.application import StudioApplication
from f8studio_server.local_integration import DetectModdingTargetRequest, LocalIntegrationService
from f8studio_server.models import CreateCatalogNodeRequest, CreateProjectRequest
from f8studio_server.project_repository import ProjectRepository
from f8studio_server.models import (
    RuntimeActionResult,
    RuntimeStateField,
    ServiceDeployResult,
    ServiceRuntimeStatus,
)
from f8studio_server.runtime import RuntimeMonitorCallback


@pytest.fixture
def engine_service_root(tmp_path: Path) -> Path:
    """Use real engine specs without requiring installed native/runtime bundles."""
    from f8pyengine.pyengine_node_registry import register_pyengine_specs
    from f8pysdk.registry import Registry

    root = tmp_path / "services" / "f8.pyengine"
    root.mkdir(parents=True)
    describe = register_pyengine_specs(Registry()).describe("f8.pyengine")
    (root / "describe.json").write_bytes(msgspec.json.encode(describe))
    (root / "service.yml").write_text(json.dumps({
        "schemaVersion": "f8serviceEntry/1", "serviceClass": "f8.pyengine",
        "label": "PyEngine", "version": "0.0.1",
        "launch": {"command": sys.executable, "args": ["-m", "f8pyengine.main"], "workdir": str(root)},
    }), encoding="utf-8")
    return root


class AgentRuntimeGateway:
    def __init__(self, *, deploy_success: bool = True) -> None:
        self.deploy_calls: list[str] = []
        self.deploy_success = deploy_success

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
        return ServiceDeployResult(
            service_id=service_id,
            success=self.deploy_success,
            error_message="runtime rejected graph" if not self.deploy_success else "",
        )

    async def status(self, service_id: str) -> ServiceRuntimeStatus:
        return ServiceRuntimeStatus(
            service_id=service_id,
            service_class="f8.pystudio",
            runtime_instance_id="agent-test-runtime",
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
        return None


class ScriptEditingProvider(AgentProviderRegistry):
    def summaries(self) -> tuple[AgentProviderSummary, ...]:
        return (AgentProviderSummary(
            provider_id="script_test", display_name="Script test", models=("test",), configured=True,
        ),)

    async def run_with_tools(
        self, *, provider_id: str, model_id: str, prompt: str,
        tools: Sequence[Callable[..., Awaitable[str]]],
        images: Sequence[AgentImage] = (), reasoning_effort: str | None = None,
    ) -> str:
        self.validate_selection(provider_id, model_id)
        del prompt
        functions = {tool.__name__: tool for tool in tools}
        assert "graph_python" in json.loads(await functions["skills_list"]())
        assert "code_read" in await functions["skill_read"]("graph_python")
        catalog = json.loads(await functions["catalog_read"]())
        assert any(item["operatorClass"] == "f8.phase" for item in catalog["operators"])
        matches = json.loads(await functions["catalog_search"]("cosine"))
        assert any(item["operatorClass"] == "f8.cosine" and item["serviceClass"] == "f8.pyengine"
                   for item in matches["matches"])
        spec = json.loads(await functions["catalog_operator"]("f8.pyengine", "f8.cosine"))
        assert any(port["name"] == "phase" for port in spec["dataInPorts"])
        phase = json.loads(await functions["catalog_create_node"](
            "agent_phase_preview", "f8.pyengine", "f8.phase", "engine", "Phase",
        ))
        assert phase["operatorClass"] == "f8.phase"
        assert any(port["name"] == "phase" and port["direction"] == "output" for port in phase["ports"])
        graph = json.loads(await functions["graph_read"]())
        assert any(node["nodeId"] == "script" and node["serviceId"] == "engine" for node in graph["nodes"])
        assert json.loads(await functions["graph_node"]("script"))["nodeId"] == "script"
        node_preview = json.loads(await functions["graph_preview_patch"](json.dumps({
            "requestId": "preview-phase", "expectedGraphRevision": graph["graphRevision"],
            "expectedLayoutRevision": graph["layoutRevision"],
            "operations": [{"op": "createNode", "node": phase}],
        })))
        assert node_preview["graphRevision"] == graph["graphRevision"] + 1
        patch = {
            "requestId": "model-rename", "expectedGraphRevision": graph["graphRevision"],
            "expectedLayoutRevision": graph["layoutRevision"],
            "operations": [{"op": "renameNode", "nodeId": "script", "name": "Edited script"}],
        }
        preview = json.loads(await functions["graph_preview_patch"](json.dumps(patch)))
        assert preview["graphRevision"] == graph["graphRevision"] + 1
        await functions["graph_apply_patch"](json.dumps(patch))
        source = json.loads(await functions["code_read"]("script"))
        code = "def onStart(ctx):\n    ctx.log('edited')\n"
        analysis = json.loads(await functions["code_analyze"]("script", code))
        assert analysis["engine"] == "basedpyright"
        await functions["code_write"](
            "script", source["graphRevision"], source["codeSha256"], code,
        )
        await functions["graph_validate"]()
        return "Renamed the node and updated its Python code."


def _session(client: TestClient, session_id: str) -> dict[str, object]:
    response = client.get(f"/api/agents/sessions/{session_id}")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


def _wait_for_status(
    client: TestClient,
    session_id: str,
    statuses: set[str],
    *,
    timeout_s: float = 3.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        payload = _session(client, session_id)
        if payload.get("status") in statuses:
            return payload
        time.sleep(0.01)
    raise AssertionError(f"agent session did not reach {statuses}: {_session(client, session_id)}")


def _approve_pending(client: TestClient, session_id: str, *, previous_id: str = "") -> str:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        payload = _session(client, session_id)
        approval = payload.get("approval")
        if isinstance(approval, dict):
            approval_id = approval.get("approvalId")
            arguments_hash = approval.get("argumentsHash")
            if (
                approval.get("status") == "pending"
                and isinstance(approval_id, str)
                and approval_id != previous_id
                and isinstance(arguments_hash, str)
            ):
                response = client.post(
                    f"/api/agents/sessions/{session_id}/approvals/{approval_id}",
                    json={"approved": True, "argumentsHash": arguments_hash},
                )
                assert response.status_code == 200, response.text
                return approval_id
        time.sleep(0.01)
    raise AssertionError("agent did not request approval")


def _create_session(client: TestClient, project_id: str) -> str:
    response = client.post(
        "/api/agents/sessions",
        json={
            "projectId": project_id,
            "title": "Build graph",
            "providerId": "deterministic",
            "modelId": "graph-builder-v1",
        },
    )
    assert response.status_code == 201
    return str(response.json()["sessionId"])


def test_deterministic_agent_builds_validates_deploys_and_publishes_graph_event(tmp_path: Path) -> None:
    runtime = AgentRuntimeGateway()
    studio = StudioApplication(
        data_dir=tmp_path / "data",
        runtime=runtime,
        service_roots=(),
        media_gateway=InProcessMediaGateway(),
    )
    app = create_app(web_dist=tmp_path, application=studio)

    with TestClient(app) as client:
        created = client.post("/api/projects", json={"projectId": "agent_project", "name": "Agent"})
        assert created.status_code == 201
        session_id = _create_session(client, "agent_project")

        with client.websocket_connect("/api/events") as websocket:
            snapshot = websocket.receive_json()
            assert snapshot["type"] == "stream.hello"
            started = client.post(
                f"/api/agents/sessions/{session_id}/runs",
                json={"prompt": "Build a controllable value graph and deploy it"},
            )
            assert started.status_code == 202
            first_approval = _approve_pending(client, session_id)

            committed_event = None
            for _index in range(20):
                event = websocket.receive_json()
                if event.get("type") == "graph.committed":
                    committed_event = event
                    break
            assert committed_event is not None
            assert committed_event["scope"] == "project:agent_project"

            second_approval = _approve_pending(client, session_id, previous_id=first_approval)
            assert second_approval != first_approval
            finished = _wait_for_status(client, session_id, {"succeeded", "failed", "cancelled"})

        assert finished["status"] == "succeeded", finished
        assert runtime.deploy_calls == ["studio"]
        project = client.get("/api/projects/agent_project").json()
        nodes = project["document"]["nodes"]
        assert {node["kind"] for node in nodes} == {"service", "operator"}
        assert any(node.get("operatorClass") == "f8.value_stepper" for node in nodes)
        assert len(finished["artifacts"]) == 3
        assert finished["messages"][-1]["role"] == "assistant"
        assert "graph.apply_patch" in [call["toolName"] for call in finished["toolCalls"]]


def test_agent_session_auto_title_rename_and_delete(tmp_path: Path) -> None:
    studio = StudioApplication(
        data_dir=tmp_path / "data", runtime=AgentRuntimeGateway(), service_roots=(),
        media_gateway=InProcessMediaGateway(),
    )
    with TestClient(create_app(web_dist=tmp_path, application=studio)) as client:
        assert client.post("/api/projects", json={"projectId": "session_lifecycle", "name": "Lifecycle"}).status_code == 201
        created = client.post("/api/agents/sessions", json={"projectId": "session_lifecycle"})
        assert created.status_code == 201
        session_id = created.json()["sessionId"]
        started = client.post(f"/api/agents/sessions/{session_id}/runs", json={
            "prompt": "Build a cosine wave and connect it to the visualizers\nKeep it at 1 Hz",
        })
        assert started.status_code == 202
        assert started.json()["title"] == "Build a cosine wave and connect it to the visualizers"
        assert client.delete(f"/api/agents/sessions/{session_id}").status_code == 422
        assert client.put(f"/api/agents/sessions/{session_id}", json={"title": "Manual title"}).status_code == 422
        assert client.delete(f"/api/agents/sessions/{session_id}/runs/current").status_code == 200

        renamed = client.put(f"/api/agents/sessions/{session_id}", json={"title": "  My cosine   graph  "})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "My cosine graph"
        assert client.get(f"/api/agents/sessions/{session_id}").json()["title"] == "My cosine graph"
        assert client.get("/api/agents/sessions?project_id=session_lifecycle").json()[0]["title"] == "My cosine graph"
        assert client.put(f"/api/agents/sessions/{session_id}", json={"title": " "}).status_code == 422
        assert client.delete(f"/api/agents/sessions/{session_id}").status_code == 204
        assert client.get(f"/api/agents/sessions/{session_id}").status_code == 404
        assert client.get("/api/agents/sessions?project_id=session_lifecycle").json() == []
        assert client.get("/api/projects/session_lifecycle").status_code == 200

        second = client.post("/api/agents/sessions", json={"projectId": "session_lifecycle"})
        second_id = second.json()["sessionId"]
        assert client.put(f"/api/agents/sessions/{second_id}", json={"title": "Studio agent"}).status_code == 200
        manual_run = client.post(f"/api/agents/sessions/{second_id}/runs", json={"prompt": "Build another graph"})
        assert manual_run.json()["title"] == "Studio agent"
        client.delete(f"/api/agents/sessions/{second_id}/runs/current")
        assert client.delete(f"/api/agents/sessions/{second_id}").status_code == 204


def test_denied_approval_finishes_run_without_stuck_waiting(tmp_path: Path) -> None:
    studio = StudioApplication(data_dir=tmp_path, runtime=AgentRuntimeGateway(), service_roots=(),
                               media_gateway=InProcessMediaGateway())
    with TestClient(create_app(web_dist=tmp_path, application=studio)) as client:
        client.post("/api/projects", json={"projectId": "denied", "name": "Denied"})
        session_id = _create_session(client, "denied")
        client.post(f"/api/agents/sessions/{session_id}/runs", json={"prompt": "Build a value graph"})
        waiting = _wait_for_status(client, session_id, {"waiting_for_approval"})
        approval = waiting["approval"]
        assert isinstance(approval, dict)
        response = client.post(f"/api/agents/sessions/{session_id}/approvals/{approval['approvalId']}",
                               json={"approved": False, "argumentsHash": approval["argumentsHash"]})
        assert response.status_code == 200
        assert response.json()["status"] == "running"
        finished = _wait_for_status(client, session_id, {"cancelled"})
        assert finished["approval"]["status"] == "denied"
        assert all(call["status"] not in {"running", "queued", "waiting_for_approval"} for call in finished["toolCalls"])


def test_run_timeout_cleans_pending_approval_and_tool(tmp_path: Path, monkeypatch) -> None:
    deadlines: list[asyncio.Timeout] = []

    def controlled_timeout(delay: float) -> asyncio.Timeout:
        assert delay == 300.0
        deadline = asyncio.timeout(None)
        deadlines.append(deadline)
        return deadline

    monkeypatch.setattr("f8studio_server.agents.service.run_timeout", controlled_timeout)
    studio = StudioApplication(data_dir=tmp_path, runtime=AgentRuntimeGateway(), service_roots=(),
                               media_gateway=InProcessMediaGateway())
    with TestClient(create_app(web_dist=tmp_path, application=studio)) as client:
        client.post("/api/projects", json={"projectId": "timeout", "name": "Timeout"})
        session_id = _create_session(client, "timeout")
        client.post(f"/api/agents/sessions/{session_id}/runs", json={"prompt": "Build a value graph"})
        _wait_for_status(client, session_id, {"waiting_for_approval"})
        async def expire_run() -> None:
            assert len(deadlines) == 1
            deadlines[0].reschedule(asyncio.get_running_loop().time())

        assert client.portal is not None
        client.portal.call(expire_run)
        finished = _wait_for_status(client, session_id, {"failed"})
        assert finished["approval"]["status"] == "expired"
        assert all(call["status"] not in {"running", "queued", "waiting_for_approval"} for call in finished["toolCalls"])
        assert studio.agents._execution.pending_count == 0


def test_agent_approval_is_invalidated_when_another_client_changes_revision(tmp_path: Path) -> None:
    studio = StudioApplication(
        data_dir=tmp_path / "data",
        runtime=AgentRuntimeGateway(),
        service_roots=(),
        media_gateway=InProcessMediaGateway(),
    )
    app = create_app(web_dist=tmp_path, application=studio)

    with TestClient(app) as client:
        client.post("/api/projects", json={"projectId": "conflict_project", "name": "Conflict"})
        session_id = _create_session(client, "conflict_project")
        client.post(
            f"/api/agents/sessions/{session_id}/runs",
            json={"prompt": "Build a value graph"},
        )
        waiting = _wait_for_status(client, session_id, {"waiting_for_approval"})
        approval = waiting["approval"]
        assert isinstance(approval, dict)
        tool_calls = waiting["toolCalls"]
        assert isinstance(tool_calls, list)
        pending_call = next(call for call in tool_calls if call["status"] == "waiting_for_approval")
        assert approval["toolCallId"] == pending_call["toolCallId"]
        assert approval["argumentsHash"] == pending_call["argumentsHash"]
        assert approval["targetGraphRevision"] == pending_call["targetGraphRevision"] == 0
        assert datetime.fromisoformat(str(approval["expiresAt"])) > datetime.now(UTC)

        wrong_hash = client.post(
            f"/api/agents/sessions/{session_id}/approvals/{approval['approvalId']}",
            json={"approved": True, "argumentsHash": "0" * 64},
        )
        assert wrong_hash.status_code == 422
        assert "argumentsHash does not match" in wrong_hash.text
        assert _session(client, session_id)["approval"]["status"] == "pending"

        service = client.post(
            "/api/catalog/nodes",
            json={"kind": "service", "nodeId": "manual_studio", "serviceClass": "f8.pystudio"},
        ).json()
        changed = client.post(
            "/api/projects/conflict_project/patch",
            json={
                "requestId": "manual-change",
                "expectedGraphRevision": 0,
                "expectedLayoutRevision": 0,
                "operations": [{"op": "createNode", "node": service}],
            },
        )
        assert changed.status_code == 200

        rejected = client.post(
            f"/api/agents/sessions/{session_id}/approvals/{approval['approvalId']}",
            json={"approved": True, "argumentsHash": approval["argumentsHash"]},
        )
        assert rejected.status_code == 409
        assert rejected.json()["detail"]["code"] == "revision_conflict"
        failed = _wait_for_status(client, session_id, {"failed"})
        assert "RevisionConflictError" in str(failed["errorMessage"])
        assert failed["tracebackId"]
        project = client.get("/api/projects/conflict_project").json()
        assert len(project["document"]["nodes"]) == 1


def test_agent_provider_api_never_exposes_server_credentials(tmp_path: Path, monkeypatch) -> None:
    secrets = {
        "OPENAI_API_KEY": "server-only-openai-key",
        "ANTHROPIC_API_KEY": "server-only-anthropic-key",
        "GEMINI_API_KEY": "server-only-gemini-key",
    }
    for name, secret in secrets.items():
        monkeypatch.setenv(name, secret)
    monkeypatch.setenv("F8STUDIO_OLLAMA_MODEL", "qwen-local")
    app = create_app(
        web_dist=tmp_path,
        data_dir=tmp_path / "data",
        runtime=AgentRuntimeGateway(),
        service_roots=(),
        media_gateway=InProcessMediaGateway(),
    )

    with TestClient(app) as client:
        providers = client.get("/api/agents/providers")
        capabilities = client.get("/api/capabilities")

    assert providers.status_code == 200
    for secret in secrets.values():
        assert secret not in providers.text
    provider_status = {item["providerId"]: item["configured"] for item in providers.json()}
    assert provider_status == {
        "deterministic": True,
        "openai": True,
        "anthropic": True,
        "google_gemini": True,
        "ollama": True,
    }
    assert capabilities.json()["capabilities"]["agent_tools"] is True


def test_provider_settings_api_enables_session_creation_without_restart(tmp_path: Path) -> None:
    app = create_app(web_dist=tmp_path, data_dir=tmp_path / "data", runtime=AgentRuntimeGateway(), service_roots=(), media_gateway=InProcessMediaGateway())
    with TestClient(app) as client:
        saved = client.put("/api/agents/providers/openai/settings", json={"model": "test-model", "apiKey": "private-settings-key"})
        assert saved.status_code == 200
        assert saved.json()["configured"] is True
        assert "private-settings-key" not in saved.text
        settings = client.get("/api/agents/providers/settings")
        assert settings.status_code == 200
        assert "private-settings-key" not in settings.text
        client.post("/api/projects", json={"projectId": "settings_test", "name": "Settings test"})
        session = client.post("/api/agents/sessions", json={"projectId": "settings_test", "title": "Configured model", "providerId": "openai", "modelId": "test-model"})
        assert session.status_code == 201
        assert client.put("/api/agents/providers/openai/settings", json={"model": ""}).status_code == 422
        assert client.delete("/api/agents/connections/openai").status_code == 204
        assert "openai" not in {item["providerId"] for item in client.get("/api/agents/providers/settings").json()}
        assert client.get(f"/api/agents/sessions/{session.json()['sessionId']}").status_code == 200
        assert client.post("/api/agents/sessions", json={"projectId": "settings_test", "providerId": "openai", "modelId": "test-model"}).status_code == 422


def test_named_connection_api_exposes_models_to_sessions(tmp_path: Path) -> None:
    app = create_app(web_dist=tmp_path, data_dir=tmp_path / "data", runtime=AgentRuntimeGateway(), service_roots=(), media_gateway=InProcessMediaGateway())
    with TestClient(app) as client:
        created = client.post("/api/agents/connections", json={
            "displayName": "Secondary OpenAI host", "protocol": "openai_chat",
            "endpoint": "https://secondary.example/v1", "apiKey": "private-key",
            "model": "model-a", "models": ["model-a", "model-b"],
        })
        assert created.status_code == 201
        connection_id = created.json()["providerId"]
        assert created.json()["models"] == ["model-a", "model-b"]
        assert "private-key" not in created.text
        listed = client.get("/api/agents/providers").json()
        assert next(item for item in listed if item["providerId"] == connection_id)["models"] == ["model-a", "model-b"]
        client.post("/api/projects", json={"projectId": "named_connection", "name": "Named connection"})
        session = client.post("/api/agents/sessions", json={
            "projectId": "named_connection", "providerId": connection_id, "modelId": "model-b",
        })
        assert session.status_code == 201
        assert client.delete(f"/api/agents/connections/{connection_id}").status_code == 204
        assert client.get(f"/api/agents/sessions/{session.json()['sessionId']}").status_code == 200


def test_existing_session_switches_model_and_delivers_image_to_provider(tmp_path: Path) -> None:
    class ImageProvider(AgentProviderRegistry):
        def __init__(self) -> None:
            self.received_images: tuple[AgentImage, ...] = ()

        def summaries(self) -> tuple[AgentProviderSummary, ...]:
            return (
                AgentProviderSummary(provider_id="deterministic", display_name="Graph", models=("graph-builder-v1",), configured=True, deterministic=True),
                AgentProviderSummary(provider_id="vision_test", display_name="Vision", models=("model-a",), configured=True, supports_images=True),
            )

        def supports_image(self, provider_id: str, model_id: str) -> bool:
            return provider_id == "vision_test" and model_id == "model-b"

        async def run_with_tools(
            self, *, provider_id: str, model_id: str, prompt: str,
            tools: Sequence[Callable[..., Awaitable[str]]], images: Sequence[AgentImage] = (),
            reasoning_effort: str | None = None,
        ) -> str:
            del tools
            assert provider_id == "vision_test" and model_id == "model-b"
            assert "Inspect" in prompt
            await asyncio.sleep(0.1)
            self.received_images = tuple(images)
            return "Image received"

    studio = StudioApplication(data_dir=tmp_path / "data", runtime=AgentRuntimeGateway(), service_roots=(), media_gateway=InProcessMediaGateway())
    provider = ImageProvider()
    studio.agents._providers = provider
    with TestClient(create_app(web_dist=tmp_path, application=studio)) as client:
        assert client.post("/api/projects", json={"projectId": "image_session", "name": "Image"}).status_code == 201
        session_id = _create_session(client, "image_session")
        model_url = f"/api/agents/sessions/{session_id}/model"
        assert client.put(model_url, json={"providerId": "vision_test", "modelId": " "}).status_code == 422
        selected = client.put(model_url, json={"providerId": "vision_test", "modelId": "model-b"})
        assert selected.status_code == 200
        assert selected.json()["modelId"] == "model-b"
        image = {"name": "graph.png", "dataUrl": "data:image/png;base64,iVBORw0KGgo="}
        run_url = f"/api/agents/sessions/{session_id}/runs"
        assert client.post(run_url, json={"prompt": "Inspect", "images": [{**image, "dataUrl": "data:image/png;base64,bad!"}]}).status_code == 422
        started = client.post(run_url, json={"prompt": "Inspect", "images": [image]})
        assert started.status_code == 202
        assert client.put(model_url, json={"providerId": "deterministic", "modelId": "graph-builder-v1"}).status_code == 422
        finished = _wait_for_status(client, session_id, {"succeeded"})
        assert provider.received_images == (AgentImage(name="graph.png", data_url=image["dataUrl"]),)
        assert finished["messages"][0]["images"] == [image]
        assert finished["messages"][0]["modelId"] == "model-b"
        assert finished["messages"][1]["content"] == "Image received"
        assert client.put(model_url, json={"providerId": "deterministic", "modelId": "graph-builder-v1"}).status_code == 200
        assert len(client.get(f"/api/agents/sessions/{session_id}").json()["messages"]) == 2


def test_run_api_passes_optional_reasoning_effort_to_provider(tmp_path: Path) -> None:
    class ReasoningProvider(AgentProviderRegistry):
        def __init__(self) -> None:
            self.efforts: list[str | None] = []

        def summaries(self) -> tuple[AgentProviderSummary, ...]:
            return (AgentProviderSummary(
                provider_id="reasoning_test", display_name="Reasoning", models=("model-a",), configured=True,
            ),)

        def supports_image(self, provider_id: str, model_id: str) -> bool:
            del provider_id, model_id
            return False

        async def run_with_tools(
            self, *, provider_id: str, model_id: str, prompt: str,
            tools: Sequence[Callable[..., Awaitable[str]]],
            images: Sequence[AgentImage] = (),
            reasoning_effort: str | None = None,
        ) -> str:
            del provider_id, model_id, prompt, tools, images
            self.efforts.append(reasoning_effort)
            return "Done"

    studio = StudioApplication(data_dir=tmp_path / "data", runtime=AgentRuntimeGateway(), service_roots=(), media_gateway=InProcessMediaGateway())
    provider = ReasoningProvider()
    studio.agents._providers = provider
    with TestClient(create_app(web_dist=tmp_path, application=studio)) as client:
        assert client.post("/api/projects", json={"projectId": "reasoning_session", "name": "Reasoning"}).status_code == 201
        session = client.post("/api/agents/sessions", json={
            "projectId": "reasoning_session", "providerId": "reasoning_test", "modelId": "model-a",
        })
        assert session.status_code == 201
        session_id = session.json()["sessionId"]
        run_url = f"/api/agents/sessions/{session_id}/runs"
        assert client.post(run_url, json={"prompt": "Inspect", "reasoningEffort": "ultra"}).status_code == 422
        assert client.post(run_url, json={"prompt": "Inspect", "reasoningEffort": "high"}).status_code == 202
        _wait_for_status(client, session_id, {"succeeded"})
        assert client.post(run_url, json={"prompt": "Inspect again"}).status_code == 202
        _wait_for_status(client, session_id, {"succeeded"})
        assert provider.efforts == ["high", None]


def test_openai_provider_sends_image_as_multimodal_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_framework
    import agent_framework.openai

    received: list[Message] = []

    class FakeResponse:
        text = "Done"

    class FakeAgent:
        def __init__(self, client: object, **kwargs: object) -> None:
            del client, kwargs

        async def run(self, message: Message, *, options: object) -> object:
            del options
            received.append(message)
            return FakeResponse()

    monkeypatch.setattr(agent_framework, "Agent", FakeAgent)
    monkeypatch.setattr(agent_framework.openai, "OpenAIChatClient", lambda **kwargs: object())
    registry = AgentProviderRegistry(tmp_path / "providers.json")
    registry.update_settings("openai", UpdateProviderSettings(
        model="test-model", api_key="test-key",
        model_capabilities=(ModelCapabilities(model_id="test-model", image_input=True),),
    ))
    result = asyncio.run(registry.run_with_tools(
        provider_id="openai", model_id="test-model", prompt="Inspect",
        tools=(), images=(AgentImage(name="graph.png", data_url="data:image/png;base64,iVBORw0KGgo="),),
    ))
    assert result == "Done"
    assert received[0].role == "user"
    assert [item.type for item in received[0].contents] == ["text", "data"]
    assert received[0].contents[1].media_type == "image/png"


@pytest.mark.parametrize("protocol, expected", [
    ("openai_responses", {"max_tokens": 4096, "store": False, "reasoning": {"effort": "high"}}),
    ("openai_chat", {"max_tokens": 4096, "reasoning_effort": "high"}),
    ("anthropic", {"max_tokens": 4096, "output_config": {"effort": "high"}}),
])
def test_reasoning_effort_reaches_the_selected_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, protocol: str, expected: dict[str, object],
) -> None:
    import agent_framework
    import agent_framework.anthropic
    import agent_framework.openai

    received: list[dict[str, object]] = []

    class FakeResponse:
        text = "Done"

    class FakeAgent:
        def __init__(self, client: object, **kwargs: object) -> None:
            del client, kwargs

        async def run(self, message: object, *, options: dict[str, object]) -> FakeResponse:
            del message
            received.append(options)
            return FakeResponse()

    monkeypatch.setattr(agent_framework, "Agent", FakeAgent)
    monkeypatch.setattr(agent_framework.openai, "OpenAIChatClient", lambda **kwargs: object())
    monkeypatch.setattr(agent_framework.openai, "OpenAIChatCompletionClient", lambda **kwargs: object())
    monkeypatch.setattr(agent_framework.anthropic, "AnthropicClient", lambda **kwargs: object())
    registry = AgentProviderRegistry(tmp_path / "providers.json")
    if protocol == "openai_responses":
        registry.update_settings("openai", UpdateProviderSettings(model="test-model", api_key="key"))
        provider_id = "openai"
    elif protocol == "anthropic":
        registry.update_settings("anthropic", UpdateProviderSettings(model="test-model", api_key="key"))
        provider_id = "anthropic"
    else:
        connection = registry.create_connection(CreateProviderConnection(
            display_name="Chat host", protocol="openai_chat", endpoint="https://example.com/v1",
            api_key="key", model="test-model",
        ))
        provider_id = connection.provider_id
    result = asyncio.run(registry.run_with_tools(
        provider_id=provider_id, model_id="test-model", prompt="Inspect", tools=(), reasoning_effort="high",
    ))
    assert result == "Done"
    assert received == [expected]


def test_named_openai_compatible_connections_use_their_own_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import agent_framework
    import agent_framework.openai

    clients: list[dict[str, str]] = []

    class FakeResponse:
        text = "Done"

    class FakeAgent:
        def __init__(self, client: object, **kwargs: object) -> None:
            del client, kwargs

        async def run(self, prompt: str, *, options: object) -> FakeResponse:
            del prompt, options
            return FakeResponse()

    def fake_client(*, model: str, api_key: str, base_url: str) -> object:
        clients.append({"model": model, "api_key": api_key, "base_url": base_url})
        return object()

    monkeypatch.setattr(agent_framework, "Agent", FakeAgent)
    monkeypatch.setattr(agent_framework.openai, "OpenAIChatCompletionClient", fake_client)
    registry = AgentProviderRegistry(tmp_path / "providers.json")
    first = registry.create_connection(CreateProviderConnection(
        display_name="Work", protocol="openai_chat", endpoint="https://work.example/v1",
        api_key="work-key", model="model-a",
    ))
    second = registry.create_connection(CreateProviderConnection(
        display_name="Personal", protocol="openai_chat", endpoint="https://personal.example/v1",
        api_key="personal-key", model="model-b",
    ))
    for setting in (first, second):
        assert asyncio.run(registry.run_with_tools(
            provider_id=setting.provider_id, model_id=setting.model, prompt="Inspect", tools=(),
        )) == "Done"
    assert clients == [
        {"model": "model-a", "api_key": "work-key", "base_url": "https://work.example/v1"},
        {"model": "model-b", "api_key": "personal-key", "base_url": "https://personal.example/v1"},
    ]


@pytest.mark.parametrize("upstream_status, expected_status", [(200, 200), (429, 429), (401, 502)])
def test_decision_gateway_uses_saved_credentials_and_maps_upstream_errors(tmp_path: Path, upstream_status: int, expected_status: int) -> None:
    studio = StudioApplication(data_dir=tmp_path / "data", runtime=AgentRuntimeGateway(), service_roots=(), media_gateway=InProcessMediaGateway())
    asyncio.run(studio.decisions.close())
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer decision-secret"
        return httpx.Response(upstream_status, json={"model": "jev-test", "answers": {"q": {"type": "noul", "noul": 0.7}}, "usage": {"input_tokens": 10, "output_tokens": 0}})
    studio.decisions = SystemOneDecisionClient(studio.agents._providers, transport=httpx.MockTransport(respond))
    with TestClient(create_app(web_dist=tmp_path, application=studio)) as client:
        configured = client.put("/api/agents/providers/typesafe/settings", json={"model": "jev-latest", "endpoint": "https://api.typesafe.ai/v1", "apiKey": "decision-secret"})
        assert configured.status_code == 200
        assert configured.json()["kind"] == "decision"
        result = client.post("/api/decisions/evaluate", json={"state": "observation", "questions": {"q": {"type": "noul", "instructions": "Relevant?"}}})
        assert result.status_code == expected_status
        assert "decision-secret" not in result.text
        if expected_status == 200:
            assert result.json()["answers"]["q"]["noul"] == 0.7
        elif expected_status == 429:
            assert result.headers["Retry-After"] == "2"
    assert studio.decisions._http.is_closed


def test_agent_run_fails_with_tool_context_when_deployment_fails(tmp_path: Path) -> None:
    studio = StudioApplication(
        data_dir=tmp_path / "data",
        runtime=AgentRuntimeGateway(deploy_success=False),
        service_roots=(),
        media_gateway=InProcessMediaGateway(),
    )
    app = create_app(web_dist=tmp_path, application=studio)

    with TestClient(app) as client:
        client.post("/api/projects", json={"projectId": "failed_deploy", "name": "Failed deploy"})
        session_id = _create_session(client, "failed_deploy")
        client.post(
            f"/api/agents/sessions/{session_id}/runs",
            json={"prompt": "Build and deploy a value graph"},
        )
        first_approval = _approve_pending(client, session_id)
        _approve_pending(client, session_id, previous_id=first_approval)
        failed = _wait_for_status(client, session_id, {"failed"})

    tool_calls = failed["toolCalls"]
    assert isinstance(tool_calls, list)
    deploy_call = next(call for call in tool_calls if call["toolName"] == "project.deploy")
    assert deploy_call["status"] == "failed"
    assert "finished with failed" in deploy_call["errorMessage"]
    assert deploy_call["tracebackId"]
    assert "RuntimeError" in str(failed["errorMessage"])
    assert failed["tracebackId"]


def test_cancelling_pending_agent_run_cancels_approval_and_tool(tmp_path: Path) -> None:
    studio = StudioApplication(
        data_dir=tmp_path / "data",
        runtime=AgentRuntimeGateway(),
        service_roots=(),
        media_gateway=InProcessMediaGateway(),
    )
    app = create_app(web_dist=tmp_path, application=studio)

    with TestClient(app) as client:
        client.post("/api/projects", json={"projectId": "cancel_agent", "name": "Cancel agent"})
        session_id = _create_session(client, "cancel_agent")
        client.post(
            f"/api/agents/sessions/{session_id}/runs",
            json={"prompt": "Build a value graph"},
        )
        waiting = _wait_for_status(client, session_id, {"waiting_for_approval"})
        approval = waiting["approval"]
        assert isinstance(approval, dict)

        response = client.delete(f"/api/agents/sessions/{session_id}/runs/current")
        assert response.status_code == 200
        cancelled = response.json()
        assert cancelled["status"] == "cancelled"
        assert cancelled["approval"]["status"] == "cancelled"
        call = next(item for item in cancelled["toolCalls"] if item["toolCallId"] == approval["toolCallId"])
        assert call["status"] == "cancelled"
        assert "not rolled back" in cancelled["errorMessage"]
        project = client.get("/api/projects/cancel_agent").json()
        assert project["document"]["graphRevision"] == 0


def test_model_tool_loop_edits_graph_and_python_node_code(tmp_path: Path, engine_service_root: Path) -> None:
    studio = StudioApplication(
        data_dir=tmp_path / "data", runtime=AgentRuntimeGateway(), service_roots=(engine_service_root,),
        media_gateway=InProcessMediaGateway(),
    )
    studio.agents._providers = ScriptEditingProvider()
    studio.projects.create(CreateProjectRequest(project_id="script_project", name="Script project"))
    service = studio.catalog.create_node(CreateCatalogNodeRequest(
        kind="service", node_id="engine", service_class="f8.pyengine",
    ))
    script = studio.catalog.create_node(CreateCatalogNodeRequest(
        kind="operator", node_id="script", service_id="engine", service_class="f8.pyengine",
        operator_class="f8.python_script",
    ))
    studio.projects.patch("script_project", PatchRequest(
        request_id="seed-script", expected_graph_revision=0, expected_layout_revision=0,
        operations=(CreateNodeOp(node=service), CreateNodeOp(node=script)),
    ))
    app = create_app(web_dist=tmp_path, application=studio)
    with TestClient(app) as client:
        response = client.post("/api/agents/sessions", json={
            "projectId": "script_project", "providerId": "script_test", "modelId": "test",
        })
        assert response.status_code == 201
        session_id = str(response.json()["sessionId"])
        Agent(OpenAIChatClient(model="test", api_key="test"), tools=studio.agents._model_tools(studio.agents.get(session_id)))
        started = client.post(f"/api/agents/sessions/{session_id}/runs", json={"prompt": "Edit the script"})
        assert started.status_code == 202
        first = _approve_pending(client, session_id)
        second = _approve_pending(client, session_id, previous_id=first)
        assert second != first
        finished = _wait_for_status(client, session_id, {"succeeded", "failed"}, timeout_s=15)
        project = client.get("/api/projects/script_project").json()

    assert finished["status"] == "succeeded", finished
    edited = next(node for node in project["document"]["nodes"] if node["nodeId"] == "script")
    assert edited["name"] == "Edited script"
    assert edited["stateValues"]["code"] == "def onStart(ctx):\n    ctx.log('edited')\n"
    assert {call["toolName"] for call in finished["toolCalls"]} >= {
        "graph.preview_patch", "graph.apply_patch", "code.read", "code.analyze", "code.write", "graph.validate",
    }
    functions = {tool.__name__: tool for tool in studio.agents._model_tools(studio.agents.get(session_id))}
    with pytest.raises(RevisionConflictError, match="code changed since read"):
        asyncio.run(functions["code_write"]("script", 1, "0" * 64, "def onStart(ctx): pass\n"))
    bypass = {
        "requestId": "bypass-code-check", "expectedGraphRevision": project["document"]["graphRevision"],
        "expectedLayoutRevision": project["document"]["layoutRevision"],
        "operations": [{"op": "setNodeState", "nodeId": "script", "field": "code", "value": "pass"}],
    }
    with pytest.raises(ValueError, match="use code_read"):
        asyncio.run(functions["graph_apply_patch"](json.dumps(bypass)))
    with pytest.raises(ValueError, match="previewed in this run"):
        asyncio.run(functions["modding_apply_unity_install"]("unseen-plan"))


def test_agent_skill_library_loads_local_skills_without_path_traversal(tmp_path: Path) -> None:
    library = AgentSkillLibrary(user_root=tmp_path / "skills")
    assert {"graph_python", "unity_modding"} <= set(library.list())
    custom = tmp_path / "skills" / "specific_game"
    custom.mkdir()
    (custom / "SKILL.md").write_text("# Specific Game\n", encoding="utf-8")
    assert "specific_game" in library.list()
    assert library.read("specific_game") == "# Specific Game\n"
    with pytest.raises(ValueError, match="invalid agent skill id"):
        library.read("../specific_game")


def test_parallel_model_tools_preserve_every_call_and_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        from f8studio_server.agents.models import CreateAgentSessionRequest

        studio = StudioApplication(
            data_dir=tmp_path / "data", runtime=AgentRuntimeGateway(), service_roots=(),
            media_gateway=InProcessMediaGateway(),
        )
        studio.projects.create(CreateProjectRequest(project_id="parallel_tools", name="Parallel tools"))
        record = studio.agents.create(CreateAgentSessionRequest(
            project_id="parallel_tools", provider_id="deterministic", model_id="graph-builder-v1",
        ))
        functions = {tool.__name__: tool for tool in studio.agents._model_tools(record)}
        results = await asyncio.gather(*(functions["catalog_operator"]("f8.pystudio", "f8.viz.wave") for _ in range(12)))
        assert all(json.loads(result)["operatorClass"] == "f8.viz.wave" for result in results)
        latest = studio.agents.get(record.session_id)
        assert len(latest.tool_calls) == 12
        assert all(call.status.value == "succeeded" for call in latest.tool_calls)
        await studio.agents.close()

    asyncio.run(scenario())


class GraphProposalProvider(AgentProviderRegistry):
    def __init__(self, *, expected_operation_count: int = 9) -> None:
        super().__init__()
        self.expected_operation_count = expected_operation_count

    def summaries(self) -> tuple[AgentProviderSummary, ...]:
        return (AgentProviderSummary(provider_id="proposal_test", display_name="Proposal test", models=("test",), configured=True),)

    async def run_with_tools(self, *, provider_id: str, model_id: str, prompt: str,
                             tools: Sequence[Callable[..., Awaitable[str]]],
                             images: Sequence[AgentImage] = (), reasoning_effort: str | None = None) -> str:
        functions = {tool.__name__: tool for tool in tools}
        await asyncio.gather(*(functions["catalog_operator"]("f8.pyengine", name)
                               for name in ("f8.phase", "f8.cosine", "f8.tcode")))
        graph = json.loads(await functions["graph_read"]())
        changes = {
            "expectedGraphRevision": graph["graphRevision"], "expectedLayoutRevision": graph["layoutRevision"],
            "nodes": [
                {"nodeId": "phase", "serviceClass": "f8.pyengine", "operatorClass": "f8.phase", "serviceId": "engine", "stateValues": {"hz": 1}},
                {"nodeId": "cosine", "serviceClass": "f8.pyengine", "operatorClass": "f8.cosine", "serviceId": "engine", "stateValues": {"dc": 0.5, "amp": 0.5}},
                {"nodeId": "tcode", "serviceClass": "f8.pyengine", "operatorClass": "f8.tcode", "serviceId": "engine", "stateValues": {"intervalMs": 20}},
                {"nodeId": "tcode_viz", "serviceClass": "f8.pystudio", "operatorClass": "f8.viz.tcode", "serviceId": "studio", "stateValues": {"upstreamSampleIntervalMs": 20}},
            ],
            "connections": [
                {"fromNodeId": "phase", "fromPort": "phase", "toNodeId": "cosine", "toPort": "phase"},
                {"fromNodeId": "cosine", "fromPort": "value", "toNodeId": "tcode", "toPort": "L0"},
                {"fromNodeId": "cosine", "fromPort": "value", "toNodeId": "wave", "toPort": "x"},
                {"fromNodeId": "tcode", "fromPort": "tcode", "toNodeId": "tcode_viz", "toPort": "tcode"},
            ],
            "stateUpdates": [{"nodeId": "wave", "field": "upstreamSampleIntervalMs", "value": 20}],
        }
        proposal = json.loads(await functions["graph_propose_changes"](json.dumps(changes)))
        assert proposal["operationCount"] == self.expected_operation_count
        result = json.loads(await functions["graph_apply_proposal"](proposal["proposalId"]))
        assert result["graphChanged"] is True and result["runtimeErrors"] == []
        await functions["graph_validate"]()
        return "Built the requested cosine and TCode graph."


def test_compact_graph_proposal_reaches_approval_and_applies_complete_cosine_chain(tmp_path: Path, engine_service_root: Path) -> None:
    from f8studio_core.compiler import compile_document

    studio = StudioApplication(data_dir=tmp_path / "data", runtime=AgentRuntimeGateway(), service_roots=(engine_service_root,), media_gateway=InProcessMediaGateway())
    studio.agents._providers = GraphProposalProvider()
    studio.projects.create(CreateProjectRequest(project_id="cosine_proposal", name="Cosine proposal"))
    nodes = [studio.catalog.create_node(request) for request in (
        CreateCatalogNodeRequest(kind="service", node_id="engine", service_class="f8.pyengine"),
        CreateCatalogNodeRequest(kind="service", node_id="studio", service_class="f8.pystudio"),
        CreateCatalogNodeRequest(kind="operator", node_id="wave", service_id="studio", service_class="f8.pystudio", operator_class="f8.viz.wave"),
    )]
    studio.projects.patch("cosine_proposal", PatchRequest(request_id="seed", expected_graph_revision=0,
        expected_layout_revision=0, operations=tuple(CreateNodeOp(node=node) for node in nodes)))
    with TestClient(create_app(web_dist=tmp_path, application=studio)) as client:
        response = client.post("/api/agents/sessions", json={"projectId": "cosine_proposal", "providerId": "proposal_test", "modelId": "test"})
        assert response.status_code == 201
        session_id = response.json()["sessionId"]
        client.post(f"/api/agents/sessions/{session_id}/runs", json={"prompt": "Build the cosine and TCode chain"})
        waiting = _wait_for_status(client, session_id, {"waiting_for_approval", "failed"})
        assert waiting["status"] == "waiting_for_approval", waiting
        assert len(studio.projects.document("cosine_proposal").nodes) == 3
        assert waiting["artifacts"]
        _approve_pending(client, session_id)
        finished = _wait_for_status(client, session_id, {"succeeded", "failed"})
        assert finished["status"] == "succeeded", finished
        document = studio.projects.document("cosine_proposal")
        assert len(document.nodes) == 7 and len(document.edges) == 4
        compiled = compile_document(document)
        requests = compiled.per_service["engine"].services[0].autoSampleRequests
        assert {(request.sourceNodeId, request.sourcePort, request.intervalMs) for request in requests} == {
            ("cosine", "value", 20), ("tcode", "tcode", 20),
        }


def test_compact_proposal_refreshes_stale_installed_specs_before_graph_edit(tmp_path: Path, engine_service_root: Path) -> None:
    data_dir = tmp_path / "data"
    studio = StudioApplication(data_dir=data_dir, runtime=AgentRuntimeGateway(), service_roots=(engine_service_root,), media_gateway=InProcessMediaGateway())
    catalog = studio.catalog
    engine = catalog.create_node(CreateCatalogNodeRequest(kind="service", node_id="engine", service_class="f8.pyengine"))
    web = catalog.create_node(CreateCatalogNodeRequest(kind="service", node_id="studio", service_class="f8.pystudio"))
    wave = catalog.create_node(CreateCatalogNodeRequest(kind="operator", node_id="wave", service_id="studio",
                                                        service_class="f8.pystudio", operator_class="f8.viz.wave"))
    tick = catalog.create_node(CreateCatalogNodeRequest(kind="operator", node_id="tick", service_id="engine",
                                                        service_class="f8.pyengine", operator_class="f8.tick"))
    decision = catalog.create_node(CreateCatalogNodeRequest(kind="operator", node_id="decision", service_id="engine",
                                                            service_class="f8.pyengine", operator_class="f8.decision"))
    assert isinstance(tick, OperatorNode) and isinstance(decision, OperatorNode)
    old_tick = replace_node_spec(tick, msgspec.structs.replace(
        tick.spec, dataOutPorts=[port for port in tick.spec.dataOutPorts if port.name != "elapsedSec"],
    ))
    old_decision = replace_node_spec(decision, msgspec.structs.replace(
        decision.spec, label="Decision (Jev)", description="Previous description",
        dataInPorts=[port for port in decision.spec.dataInPorts if port.name != "video"],
        stateFields=[field for field in decision.spec.stateFields if field.name not in {"providerId", "imageMaxSide"}],
    ))
    document = msgspec.structs.replace(new_document(project_id="stale_agent_graph"),
                                       nodes=(engine, web, wave, old_tick, old_decision))
    with pytest.raises(ValueError, match="Supply at least one"):
        build_patch(document, studio.catalog.snapshot(), GraphChanges(
            expected_graph_revision=document.graph_revision, expected_layout_revision=document.layout_revision,
        ), request_id="empty-edit")
    studio.projects.create(CreateProjectRequest(project_id="stale_agent_graph", name="Stale agent graph"))
    ProjectRepository(data_dir / "studio.sqlite3").replace_document("stale_agent_graph", document)
    studio = StudioApplication(data_dir=data_dir, runtime=AgentRuntimeGateway(), service_roots=(engine_service_root,), media_gateway=InProcessMediaGateway())
    changes = msgspec.convert({
        "expectedGraphRevision": 0, "expectedLayoutRevision": 0,
        "nodes": [
            {"nodeId": "phase", "serviceClass": "f8.pyengine", "operatorClass": "f8.phase", "serviceId": "engine", "stateValues": {"hz": 1}},
            {"nodeId": "cosine", "serviceClass": "f8.pyengine", "operatorClass": "f8.cosine", "serviceId": "engine", "stateValues": {"amp": 0.5, "dc": 0.5}},
            {"nodeId": "tcode", "serviceClass": "f8.pyengine", "operatorClass": "f8.tcode", "serviceId": "engine", "stateValues": {"intervalMs": 20}},
            {"nodeId": "tcode_viz", "serviceClass": "f8.pystudio", "operatorClass": "f8.viz.tcode", "serviceId": "studio", "stateValues": {}},
        ],
        "connections": [
            {"fromNodeId": "phase", "fromPort": "phase", "toNodeId": "cosine", "toPort": "phase"},
            {"fromNodeId": "cosine", "fromPort": "value", "toNodeId": "tcode", "toPort": "L0"},
            {"fromNodeId": "cosine", "fromPort": "value", "toNodeId": "wave", "toPort": "x"},
            {"fromNodeId": "tcode", "fromPort": "tcode", "toNodeId": "tcode_viz", "toPort": "tcode"},
            {"fromNodeId": "tick", "fromPort": "elapsedSec", "toNodeId": "wave", "toPort": "y"},
        ],
        "stateUpdates": [
            {"nodeId": "wave", "field": "upstreamSampleIntervalMs", "value": 20},
            {"nodeId": "tcode_viz", "field": "upstreamSampleIntervalMs", "value": 20},
        ],
    }, type=GraphChanges)
    patch = build_patch(studio.projects.document("stale_agent_graph"), studio.catalog.snapshot(), changes,
                        request_id="refresh_and_build")
    assert [operation.node_id for operation in patch.operations[:2]] == ["tick", "decision"]
    preview = studio.tools.preview_patch("stale_agent_graph", patch)
    assert studio.projects.document("stale_agent_graph") == document
    studio.projects.validate(preview.document)
    assert len(preview.document.nodes) == 9 and len(preview.document.edges) == 5
    updated = {node.node_id: node for node in preview.document.nodes}
    assert any(port.name == "elapsedSec" for port in updated["tick"].ports)
    assert any(port.name == "video" for port in updated["decision"].ports)
    studio.agents._providers = GraphProposalProvider(expected_operation_count=11)
    with TestClient(create_app(web_dist=tmp_path, application=studio)) as client:
        response = client.post("/api/agents/sessions", json={
            "projectId": "stale_agent_graph", "providerId": "proposal_test", "modelId": "test",
        })
        assert response.status_code == 201
        session_id = response.json()["sessionId"]
        client.post(f"/api/agents/sessions/{session_id}/runs", json={"prompt": "Build a cosine graph"})
        waiting = _wait_for_status(client, session_id, {"waiting_for_approval", "failed"})
        assert waiting["status"] == "waiting_for_approval", waiting
        assert studio.projects.document("stale_agent_graph") == document
        _approve_pending(client, session_id)
        finished = _wait_for_status(client, session_id, {"succeeded", "failed"})
        assert finished["status"] == "succeeded", finished
    applied = studio.projects.document("stale_agent_graph")
    assert len(applied.nodes) == 9 and len(applied.edges) == 4
    assert any(port.name == "elapsedSec" for port in next(node for node in applied.nodes if node.node_id == "tick").ports)


def test_unreal_executable_detection_precedes_generic_exe_detection(tmp_path: Path) -> None:
    executable = tmp_path / "MyGame" / "Binaries" / "Win64" / "MyGame.exe"
    executable.parent.mkdir(parents=True)
    executable.touch()
    result = LocalIntegrationService().detect_modding_target(
        DetectModdingTargetRequest(target_path=str(executable))
    )
    assert isinstance(result, dict)
    assert result["engine"] == "unreal"
    assert result["supported"] is False


def test_agent_approval_is_invalidated_by_layout_only_edit(tmp_path: Path) -> None:
    studio = StudioApplication(data_dir=tmp_path / "data", runtime=AgentRuntimeGateway(),
                               service_roots=(), media_gateway=InProcessMediaGateway())
    with TestClient(create_app(web_dist=tmp_path, application=studio)) as client:
        client.post('/api/projects', json={'projectId': 'layout-conflict', 'name': 'Layout'})
        service = client.post('/api/catalog/nodes', json={
            'kind': 'service', 'nodeId': 'manual_studio', 'serviceClass': 'f8.pystudio',
        }).json()
        assert client.post('/api/projects/layout-conflict/patch', json={
            'requestId': 'seed', 'expectedGraphRevision': 0, 'expectedLayoutRevision': 0,
            'operations': [{'op': 'createNode', 'node': service}],
        }).status_code == 200
        session_id = _create_session(client, 'layout-conflict')
        client.post(f'/api/agents/sessions/{session_id}/runs', json={'prompt': 'Build a value graph'})
        waiting = _wait_for_status(client, session_id, {'waiting_for_approval'})
        approval = waiting['approval']
        assert client.post('/api/projects/layout-conflict/patch', json={
            'requestId': 'move', 'expectedGraphRevision': 1, 'expectedLayoutRevision': 0,
            'operations': [{'op': 'setNodeLayout', 'layout': {'nodeId': 'manual_studio', 'x': 20, 'y': 30}}],
        }).status_code == 200
        response = client.post(f"/api/agents/sessions/{session_id}/approvals/{approval['approvalId']}",
                               json={'approved': True, 'argumentsHash': approval['argumentsHash']})
        assert response.status_code == 409
        assert 'layout expected' in response.text
