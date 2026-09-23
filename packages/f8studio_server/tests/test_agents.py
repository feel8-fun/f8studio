from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from f8pysdk.specs import F8JsonValue, F8RuntimeGraph
from f8studio_server import create_app
from f8studio_server.application import StudioApplication
from f8studio_server.models import (
    RuntimeActionResult,
    RuntimeStateField,
    ServiceDeployResult,
    ServiceRuntimeStatus,
)
from f8studio_server.runtime import RuntimeMonitorCallback


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
    )
    app = create_app(web_dist=tmp_path, application=studio)

    with TestClient(app) as client:
        created = client.post("/api/projects", json={"projectId": "agent_project", "name": "Agent"})
        assert created.status_code == 201
        session_id = _create_session(client, "agent_project")

        with client.websocket_connect("/api/events") as websocket:
            snapshot = websocket.receive_json()
            assert snapshot["type"] == "stream.snapshot"
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


def test_agent_approval_is_invalidated_when_another_client_changes_revision(tmp_path: Path) -> None:
    studio = StudioApplication(
        data_dir=tmp_path / "data",
        runtime=AgentRuntimeGateway(),
        service_roots=(),
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


def test_agent_run_fails_with_tool_context_when_deployment_fails(tmp_path: Path) -> None:
    studio = StudioApplication(
        data_dir=tmp_path / "data",
        runtime=AgentRuntimeGateway(deploy_success=False),
        service_roots=(),
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
