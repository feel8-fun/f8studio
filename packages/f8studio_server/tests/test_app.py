import asyncio
from pathlib import Path
import time

from aiortc import RTCPeerConnection
from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import msgspec
import pytest
from starlette.websockets import WebSocketDisconnect

from f8media_gateway.app import create_app as create_media_gateway_app
from f8media_protocol.client import RemoteMediaGateway, RemoteMediaGatewayConfig
from f8media_protocol.models import MEDIA_API_VERSION
from f8media_gateway.service import InProcessMediaGateway
from f8pysdk.specs import F8JsonValue, F8RuntimeGraph, F8ServiceSpec
from f8studio_core.graph import CreateNodeOp, HistoryRequest, NodeCatalog, PatchRequest, PatchResult, new_document

from f8studio_server import create_app
from f8studio_server.app import _patch_payload
from f8studio_server.application import StudioApplication
from f8studio_server.models import (
    BrowserIceServer,
    BrowserRtcConfiguration,
    RuntimeActionResult,
    RuntimeStateField,
    ServiceDeployResult,
    ServiceRuntimeStatus,
)
from f8studio_server.runtime import RuntimeMonitorCallback


def test_patch_payload_includes_runtime_sync_errors() -> None:
    result = PatchResult(
        request_id="state-change",
        document=new_document(project_id="project1"),
        graph_changed=True,
        layout_changed=False,
        runtime_errors=("player.volume: rejected",),
    )

    encoded = msgspec.json.encode(_patch_payload(result))
    assert msgspec.json.decode(encoded)["runtimeErrors"] == ["player.volume: rejected"]


class FakeRuntimeGateway:
    def __init__(self) -> None:
        self.deploy_calls: list[str] = []
        self.closed = False
        self.state_values: dict[tuple[str, str, str], F8JsonValue] = {}

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
        key = (service_id, node_id, field)
        return RuntimeStateField(
            field=field,
            found=key in self.state_values,
            value=self.state_values.get(key),
            ts_ms=123 if key in self.state_values else None,
        )

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
        self.closed = True


class DisconnectedRuntimeGateway(FakeRuntimeGateway):
    async def status(self, service_id: str) -> ServiceRuntimeStatus:
        raise OSError(f"Zenoh endpoint disconnected: {service_id}")


async def request(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(path)


async def create_video_offer() -> str:
    peer = RTCPeerConnection()
    peer.addTransceiver("video", direction="recvonly")
    try:
        offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        local = peer.localDescription
        return local.sdp
    finally:
        await peer.close()


def test_recent_logs_endpoint_exposes_bounded_service_output(tmp_path: Path) -> None:
    studio = StudioApplication(data_dir=tmp_path / "data", runtime=FakeRuntimeGateway(), service_roots=())
    app = create_app(web_dist=tmp_path, application=studio)

    asyncio.run(studio.events.publish(
        event_type="service.log", scope="service:capture",
        payload={"serviceId": "capture", "line": "capture started"}, reliable=False,
    ))
    response = asyncio.run(request(app, "/api/logs?limit=1"))

    assert response.status_code == 200
    assert response.json()[0]["payload"] == {"serviceId": "capture", "line": "capture started"}
    assert asyncio.run(request(app, f'/api/logs?before_sequence={response.json()[0]["sequence"]}')).json() == []
    assert asyncio.run(request(app, "/api/logs?limit=1001")).status_code == 422


async def create_audio_offer() -> str:
    peer = RTCPeerConnection()
    peer.addTransceiver("audio", direction="recvonly")
    try:
        offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        return peer.localDescription.sdp
    finally:
        await peer.close()


def test_runtime_state_read_returns_retained_node_values(tmp_path: Path) -> None:
    runtime = FakeRuntimeGateway()
    runtime.state_values[("capture", "capture", "captureRunning")] = True
    runtime.state_values[("capture", "capture", "videoWidth")] = 1920
    app = create_app(web_dist=tmp_path, data_dir=tmp_path / "data", runtime=runtime, service_roots=())

    with TestClient(app) as client:
        response = client.post(
            "/api/runtime/services/capture/nodes/capture/state:read",
            json={"fields": ["captureRunning", "videoWidth", "videoHeight"]},
        )

    assert response.status_code == 200
    assert response.json() == {
        "serviceId": "capture",
        "nodeId": "capture",
        "fields": [
            {"field": "captureRunning", "found": True, "value": True, "tsMs": 123},
            {"field": "videoWidth", "found": True, "value": 1920, "tsMs": 123},
            {"field": "videoHeight", "found": False, "value": None, "tsMs": None},
        ],
    }


def test_media_api_rejects_invalid_source_and_quality(tmp_path: Path) -> None:
    app = create_app(
        web_dist=tmp_path,
        data_dir=tmp_path / "data",
        runtime=FakeRuntimeGateway(),
        service_roots=(),
    )
    with TestClient(app) as client:
        bad_quality = client.post(
            "/api/media/sessions",
            json={"source": "synthetic://bars", "quality": "ultra", "sdp": "offer", "type": "offer"},
        )
        assert bad_quality.status_code == 422
        assert bad_quality.json()["detail"] == "media quality must be thumbnail or main"

        bad_source = client.post(
            "/api/media/sessions",
            json={"source": "file:///tmp/video", "quality": "thumbnail", "sdp": "offer", "type": "offer"},
        )
        assert bad_source.status_code == 422
        assert bad_source.json()["detail"] == (
            "media source must be synthetic://bars, synthetic://bars-1080p, or an f8/ Zenoh key"
        )


def test_media_rtc_configuration_defaults_to_direct_ice(tmp_path: Path) -> None:
    app = create_app(
        web_dist=tmp_path,
        data_dir=tmp_path / "data",
        runtime=FakeRuntimeGateway(),
        service_roots=(),
    )
    with TestClient(app) as client:
        response = client.get("/api/media/rtc-configuration")

    assert response.status_code == 200
    assert response.json() == {"iceServers": [], "iceTransportPolicy": "all"}


def test_media_rtc_configuration_exposes_turn_relay_config(tmp_path: Path) -> None:
    rtc_configuration = BrowserRtcConfiguration(
        ice_servers=(
            BrowserIceServer(
                urls=("turn:localhost:3478?transport=tcp",),
                username="studio",
                credential="secret",
            ),
        ),
        ice_transport_policy="relay",
    )
    app = create_app(
        web_dist=tmp_path,
        data_dir=tmp_path / "data",
        runtime=FakeRuntimeGateway(),
        service_roots=(),
        rtc_configuration=rtc_configuration,
    )
    with TestClient(app) as client:
        response = client.get("/api/media/rtc-configuration")

    assert response.status_code == 200
    assert response.json() == {
        "iceServers": [
            {
                "urls": ["turn:localhost:3478?transport=tcp"],
                "username": "studio",
                "credential": "secret",
            }
        ],
        "iceTransportPolicy": "relay",
    }


def test_media_api_unknown_session_is_not_found(tmp_path: Path) -> None:
    app = create_app(
        web_dist=tmp_path,
        data_dir=tmp_path / "data",
        runtime=FakeRuntimeGateway(),
        service_roots=(),
    )
    with TestClient(app) as client:
        response = client.delete("/api/media/sessions/missing")
        assert response.status_code == 404
        assert response.json() == {"detail": "media session not found"}


def test_studio_proxies_remote_gateway_responses(tmp_path: Path) -> None:
    gateway_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_media_gateway_app()),
        base_url="http://testserver",
    )
    gateway = RemoteMediaGateway(
        RemoteMediaGatewayConfig(base_url="http://testserver", manage_process=False),
        client=gateway_client,
    )
    app = create_app(
        web_dist=tmp_path,
        data_dir=tmp_path / "data",
        runtime=FakeRuntimeGateway(),
        service_roots=(),
        media_gateway=gateway,
    )
    try:
        with TestClient(app) as client:
            health = client.get("/api/media/gateway")
            invalid = client.post(
                "/api/media/sessions",
                json={
                    "source": "synthetic://bars",
                    "quality": "ultra",
                    "sdp": "offer",
                    "type": "offer",
                },
            )
            missing = client.delete("/api/media/sessions/missing")

        assert health.status_code == 200
        assert health.json()["protocolVersion"] == MEDIA_API_VERSION
        assert invalid.status_code == 422
        assert invalid.json() == {"detail": "media quality must be thumbnail or main"}
        assert missing.status_code == 404
        assert missing.json() == {"detail": "media session not found"}
    finally:
        asyncio.run(gateway_client.aclose())


def test_studio_reports_gateway_disconnect_as_service_unavailable(tmp_path: Path) -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "service": "f8media-gateway",
                    "version": "0.1.0",
                    "protocolVersion": MEDIA_API_VERSION,
                    "gatewayEpoch": "available-at-startup",
                    "processId": 1,
                },
            )
        raise httpx.ConnectError("gateway stopped", request=request)

    gateway_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver",
    )
    gateway = RemoteMediaGateway(
        RemoteMediaGatewayConfig(base_url="http://testserver", manage_process=False),
        client=gateway_client,
    )
    app = create_app(
        web_dist=tmp_path,
        data_dir=tmp_path / "data",
        runtime=FakeRuntimeGateway(),
        service_roots=(),
        media_gateway=gateway,
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/media/gateway")

        assert response.status_code == 503
        assert response.json()["detail"].startswith("Media Gateway request failed: ConnectError")
    finally:
        asyncio.run(gateway_client.aclose())


def test_media_sample_api_returns_exact_value_and_releases_source(tmp_path: Path) -> None:
    gateway = InProcessMediaGateway()
    studio = StudioApplication(
        data_dir=tmp_path / "data",
        runtime=FakeRuntimeGateway(),
        service_roots=(),
        media_gateway=gateway,
    )
    app = create_app(web_dist=tmp_path, application=studio)
    with TestClient(app) as client:
        response = client.get(
            "/api/media/sample",
            params={"source": "synthetic://bars", "x": 10, "y": 10},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["format"] == "bgra32"
        assert payload["frameId"] > 0
        assert payload["value"]["alpha"] == 255
        assert gateway.video.source_count == 0

        outside = client.get(
            "/api/media/sample",
            params={"source": "synthetic://bars", "x": 640, "y": 0},
        )
        assert outside.status_code == 422
        assert "outside 640x360" in outside.json()["detail"]
        assert gateway.video.source_count == 0


def test_application_shutdown_closes_active_media_sessions(tmp_path: Path) -> None:
    gateway = InProcessMediaGateway()
    studio = StudioApplication(
        data_dir=tmp_path / "data",
        runtime=FakeRuntimeGateway(),
        service_roots=(),
        media_gateway=gateway,
    )
    app = create_app(web_dist=tmp_path, application=studio)
    offer_sdp = asyncio.run(create_video_offer())

    with TestClient(app) as client:
        response = client.post(
            "/api/media/sessions",
            json={"source": "synthetic://bars", "quality": "thumbnail", "sdp": offer_sdp, "type": "offer"},
        )
        assert response.status_code == 201
        assert gateway.video.session_count == 1
        assert gateway.video.source_count == 1

    assert gateway.video.session_count == 0
    assert gateway.video.source_count == 0


def test_application_shutdown_closes_active_audio_sessions(tmp_path: Path) -> None:
    gateway = InProcessMediaGateway()
    studio = StudioApplication(
        data_dir=tmp_path / "data",
        runtime=FakeRuntimeGateway(),
        service_roots=(),
        media_gateway=gateway,
    )
    app = create_app(web_dist=tmp_path, application=studio)
    offer_sdp = asyncio.run(create_audio_offer())

    with TestClient(app) as client:
        response = client.post(
            "/api/audio/sessions",
            json={"source": "synthetic://tone", "sdp": offer_sdp, "type": "offer"},
        )
        assert response.status_code == 201
        assert response.json()["transportPolicy"].startswith("bounded-queue-16")
        assert gateway.audio.session_count == 1
        assert gateway.audio.source_count == 1

    assert gateway.audio.session_count == 0
    assert gateway.audio.source_count == 0


def test_overlay_api_validates_identity_and_reports_monitor_metrics(tmp_path: Path) -> None:
    app = create_app(
        web_dist=tmp_path,
        data_dir=tmp_path / "data",
        runtime=FakeRuntimeGateway(),
        service_roots=(),
    )
    payload = {
        "source": "f8/test/video",
        "streamId": "camera",
        "streamEpoch": "epoch",
        "frameId": 1,
        "captureTimestampMs": 100,
        "detections": [{"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5, "score": 2.0}],
    }
    with TestClient(app) as client:
        rejected = client.post("/api/media/overlays", json=payload)
        assert rejected.status_code == 422
        assert "score" in rejected.json()["detail"]
        payload["detections"][0]["score"] = 0.9
        accepted = client.post("/api/media/overlays", json=payload)
        assert accepted.status_code == 202
        metrics = client.get("/api/media/metrics")
        assert metrics.status_code == 200
        assert metrics.json()["overlayRejected"] == 1
        assert metrics.json()["videoSessions"] == 0


def test_health_and_capabilities_report_current_scope(tmp_path: Path) -> None:
    app = create_app(web_dist=tmp_path, data_dir=tmp_path / "data", service_roots=())

    health = asyncio.run(request(app, "/api/health"))
    capabilities = asyncio.run(request(app, "/api/capabilities"))

    assert health.status_code == 200
    assert health.json()["protocol_version"] == "f8studio-api/1"
    assert health.json()["server_epoch"]
    assert capabilities.status_code == 200
    assert capabilities.json()["capabilities"] == {
        "graph_editing": True,
        "runtime_control": True,
        "web_assets": False,
        "web_rtc_video": True,
        "web_rtc_audio": True,
        "three_d": True,
        "agent_tools": True,
    }


def test_built_web_app_is_served_with_history_fallback(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>Web Studio</h1>", encoding="utf-8")
    app = create_app(web_dist=tmp_path, data_dir=tmp_path / "data", service_roots=())

    response = asyncio.run(request(app, "/projects/example"))

    assert response.status_code == 200
    assert "Web Studio" in response.text
    capabilities = asyncio.run(request(app, "/api/capabilities"))
    assert capabilities.json()["capabilities"]["web_assets"] is True


def test_unknown_api_route_is_not_replaced_by_web_app(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<h1>Web Studio</h1>", encoding="utf-8")
    app = create_app(web_dist=tmp_path, data_dir=tmp_path / "data", service_roots=())

    response = asyncio.run(request(app, "/api/misspelled"))

    assert response.status_code == 404
    assert response.json() == {"detail": "API route not found"}


def test_project_patch_api_persists_and_reports_revision_conflicts(tmp_path: Path) -> None:
    async def scenario() -> None:
        data_dir = tmp_path / "data"
        app = create_app(web_dist=tmp_path, data_dir=data_dir, service_roots=())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            created = await client.post(
                "/api/projects",
                json={"projectId": "project1", "name": "Example"},
            )
            assert created.status_code == 201
            latest_deployment = await client.get("/api/projects/project1/deployments/latest")
            assert latest_deployment.status_code == 200
            assert latest_deployment.json() is None
            catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.pyengine", label="Engine")])
            engine = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")
            patch = PatchRequest(
                request_id="create-engine",
                expected_graph_revision=0,
                expected_layout_revision=0,
                operations=(CreateNodeOp(node=engine),),
            )
            committed = await client.post(
                "/api/projects/project1/patch",
                content=msgspec.json.encode(patch),
                headers={"content-type": "application/json"},
            )
            assert committed.status_code == 200
            assert committed.json()["document"]["graphRevision"] == 1

            stale = await client.post(
                "/api/projects/project1/patch",
                content=msgspec.json.encode(
                    PatchRequest(
                        request_id="stale",
                        expected_graph_revision=0,
                        expected_layout_revision=0,
                        operations=(),
                    )
                ),
                headers={"content-type": "application/json"},
            )
            assert stale.status_code == 409
            assert stale.json()["detail"]["code"] == "revision_conflict"

            undone = await client.post(
                "/api/projects/project1/undo",
                content=msgspec.json.encode(
                    HistoryRequest(
                        request_id="undo-create",
                        expected_graph_revision=1,
                        expected_layout_revision=0,
                    )
                ),
                headers={"content-type": "application/json"},
            )
            assert undone.status_code == 200
            assert undone.json()["document"]["graphRevision"] == 2
            assert undone.json()["document"]["nodes"] == []

            redone = await client.post(
                "/api/projects/project1/redo",
                content=msgspec.json.encode(
                    HistoryRequest(
                        request_id="redo-create",
                        expected_graph_revision=2,
                        expected_layout_revision=0,
                    )
                ),
                headers={"content-type": "application/json"},
            )
            assert redone.status_code == 200
            assert redone.json()["document"]["graphRevision"] == 3

        restarted = create_app(web_dist=tmp_path, data_dir=data_dir, service_roots=())
        restart_transport = httpx.ASGITransport(app=restarted)
        async with httpx.AsyncClient(transport=restart_transport, base_url="http://testserver") as client:
            loaded = await client.get("/api/projects/project1")
            assert loaded.status_code == 200
            assert loaded.json()["document"]["graphRevision"] == 3

    asyncio.run(scenario())


def test_catalog_creates_valid_graph_nodes_with_authoritative_ports(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = create_app(web_dist=tmp_path, data_dir=tmp_path / "data", service_roots=())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            service = await client.post(
                "/api/catalog/nodes",
                json={"kind": "service", "nodeId": "studio", "serviceClass": "f8.pystudio"},
            )
            assert service.status_code == 200
            assert service.json()["kind"] == "service"
            assert service.json()["serviceId"] == "studio"
            assert service.json()["ports"]

            operator = await client.post(
                "/api/catalog/nodes",
                json={
                    "kind": "operator",
                    "nodeId": "video",
                    "serviceId": "studio",
                    "serviceClass": "f8.pystudio",
                    "operatorClass": "f8.viz.video",
                },
            )
            assert operator.status_code == 200
            assert operator.json()["kind"] == "operator"
            assert operator.json()["serviceId"] == "studio"
            assert operator.json()["ports"]

    asyncio.run(scenario())


def test_mutating_api_rejects_non_loopback_origin(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = create_app(web_dist=tmp_path, data_dir=tmp_path / "data", service_roots=())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/projects",
                json={"projectId": "project1", "name": "Example"},
                headers={"origin": "https://attacker.example"},
            )
            assert response.status_code == 403

    asyncio.run(scenario())


def test_explicit_vpn_host_is_trusted_for_same_origin_requests(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = create_app(
            web_dist=tmp_path,
            data_dir=tmp_path / "data",
            runtime=FakeRuntimeGateway(),
            service_roots=(),
            allowed_hosts=("vpn.test",),
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://vpn.test:8260") as client:
            response = await client.post(
                "/api/projects",
                json={"projectId": "project1", "name": "Example"},
                headers={"origin": "http://vpn.test:8260"},
            )
            assert response.status_code == 201

    asyncio.run(scenario())


def test_event_websocket_snapshot_commit_replay_and_origin(tmp_path: Path) -> None:
    app = create_app(
        web_dist=tmp_path,
        data_dir=tmp_path / "data",
        runtime=FakeRuntimeGateway(),
        service_roots=(),
    )
    catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.pyengine", label="Engine")])
    engine = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")
    patch = PatchRequest(
        request_id="create-engine",
        expected_graph_revision=0,
        expected_layout_revision=0,
        operations=(CreateNodeOp(node=engine),),
    )

    with TestClient(app) as client:
        with client.websocket_connect("/api/events") as websocket:
            snapshot = websocket.receive_json()
            assert snapshot["type"] == "stream.snapshot"
            assert snapshot["sequence"] == 0
            epoch = snapshot["serverEpoch"]

            created = client.post("/api/projects", json={"projectId": "project1", "name": "Example"})
            assert created.status_code == 201
            created_event = websocket.receive_json()
            assert created_event["type"] == "project.created"

            committed = client.post(
                "/api/projects/project1/patch",
                content=msgspec.json.encode(patch),
                headers={"content-type": "application/json"},
            )
            assert committed.status_code == 200
            graph_event = websocket.receive_json()
            assert graph_event["type"] == "graph.committed"
            assert graph_event["sequence"] == 2

            replayed = client.post(
                "/api/projects/project1/patch",
                content=msgspec.json.encode(patch),
                headers={"content-type": "application/json"},
            )
            assert replayed.status_code == 200
            updated = client.put(
                "/api/projects/project1",
                json={"name": "Renamed", "description": ""},
            )
            assert updated.status_code == 200
            next_event = websocket.receive_json()
            assert next_event["type"] == "project.updated"
            assert next_event["sequence"] == 3

        with client.websocket_connect(f"/api/events?epoch={epoch}&after=1") as websocket:
            replayed_graph = websocket.receive_json()
            replayed_update = websocket.receive_json()
            assert [replayed_graph["type"], replayed_update["type"]] == [
                "graph.committed",
                "project.updated",
            ]
            assert [replayed_graph["sequence"], replayed_update["sequence"]] == [2, 3]

        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                "/api/events",
                headers={"origin": "https://attacker.example"},
            ):
                pass
        assert rejected.value.code == 1008


def test_deploy_api_runs_job_through_injected_runtime(tmp_path: Path) -> None:
    runtime = FakeRuntimeGateway()
    app = create_app(
        web_dist=tmp_path,
        data_dir=tmp_path / "data",
        runtime=runtime,
        service_roots=(),
    )
    catalog = NodeCatalog(services=[F8ServiceSpec(serviceClass="f8.pyengine", label="Engine")])
    engine = catalog.create_service_node(node_id="engine", service_class="f8.pyengine")

    with TestClient(app) as client:
        assert client.post(
            "/api/projects",
            json={"projectId": "project1", "name": "Example"},
        ).status_code == 201
        patch = PatchRequest(
            request_id="create-engine",
            expected_graph_revision=0,
            expected_layout_revision=0,
            operations=(CreateNodeOp(node=engine),),
        )
        assert client.post(
            "/api/projects/project1/patch",
            content=msgspec.json.encode(patch),
            headers={"content-type": "application/json"},
        ).status_code == 200

        submitted = client.post(
            "/api/projects/project1/deploy",
            json={"requestId": "deploy1", "expectedGraphRevision": 1},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["jobId"]
        for _ in range(100):
            job = client.get(f"/api/jobs/{job_id}")
            assert job.status_code == 200
            if job.json()["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        assert job.json()["status"] == "succeeded"
        assert runtime.deploy_calls == ["engine"]

    assert runtime.closed is True


def test_runtime_disconnect_maps_to_service_unavailable(tmp_path: Path) -> None:
    runtime = DisconnectedRuntimeGateway()
    app = create_app(
        web_dist=tmp_path,
        data_dir=tmp_path / "data",
        runtime=runtime,
        service_roots=(),
    )

    with TestClient(app) as client:
        response = client.get("/api/runtime/services/offline/status")
        assert response.status_code == 503
        assert response.json()["detail"] == "OSError: Zenoh endpoint disconnected: offline"

    assert runtime.closed is True
