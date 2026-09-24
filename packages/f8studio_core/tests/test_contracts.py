from f8studio_core import API_PROTOCOL_VERSION, HealthStatus, ServerCapabilities


def test_health_status_uses_explicit_wire_names() -> None:
    status = HealthStatus(
        status="ok",
        service="f8studio-server",
        version="0.1.0",
        protocol_version=API_PROTOCOL_VERSION,
        server_epoch="epoch-1",
    )

    assert status.to_json_object() == {
        "status": "ok",
        "service": "f8studio-server",
        "version": "0.1.0",
        "protocol_version": "f8studio-api/1",
        "server_epoch": "epoch-1",
    }


def test_capabilities_do_not_imply_unimplemented_features() -> None:
    capabilities = ServerCapabilities(
        graph_editing=False,
        runtime_control=False,
        web_assets=True,
        web_rtc_video=False,
        web_rtc_audio=False,
        three_d=False,
        agent_tools=False,
    )

    payload = capabilities.to_json_object()
    assert payload["web_assets"] is True
    assert payload["web_rtc_video"] is False
    assert payload["web_rtc_audio"] is False
