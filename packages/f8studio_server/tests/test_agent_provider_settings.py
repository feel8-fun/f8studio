from __future__ import annotations

import os
import asyncio
from pathlib import Path

import httpx
import msgspec
import pytest

from f8studio_server.agents.provider_settings import CreateProviderConnection, ModelCapabilities, ProviderSettingsStore, UpdateProviderSettings
from f8studio_server.agents.provider_probe import ProbeProviderRequest, probe_provider
from f8studio_server.agents.providers import AgentProviderRegistry


def test_settings_persist_and_keep_credentials_private(tmp_path: Path) -> None:
    path = tmp_path / "providers.json"
    registry = AgentProviderRegistry(path)
    saved = registry.update_settings("openai", UpdateProviderSettings(model="custom-model", api_key="private-test-key"))
    assert saved.configured
    assert saved.api_key_set
    assert b"private-test-key" not in msgspec.json.encode(registry.settings())
    assert b"private-test-key" not in msgspec.json.encode(registry.summaries())
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600

    restored = AgentProviderRegistry(path)
    restored.validate_selection("openai", "custom-model")
    changed = restored.update_settings("openai", UpdateProviderSettings(model="second-model"))
    assert changed.api_key_set
    cleared = restored.update_settings("openai", UpdateProviderSettings(model="second-model", clear_api_key=True))
    assert not cleared.api_key_set
    assert not cleared.configured
    assert all(item.provider_id != "openai" for item in restored.settings())
    with pytest.raises(ValueError, match="not configured"):
        AgentProviderRegistry(path).validate_selection("openai", "second-model")


def test_saved_settings_override_environment_without_modifying_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    path = tmp_path / "providers.json"
    registry = AgentProviderRegistry(path)
    assert registry.settings()[0].api_key_set
    registry.update_settings("openai", UpdateProviderSettings(model="custom", clear_api_key=True))
    assert not ProviderSettingsStore(path).get("openai").api_key
    assert os.environ["OPENAI_API_KEY"] == "environment-key"


def test_deleting_legacy_connection_suppresses_environment_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    path = tmp_path / "providers.json"
    store = ProviderSettingsStore(path)
    store.update("openai", UpdateProviderSettings(model="work-model", api_key="saved-secret"))
    assert any(item.provider_id == "openai" for item in store.views())

    store.delete("openai")
    assert all(item.provider_id != "openai" for item in store.views())
    assert b"saved-secret" not in path.read_bytes()
    assert b"environment-key" not in path.read_bytes()
    assert all(item.provider_id != "openai" for item in ProviderSettingsStore(path).views())
    assert os.environ["OPENAI_API_KEY"] == "environment-key"
    with pytest.raises(ValueError, match="deleted"):
        store.get("openai")


def test_deleting_environment_only_legacy_connection_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    path = tmp_path / "providers.json"
    store = ProviderSettingsStore(path)
    assert store.views()[0].provider_id == "openai"
    store.delete("openai")
    assert ProviderSettingsStore(path).views() == ()
    with pytest.raises(ValueError, match="deleted"):
        store.delete("openai")


@pytest.mark.parametrize("endpoint", ["file:///tmp/model", "https://user:secret@example.com/v1", "https://example.com/v1?api_key=secret", "https://example.com/#secret", "not-a-url"])
def test_invalid_endpoints_do_not_replace_working_settings(tmp_path: Path, endpoint: str) -> None:
    path = tmp_path / "providers.json"
    store = ProviderSettingsStore(path)
    store.update("ollama", UpdateProviderSettings(model="local-model", endpoint="http://localhost:11434/v1"))
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        store.update("ollama", UpdateProviderSettings(model="changed", endpoint=endpoint))
    assert path.read_bytes() == previous
    assert store.get("ollama").model == "local-model"


def test_local_provider_needs_no_key_and_unknown_providers_are_rejected(tmp_path: Path) -> None:
    registry = AgentProviderRegistry(tmp_path / "providers.json")
    saved = registry.update_settings("ollama", UpdateProviderSettings(model="local-model", endpoint="http://localhost:11434/v1"))
    assert saved.configured and not saved.requires_api_key
    registry.validate_selection("ollama", "local-model")
    with pytest.raises(ValueError, match="Unknown"):
        registry.update_settings("deterministic", UpdateProviderSettings(model="other"))


def test_failed_save_leaves_active_configuration_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "providers.json"
    store = ProviderSettingsStore(path)
    store.update("ollama", UpdateProviderSettings(model="working", endpoint="http://localhost:11434/v1"))
    previous = path.read_bytes()

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("Disk unavailable")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="Disk unavailable"):
        store.update("ollama", UpdateProviderSettings(model="replacement", endpoint="http://localhost:11434/v1"))
    assert store.get("ollama").model == "working"
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_named_connections_share_protocol_but_keep_keys_and_models_separate(tmp_path: Path) -> None:
    path = tmp_path / "providers.json"
    store = ProviderSettingsStore(path)
    first = store.create(CreateProviderConnection(
        display_name="Work", protocol="openai_chat", endpoint="https://work.example/v1",
        api_key="work-secret", model="model-a", models=("model-a", "model-b"),
    ))
    second = store.create(CreateProviderConnection(
        display_name="Personal", protocol="openai_chat", endpoint="https://personal.example/v1",
        api_key="personal-secret", model="model-c",
    ))
    assert first.provider_id != second.provider_id
    assert first.models == ("model-a", "model-b")
    assert second.models == ("model-c",)
    assert b"work-secret" not in msgspec.json.encode(store.views())
    restored = ProviderSettingsStore(path)
    assert restored.get(first.provider_id).api_key == "work-secret"
    assert restored.get(second.provider_id).api_key == "personal-secret"
    restored.update(first.provider_id, UpdateProviderSettings(
        display_name="Work renamed", endpoint="https://work.example/v1", model="model-b", models=("model-a", "model-b"),
    ))
    assert ProviderSettingsStore(path).get(first.provider_id).display_name == "Work renamed"
    restored.delete(first.provider_id)
    assert all(item.provider_id != first.provider_id for item in ProviderSettingsStore(path).views())
    assert ProviderSettingsStore(path).get(second.provider_id).api_key == "personal-secret"


def test_empty_legacy_templates_are_hidden_and_default_model_is_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "TYPESAFE_API_KEY", "F8STUDIO_OLLAMA_MODEL"):
        monkeypatch.delenv(key, raising=False)
    store = ProviderSettingsStore(tmp_path / "providers.json")
    assert store.views() == ()
    connection = store.create(CreateProviderConnection(
        display_name="Work", protocol="openai_chat", endpoint="https://example.com/v1",
        model="first", models=("first", "second"),
    ))
    updated = store.update(connection.provider_id, UpdateProviderSettings(
        display_name="Work", endpoint="https://example.com/v1", model="second", models=("first", "second"),
    ))
    assert updated.models == ("second", "first")
    assert tuple(item.provider_id for item in store.views()) == (connection.provider_id,)


def test_systemone_connection_and_model_capabilities_are_persisted(tmp_path: Path) -> None:
    registry = AgentProviderRegistry(tmp_path / "providers.json")
    connection = registry.create_connection(CreateProviderConnection(
        display_name="Local decisions", protocol="systemone", endpoint="http://localhost:8001/v1",
        model="vision", models=("vision", "text"), model_capabilities=(
            ModelCapabilities(model_id="vision", image_input=True, thinking=False),
            ModelCapabilities(model_id="text", image_input=False),
        ),
    ))
    assert connection.kind == "decision" and connection.configured
    assert connection.provider_id not in {item.provider_id for item in registry.summaries()}
    assert registry.decision_config(connection.provider_id).model == "vision"
    assert registry.supports_image(connection.provider_id, "vision")
    assert not registry.supports_image(connection.provider_id, "text")
    assert AgentProviderRegistry(tmp_path / "providers.json").settings()[0].model_capabilities[0].thinking is False


def test_legacy_local_image_setting_survives_model_edit(tmp_path: Path) -> None:
    store = ProviderSettingsStore(tmp_path / "providers.json")
    store.update("systemone_local", UpdateProviderSettings(
        model="vision-a", endpoint="http://localhost:8001/v1", supports_image=True,
    ))
    store.update("systemone_local", UpdateProviderSettings(
        model="vision-b", endpoint="http://localhost:8001/v1",
    ))
    assert store.get("systemone_local").supports_image


def test_old_custom_image_flag_is_limited_to_its_default_model_and_marked_legacy(tmp_path: Path) -> None:
    store = ProviderSettingsStore(tmp_path / "providers.json")
    connection = store.create(CreateProviderConnection(
        display_name="Old host", protocol="openai_chat", endpoint="https://example.com/v1",
        model="text-model", supports_image=True,
    ))
    assert connection.supports_image
    assert connection.model_capabilities[0].source == "legacy"
    assert AgentProviderRegistry(tmp_path / "providers.json").supports_image(connection.provider_id, "text-model")
    assert not AgentProviderRegistry(tmp_path / "providers.json").supports_image(connection.provider_id, "other-model")
    changed = store.update(connection.provider_id, UpdateProviderSettings(
        display_name="Old host", endpoint="https://example.com/v1", model="other-model",
        models=("text-model", "other-model"), model_capabilities=connection.model_capabilities,
    ))
    assert not changed.supports_image
    assert changed.model_capabilities[0].model_id == "text-model"


def test_legacy_image_flag_migration_keeps_thinking_and_can_be_cleared(tmp_path: Path) -> None:
    store = ProviderSettingsStore(tmp_path / "providers.json")
    connection = store.create(CreateProviderConnection(
        display_name="Old host", protocol="openai_chat", endpoint="https://example.com/v1",
        model="model-a", models=("model-a",), supports_image=True,
        model_capabilities=(ModelCapabilities(model_id="model-a", thinking=True),),
    ))
    legacy = connection.model_capabilities[0]
    assert legacy.image_input is True and legacy.thinking is True
    cleared = store.update(connection.provider_id, UpdateProviderSettings(
        display_name="Old host", endpoint="https://example.com/v1", model="model-a",
        models=("model-a",), supports_image=False,
        model_capabilities=(ModelCapabilities(
            model_id="model-a", image_input=None, thinking=True, image_source="catalog"),),
    ))
    assert cleared.model_capabilities[0].image_input is None
    assert cleared.model_capabilities[0].thinking is True
    assert not cleared.supports_image


def test_probe_reads_only_explicit_per_model_capabilities() -> None:
    response = {"data": [
        {"id": "vision", "input_modalities": ["text", "image"], "capabilities": {"reasoning": True}},
        {"id": "text", "input_modalities": ["text"], "capabilities": {"thinking": False}},
        {"id": "opaque"},
    ]}
    result = asyncio.run(probe_provider(ProbeProviderRequest(
        protocol="systemone", endpoint="http://localhost:8001/v1",
    ), transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))))
    assert result.models == ("vision", "text", "opaque")
    assert [(item.image_input, item.thinking) for item in result.model_capabilities] == [
        (True, True), (False, False), (None, None),
    ]


def test_probe_reads_extended_input_modalities_without_guessing_model_names() -> None:
    response = {"data": [
        {"id": "gateway-vision", "architecture": {"input_modalities": ["text", "image"]},
         "supported_parameters": ["reasoning", "temperature"]},
        {"id": "vision-in-name-only"},
        {"id": "text-model", "architecture": {"modality": "text->text"},
         "supported_parameters": ["temperature"]},
    ]}
    result = asyncio.run(probe_provider(ProbeProviderRequest(
        protocol="openai_chat", endpoint="http://localhost:8001/v1",
    ), transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))))
    assert [(item.image_input, item.thinking) for item in result.model_capabilities] == [
        (True, True), (None, None), (False, False),
    ]


def test_manual_model_capability_is_saved_per_model_and_survives_restart(tmp_path: Path) -> None:
    path = tmp_path / "providers.json"
    store = ProviderSettingsStore(path)
    connection = store.create(CreateProviderConnection(
        display_name="Gateway", protocol="openai_chat", endpoint="https://example.com/v1",
        model="image-model", models=("image-model", "text-model"),
        model_capabilities=(
            ModelCapabilities(model_id="image-model", image_input=True, image_source="manual"),
            ModelCapabilities(model_id="text-model", image_input=False, image_source="catalog"),
        ),
    ))
    restored = ProviderSettingsStore(path).view(connection.provider_id)
    assert restored.model_capabilities[0].image_source == "manual"
    registry = AgentProviderRegistry(path)
    assert registry.supports_image(connection.provider_id, "image-model")
    assert not registry.supports_image(connection.provider_id, "text-model")


def test_systemone_model_test_works_without_models_endpoint() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/systemone"
        body = msgspec.json.decode(request.content)
        assert body["model"] == "local-decision"
        assert set(body["questions"]) == {"probe"}
        return httpx.Response(200, json={
            "model": "local-decision", "answers": {"probe": {"type": "choice", "choice": "connected",
                "probabilities": {"connected": 0.9, "other": 0.1}, "confidence": 0.8}},
            "usage": {"input_tokens": 10, "output_tokens": 2},
        })
    result = asyncio.run(probe_provider(ProbeProviderRequest(
        protocol="systemone", endpoint="http://localhost:8001/v1", model="local-decision", verify_model=True,
    ), transport=httpx.MockTransport(respond)))
    assert result.connected and result.verified == "model"


def test_probe_lists_models_and_tests_inference_without_returning_key() -> None:
    seen: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer private-key"
        seen.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "model-a"}, {"id": "model-b"}]})
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    transport = httpx.MockTransport(respond)
    listed = asyncio.run(probe_provider(ProbeProviderRequest(
        protocol="openai_chat", endpoint="https://example.com/v1", api_key="private-key",
    ), transport=transport))
    tested = asyncio.run(probe_provider(ProbeProviderRequest(
        protocol="openai_chat", endpoint="https://example.com/v1", api_key="private-key", model="model-b", verify_model=True,
    ), transport=transport))
    assert listed.connected and listed.models == ("model-a", "model-b") and listed.verified == "catalog"
    assert tested.connected and tested.verified == "model"
    assert seen == ["/v1/models", "/v1/chat/completions"]
    assert "private-key" not in msgspec.json.encode((listed, tested)).decode()


def test_probe_reports_unsupported_listing_without_claiming_inference() -> None:
    result = asyncio.run(probe_provider(ProbeProviderRequest(
        protocol="openai_chat", endpoint="https://example.com/v1", model="manual-model",
    ), transport=httpx.MockTransport(lambda request: httpx.Response(404))))
    assert not result.connected and result.verified == "none"
    assert "enter a model ID" in result.detail

    invalid_model = asyncio.run(probe_provider(ProbeProviderRequest(
        protocol="openai_chat", endpoint="https://example.com/v1", model="manual-model", verify_model=True,
    ), transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "ok"}))))
    assert not invalid_model.connected and invalid_model.verified == "none"


def test_probe_does_not_send_saved_key_to_a_changed_endpoint(tmp_path: Path) -> None:
    registry = AgentProviderRegistry(tmp_path / "providers.json")
    registry.update_settings("openai", UpdateProviderSettings(model="model-a", api_key="saved-secret"))
    result = asyncio.run(registry.probe(ProbeProviderRequest(
        provider_id="openai", protocol="openai_responses", endpoint="https://other.example/v1",
    )))
    assert not result.connected
    assert result.detail == "API key is required"


def test_default_model_capability_does_not_require_duplicate_model_list(tmp_path: Path) -> None:
    store = ProviderSettingsStore(tmp_path / "providers.json")
    connection = store.create(CreateProviderConnection(
        display_name="Gateway", protocol="openai_chat", endpoint="http://localhost:8001/v1",
        model="vision", model_capabilities=(ModelCapabilities(model_id="vision", image_input=True),),
    ))
    assert connection.models == ("vision",)
    assert connection.supports_image
    assert store.get(connection.provider_id).models == ("vision",)


def test_legacy_decision_protocol_and_images_are_model_specific(tmp_path: Path) -> None:
    registry = AgentProviderRegistry(tmp_path / "providers.json")
    connection = registry.update_settings("systemone_local", UpdateProviderSettings(
        model="vision", endpoint="http://localhost:8001/v1", supports_image=True,
        models=("vision", "unknown"),
    ))
    assert connection.protocol == "systemone"
    assert registry.supports_image(connection.provider_id, "vision")
    assert not registry.supports_image(connection.provider_id, "unknown")


def test_probe_recognizes_reasoning_effort_parameter() -> None:
    result = asyncio.run(probe_provider(ProbeProviderRequest(
        protocol="openai_chat", endpoint="http://localhost:8001/v1",
    ), transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
        "data": [{"id": "reasoner", "supported_parameters": ["reasoning_effort"]}],
    }))))
    assert result.model_capabilities[0].thinking is True
