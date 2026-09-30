from pathlib import Path

import pytest

from scripts.install_services import copy_verified, migrate_resources


def test_model_migration_preserves_sources_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "services" / "f8" / "dl" / "weights" / "test.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"model contents")
    destination = tmp_path / "resources"
    assert migrate_resources(tmp_path / "services", destination) == 1
    assert migrate_resources(tmp_path / "services", destination) == 1
    assert source.read_bytes() == (destination / "onnx" / "test.onnx").read_bytes()


def test_migration_refuses_to_overwrite_different_resource(tmp_path: Path) -> None:
    source = tmp_path / "old.onnx"
    target = tmp_path / "new.onnx"
    source.write_bytes(b"old")
    target.write_bytes(b"new")
    with pytest.raises(ValueError, match="conflict"):
        copy_verified(source, target)
    assert target.read_bytes() == b"new"


def test_runtime_layout_migration_preserves_executable_and_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.install_services import migrate_runtime_layout
    old = tmp_path / "old"
    executable = old / "f8" / "cppengine" / "linux" / "engine"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"runtime")
    executable.chmod(0o755)
    config = old / "f8" / "implayer" / "imgui.ini"
    config.parent.mkdir(parents=True)
    config.write_text("user settings")
    monkeypatch.setenv("F8_CONFIG_ROOT", str(tmp_path / "user-config"))
    assert migrate_runtime_layout(old, tmp_path / "install") == 2
    installed = tmp_path / "install/runtime/bundles/f8.cppengine/0.0.1/linux/engine"
    assert installed.read_bytes() == executable.read_bytes()
    assert installed.stat().st_mode & 0o111
    assert (tmp_path / "user-config/f8.implayer/imgui.ini").read_text() == "user settings"
    assert executable.exists() and config.exists()


def test_document_path_migration_only_changes_known_model_defaults() -> None:
    from scripts.migrate_service_paths import migrate_document
    document = {"schemaVersion": "f8studio-document/2", "nodes": [
        {"serviceClass": "f8.dl.detector", "stateValues": {"weightsDir": "services/f8/dl/weights", "custom": "services/f8/dl/weights"}},
        {"serviceClass": "f8.dl.detector", "stateValues": {"weightsDir": "/my/models"}},
        {"serviceClass": "f8.cvkit.tracking", "stateValues": {"modelDir": "models"}},
        {"serviceClass": "test.external", "stateValues": {"weightsDir": "services/f8/dl/weights"}},
    ]}
    assert migrate_document(document) == 2
    assert document["nodes"][0]["stateValues"] == {"weightsDir": "", "custom": "services/f8/dl/weights"}
    assert document["nodes"][1]["stateValues"]["weightsDir"] == "/my/models"
    assert document["nodes"][3]["stateValues"]["weightsDir"] == "services/f8/dl/weights"
    assert migrate_document(document) == 0
