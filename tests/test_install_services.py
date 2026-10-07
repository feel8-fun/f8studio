from f8pysdk._specs.builtin_fields import normalize_describe_payload_dict

from pathlib import Path
import json
import os

import pytest

from scripts.install_services import copy_verified, migrate_resources
from scripts.extension_workspace import workspace_index


def _describe_payload(service_class: str) -> str:
    return json.dumps(normalize_describe_payload_dict({
        'service': {'schemaVersion': 'f8service/1', 'serviceClass': service_class, 'label': service_class},
        'operators': [],
    }))


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
    if os.name != 'nt':
        assert installed.stat().st_mode & 0o111
    assert (tmp_path / "user-config/f8.implayer/imgui.ini").read_text() == "user settings"
    assert executable.exists() and config.exists()


@pytest.fixture
def service_index(tmp_path: Path) -> Path:
    import json
    import yaml

    root = Path.cwd()
    base_index = workspace_index()
    index = json.loads(base_index.read_text())
    index.pop('packageRoot', None)
    index["services"] = [item for item in index["services"] if item["serviceClass"] in {"f8.pyexpr", "f8.pyscript"}]
    for item in index["services"]:
        original = Path(item['manifests']['any'].replace('${F8_PACKAGE_ROOT}', str(root)))
        entry = yaml.safe_load(original.read_text())
        entry["launch"]["workdir"] = str(root / "extensions/f8pyengine")
        manifest = tmp_path / (item["serviceClass"] + ".yml")
        manifest.write_text(yaml.safe_dump(entry))
        item["manifests"] = {"any": str(manifest)}
        item["describe"] = str(tmp_path / (item["serviceClass"] + ".json"))
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))
    return path


def test_refresh_prepares_shared_environment_once_before_timed_describes(service_index: Path) -> None:
    import subprocess
    from unittest.mock import patch
    from scripts.install_services import install

    payloads = {name: _describe_payload(name) for name in ('f8.pyexpr', 'f8.pyscript')}
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[1] == "install":
            assert "timeout" not in kwargs
            assert command.count("-e") == 1
            assert "--locked" in command
            return subprocess.CompletedProcess(command, 0)
        assert calls[0][1] == "install"
        assert "--no-install" in command and "--frozen" in command
        assert kwargs["timeout"] == 30.0
        name = "f8.pyexpr" if "f8pyexpr" in command else "f8.pyscript"
        return subprocess.CompletedProcess(command, 0, stdout=payloads[name])

    with patch("scripts.install_services.subprocess.run", side_effect=run):
        assert install(service_index, refresh=True, service_classes=set()) == 2
    assert len(calls) == 3
    assert (service_index.parent / "f8.pyexpr.json").is_file()


@pytest.mark.parametrize("args", [["run", "f8pyexpr"], ["run", "-e", "missing", "f8pyexpr"],
                                   ["run", "-e", "ci", "f8pyexpr"]])
def test_invalid_binding_fails_before_any_subprocess(service_index: Path, args: list[str]) -> None:
    import json
    import yaml
    from unittest.mock import patch
    from scripts.install_services import install

    index = json.loads(service_index.read_text())
    manifest = Path(index["services"][-1]["manifests"]["any"])
    entry = yaml.safe_load(manifest.read_text())
    entry["launch"]["args"] = args
    manifest.write_text(yaml.safe_dump(entry))
    with patch("scripts.install_services.subprocess.run") as run:
        with pytest.raises(ValueError):
            install(service_index, refresh=True, service_classes=set())
        run.assert_not_called()
    assert not list(service_index.parent.glob("f8.*.json"))


@pytest.mark.parametrize("timeout", [False, True])
def test_describe_failure_reports_service_and_captured_output(timeout: bool) -> None:
    import subprocess
    from unittest.mock import patch
    from f8pysdk.specs import F8ServiceEntry, F8ServiceLaunchSpec
    from scripts.install_services import describe_service

    entry = F8ServiceEntry(serviceClass="f8.example", launch=F8ServiceLaunchSpec(
        command="pixi", args=["run", "-e", "onnx", "example"], workdir="."))
    error = (subprocess.TimeoutExpired("example", 30, output=b"partial", stderr=b"reason") if timeout
             else subprocess.CalledProcessError(1, "example", output="partial", stderr="reason"))
    with patch("scripts.install_services.subprocess.run", side_effect=error):
        with pytest.raises(RuntimeError, match="f8.example") as caught:
            describe_service(entry)
    assert "partial" in str(caught.value) and "reason" in str(caught.value)
    assert caught.value.__cause__ is error


def test_failed_refresh_keeps_all_previous_descriptions(service_index: Path) -> None:
    import json
    import subprocess
    from unittest.mock import patch
    from scripts.install_services import install

    index = json.loads(service_index.read_text())
    for item in index["services"]:
        Path(item["describe"]).write_text("previous description")
    first_class = index["services"][0]["serviceClass"]
    payload = _describe_payload(first_class)
    with patch("scripts.install_services.subprocess.run", side_effect=[
        subprocess.CompletedProcess([], 0, stdout=payload),
        subprocess.CalledProcessError(1, "second service", stderr="import failed"),
    ]) as run:
        with pytest.raises(RuntimeError, match="import failed"):
            install(service_index, refresh=True, service_classes=set(), no_install=True)
        assert run.call_count == 2
        assert all(call.args[0][1] == "run" for call in run.call_args_list)
    for item in index["services"]:
        assert Path(item["describe"]).read_text() == "previous description"
