from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from f8pysdk.codec import dump_json
from f8pysdk.service_runtime_tools.inventory.catalog import ServiceCatalog
from f8pysdk.service_runtime_tools.inventory.describe import (
    discovery_parallelism,
    discovery_slow_ms_default,
    read_static_describe_payload,
)
from f8pysdk.service_runtime_tools.inventory.discovery import load_discovery_into_catalog
from f8pysdk.service_runtime_tools.inventory.entry import find_service_dirs, load_service_entry
from f8pysdk.service_runtime_tools.inventory.policy import (
    SERVICE_DISCOVERY_POLICY_ENV,
    ServiceDiscoveryPolicy,
    load_default_service_discovery_policy,
    load_service_discovery_policy,
    merge_disabled_service_classes,
)


def test_find_service_dirs_discovers_nested_service_files(tmp_path: Path) -> None:
    root = tmp_path / "services"
    alpha = root / "f8" / "alpha"
    beta = root / "f8" / "beta"
    gamma = root / "f8" / "gamma"
    nested = root / "f8" / "group" / "nested"
    for service_dir, filename in (
        (alpha, "service.yml"),
        (beta, "service.linux.yml"),
        (gamma, "service.mac.yml"),
        (nested, "service.win.yml"),
    ):
        service_dir.mkdir(parents=True)
        (service_dir / filename).write_text("launch:\n  command: echo\n", encoding="utf-8")
    (root / "f8" / "not_service").mkdir(parents=True)
    (root / "f8" / "not_service" / "other.yml").write_text("{}\n", encoding="utf-8")

    found = find_service_dirs([root])

    assert found == sorted([alpha.resolve(), beta.resolve(), gamma.resolve(), nested.resolve()])


def test_find_service_dirs_ignores_missing_roots(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    assert find_service_dirs([missing]) == []


def test_find_service_dirs_logs_recursive_scan_fallback(tmp_path: Path, monkeypatch: Any, caplog: pytest.LogCaptureFixture) -> None:
    root = tmp_path / "services"
    service_dir = root / "alpha"
    service_dir.mkdir(parents=True)
    (service_dir / "service.yml").write_text("launch:\n  command: echo\n", encoding="utf-8")
    original_rglob = Path.rglob

    def _failing_rglob(path: Path, pattern: str) -> Any:
        if path == root:
            raise OSError("scan failed")
        return original_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", _failing_rglob)

    caplog.set_level("DEBUG", logger="f8pysdk.service_runtime_tools.inventory.entry")
    found = find_service_dirs([root])

    assert found == [service_dir.resolve()]
    assert "recursive service discovery failed" in caplog.text


def _write_entry(service_dir: Path, *, workdir: str = "./", command: str = "runner") -> None:
    service_dir.mkdir(parents=True)
    (service_dir / "service.yml").write_text(
        "schemaVersion: f8serviceEntry/1\n"
        "serviceClass: f8.tests.entry\n"
        "label: Entry\n"
        "version: 0.0.1\n"
        "launch:\n"
        f"  command: {command}\n"
        "  args: []\n"
        "  env: {}\n"
        f"  workdir: {workdir}\n",
        encoding="utf-8",
    )


def _write_discoverable_service(service_dir: Path, *, service_class: str) -> None:
    service_dir.mkdir(parents=True)
    (service_dir / "service.yml").write_text(
        "schemaVersion: f8serviceEntry/1\n"
        f"serviceClass: {service_class}\n"
        "label: Entry\n"
        "version: 0.0.1\n"
        "launch:\n"
        "  command: runner\n"
        "  args: []\n"
        "  env: {}\n"
        "  workdir: ./\n",
        encoding="utf-8",
    )
    (service_dir / "describe.json").write_text(
        "{\n"
        '  "service": {\n'
        '    "schemaVersion": "f8service/1",\n'
        f'    "serviceClass": "{service_class}",\n'
        '    "label": "Entry",\n'
        '    "version": "0.0.1"\n'
        "  },\n"
        '  "operators": []\n'
        "}\n",
        encoding="utf-8",
    )


def test_force_dynamic_discovery_bypasses_static_describe(tmp_path: Path, monkeypatch: Any) -> None:
    service_class = "f8.tests.dynamic"
    service_dir = tmp_path / "services" / "dynamic"
    _write_discoverable_service(service_dir, service_class=service_class)
    catalog = ServiceCatalog()
    calls: list[object] = []
    dynamic_payload = {
        "service": {
            "schemaVersion": "f8service/1", "serviceClass": service_class,
            "label": "Entry", "version": "0.0.1",
        },
        "operators": [{
            "schemaVersion": "f8operator/1", "serviceClass": service_class,
            "operatorClass": "f8.tests.new_node", "label": "New Node", "version": "0.0.1",
        }],
    }

    def run_describe(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps(dynamic_payload), stderr="")

    monkeypatch.setattr("f8pysdk.service_runtime_tools.inventory.describe.subprocess.run", run_describe)
    load_discovery_into_catalog(roots=[service_dir], catalog=catalog)
    assert not calls
    assert not catalog.operators.has(service_class, "f8.tests.new_node")

    load_discovery_into_catalog(
        roots=[service_dir], catalog=catalog, force_dynamic_service_classes=(service_class,),
    )
    assert len(calls) == 1
    assert catalog.operators.has(service_class, "f8.tests.new_node")


def test_load_service_entry_matches_for_relative_and_absolute_service_dir(tmp_path: Path, monkeypatch: Any) -> None:
    root = tmp_path / "services"
    service_dir = root / "f8" / "entry"
    _write_entry(service_dir, workdir="../entry", command="runner")
    monkeypatch.chdir(tmp_path)

    relative_payload = dump_json(load_service_entry(Path('services/f8/entry')))
    absolute_payload = dump_json(load_service_entry(service_dir.resolve()))

    assert relative_payload == absolute_payload
    assert relative_payload["launch"]["workdir"] == str(service_dir.resolve())


def test_load_service_entry_logs_platform_candidate_path_failure(
    tmp_path: Path,
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service_dir = tmp_path / "services" / "f8" / "entry"
    service_dir.mkdir(parents=True)
    (service_dir / "service.linux.yml").write_text(
        "schemaVersion: f8serviceEntry/1\n"
        "serviceClass: f8.tests.platform\n"
        "launch:\n"
        "  command: ./runner.py\n"
        "  workdir: ./\n",
        encoding="utf-8",
    )
    (service_dir / "service.yml").write_text(
        "schemaVersion: f8serviceEntry/1\n"
        "serviceClass: f8.tests.fallback\n"
        "launch:\n"
        "  command: runner\n"
        "  workdir: ./\n",
        encoding="utf-8",
    )
    original_resolve = Path.resolve

    def _failing_resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        if path.name == "runner.py":
            raise OSError("resolve failed")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _failing_resolve)
    monkeypatch.setattr(
        "f8pysdk.service_runtime_tools.inventory.entry._platform_service_yml_names",
        lambda: ["service.linux.yml"],
    )

    caplog.set_level("DEBUG", logger="f8pysdk.service_runtime_tools.inventory.entry")
    entry = load_service_entry(service_dir)

    assert str(entry.serviceClass) == "f8.tests.platform"
    assert "platform service entry command probe failed" in caplog.text


def test_read_static_describe_payload_logs_invalid_json(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    service_dir = tmp_path / "services" / "f8" / "entry"
    _write_entry(service_dir)
    (service_dir / "describe.json").write_text("{not-json", encoding="utf-8")
    entry = load_service_entry(service_dir)

    caplog.set_level("DEBUG", logger="f8pysdk.service_runtime_tools.inventory.describe")
    payload, source = read_static_describe_payload(service_dir, entry)

    assert payload is None
    assert source is None
    assert "Failed to read static describe JSON" in caplog.text


def test_discovery_env_parse_failures_are_logged(monkeypatch: Any, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("F8_DESCRIBE_JOBS", "many")
    monkeypatch.setenv("F8_DISCOVERY_SLOW_MS", "slow")

    caplog.set_level("DEBUG", logger="f8pysdk.service_runtime_tools.inventory.describe")

    assert discovery_parallelism(4) == 1
    assert discovery_slow_ms_default() == 0.0
    assert "Invalid discovery parallelism env value" in caplog.text
    assert "Invalid discovery slow threshold env value" in caplog.text


def test_load_discovery_into_catalog_skips_disabled_service_class(tmp_path: Path) -> None:
    root = tmp_path / "services"
    disabled_dir = root / "f8" / "disabled"
    enabled_dir = root / "f8" / "enabled"
    _write_discoverable_service(disabled_dir, service_class="f8.tests.disabled")
    _write_discoverable_service(enabled_dir, service_class="f8.tests.enabled")
    catalog = ServiceCatalog()
    catalog.clear()

    try:
        found = load_discovery_into_catalog(
            roots=[root],
            catalog=catalog,
            disabled_service_classes=("f8.tests.disabled",),
        )

        assert found == ["f8.tests.enabled"]
        assert not catalog.services.has("f8.tests.disabled")
        assert catalog.services.has("f8.tests.enabled")
    finally:
        catalog.clear()


def test_load_service_discovery_policy_reads_disabled_service_classes(tmp_path: Path) -> None:
    policy_path = tmp_path / "service_discovery_policy.yml"
    policy_path.write_text(
        "schemaVersion: f8serviceDiscoveryPolicy/1\n"
        "disabledServiceClasses:\n"
        "  - f8.cppengine\n"
        "  - f8.tests.experimental\n",
        encoding="utf-8",
    )

    policy = load_service_discovery_policy(policy_path)

    assert policy.disabled_service_classes == ("f8.cppengine", "f8.tests.experimental")


def test_load_default_service_discovery_policy_uses_env_path(tmp_path: Path, monkeypatch: Any) -> None:
    policy_path = tmp_path / "policy.yml"
    policy_path.write_text(
        "schemaVersion: f8serviceDiscoveryPolicy/1\n"
        "disabledServiceClasses:\n"
        "  - f8.cppengine\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(SERVICE_DISCOVERY_POLICY_ENV, str(policy_path))

    policy = load_default_service_discovery_policy(start_path=tmp_path)

    assert policy.disabled_service_classes == ("f8.cppengine",)


def test_merge_disabled_service_classes_dedupes_policy_explicit_and_env(monkeypatch: Any) -> None:
    policy_path_value = "f8.policy,f8.shared"
    monkeypatch.setenv("F8_DISABLED_SERVICE_CLASSES", f"{policy_path_value}{os.pathsep}f8.env")

    merged = merge_disabled_service_classes(
        policy=ServiceDiscoveryPolicy(disabled_service_classes=("f8.policy",)),
        explicit_service_classes=("f8.cli", "f8.shared"),
        include_env=True,
    )

    assert merged == ("f8.policy", "f8.cli", "f8.shared", "f8.env")


def test_checkout_describe_requires_matching_source_fingerprint(tmp_path: Path) -> None:
    from f8pysdk.service_runtime_tools.inventory.describe_freshness import static_is_fresh, write_freshness

    (tmp_path / "pixi.toml").write_text("[workspace]\n")
    source = tmp_path / "packages" / "example.py"
    source.parent.mkdir()
    source.write_text("value = 1\n")
    service = tmp_path / "services" / "example"
    service.mkdir(parents=True)
    assert not static_is_fresh(service)
    write_freshness(service)
    assert static_is_fresh(service)
    source.write_text("value = 2\n")
    assert not static_is_fresh(service)
    # Packaged descriptions remain supported without a source checkout.
    (tmp_path / "pixi.toml").unlink()
    assert static_is_fresh(service)


def test_discovery_rejects_invalid_authoring_instead_of_returning_raw_payload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    service_dir = tmp_path / "invalid"
    _write_discoverable_service(service_dir, service_class="f8.tests.invalid")
    path = service_dir / "describe.json"
    payload = json.loads(path.read_text())
    payload["service"]["dataInPorts"] = [{"name": "old", "valueSchema": {"type": "any"}}]
    path.write_text(json.dumps(payload))
    catalog = ServiceCatalog()
    with caplog.at_level("ERROR"):
        load_discovery_into_catalog(roots=[service_dir], catalog=catalog)
    assert any("Describe payload validation failed" in record.message and record.exc_info for record in caplog.records)


@pytest.mark.parametrize("path,value", [
    (("operators",), [None]),
    (("operators",), {}),
    (("service", "stateFields"), ["old-field"]),
    (("service", "dataOutPorts"), "old-port"),
])
def test_discovery_rejects_malformed_descriptor_collections(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, path: tuple[str, ...], value: Any,
) -> None:
    service_dir = tmp_path / "invalid"
    _write_discoverable_service(service_dir, service_class="f8.tests.invalid")
    describe_path = service_dir / "describe.json"
    payload = json.loads(describe_path.read_text())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    describe_path.write_text(json.dumps(payload))
    with caplog.at_level("ERROR"):
        load_discovery_into_catalog(roots=[service_dir], catalog=ServiceCatalog())
    assert any("Describe payload validation failed" in record.message and record.exc_info for record in caplog.records)
