from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from f8pysdk.service_runtime_tools.inventory.catalog import ServiceCatalog
from f8pysdk.service_runtime_tools.inventory.discovery import load_discovery_into_catalog
from f8pysdk.service_runtime_tools.inventory.index import load_index_into_catalog


def make_index(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "service.yml").write_text(
        'schemaVersion: f8serviceEntry/1\nserviceClass: test.service\nlabel: Test\nversion: 1.0.0\n'
        'launch:\n  command: python\n  args: ["-m", "test_service"]\n  workdir: .\n'
    )
    (root / "describe.json").write_text(json.dumps({
        "service": {"schemaVersion": "f8service/1", "serviceClass": "test.service", "label": "Test"},
        "operators": [],
    }))
    path = root / "index.json"
    path.write_text(json.dumps({
        "schemaVersion": "f8serviceIndex/1", "modelRoot": "models",
        "services": [{"serviceClass": "test.service", "manifests": {"any": "service.yml"}, "describe": "describe.json"}],
    }))
    return path


def test_default_loading_never_scans_hashes_or_runs_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = make_index(tmp_path)
    monkeypatch.setenv("F8_SERVICE_INDEX", str(path))
    with patch("pathlib.Path.rglob", side_effect=AssertionError("scan")), \
         patch("subprocess.run", side_effect=AssertionError("spawn")), \
         patch("f8pysdk.service_runtime_tools.inventory.describe_freshness.source_fingerprint", side_effect=AssertionError("hash")):
        catalog = ServiceCatalog()
        assert load_discovery_into_catalog(catalog=catalog) == ["test.service"]
    entry = catalog.service_entry("test.service")
    assert entry is not None
    assert entry.launch.workdir == str(tmp_path)
    assert entry.launch.env["F8_MODEL_ROOT"] == str(tmp_path / "models")
    # Removing installed source metadata cannot change an already loaded launch snapshot.
    (tmp_path / "service.yml").unlink()
    assert catalog.service_entry("test.service") == entry
    entry.launch.env["F8_MODEL_ROOT"] = "mutated"
    assert catalog.service_entry("test.service").launch.env["F8_MODEL_ROOT"] == str(tmp_path / "models")


def test_index_is_relocatable(tmp_path: Path) -> None:
    path = make_index(tmp_path / "before")
    path.parent.rename(tmp_path / "after")
    catalog = ServiceCatalog()
    load_index_into_catalog(path=tmp_path / "after" / "index.json", catalog=catalog)
    assert catalog.service_entry("test.service").launch.workdir == str(tmp_path / "after")


def test_missing_describe_does_not_fall_back_to_subprocess(tmp_path: Path) -> None:
    path = make_index(tmp_path)
    (tmp_path / "describe.json").unlink()
    with patch("subprocess.run", side_effect=AssertionError("spawn")), pytest.raises(FileNotFoundError, match="install_services"):
        load_index_into_catalog(path=path, catalog=ServiceCatalog())


def test_index_rejects_duplicates_and_platform_guessing(tmp_path: Path) -> None:
    path = make_index(tmp_path)
    raw = json.loads(path.read_text())
    raw["services"].append(raw["services"][0])
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="Duplicate"):
        load_index_into_catalog(path=path, catalog=ServiceCatalog())
    raw["services"].pop()
    raw["services"][0]["manifests"] = {"win32": "does-not-exist.yml"}
    path.write_text(json.dumps(raw))
    with patch("f8pysdk.service_runtime_tools.inventory.index.sys.platform", "linux"):
        assert load_index_into_catalog(path=path, catalog=ServiceCatalog()) == []


def test_invalid_description_is_rejected(tmp_path: Path) -> None:
    path = make_index(tmp_path)
    (tmp_path / "describe.json").write_text('{"service": {"serviceClass": "test.wrong", "label": "Wrong"}}')
    with pytest.raises(ValueError, match="mismatch"):
        load_index_into_catalog(path=path, catalog=ServiceCatalog())
