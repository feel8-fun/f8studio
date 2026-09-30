from pathlib import Path

import pytest

from f8pydl import service_paths
from f8pysdk.resource_paths import model_root


def test_explicit_relative_user_path_uses_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert service_paths.resolve_user_path("models/local.yaml") == tmp_path / "models/local.yaml"


def test_installed_model_root_is_independent_of_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "installed"
    monkeypatch.setenv("F8_MODEL_ROOT", str(root))
    monkeypatch.chdir(tmp_path)
    assert service_paths.default_weights_dir() == root / "onnx"
    assert service_paths.resolve_user_path("") == root / "onnx"


def test_relative_installed_root_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("F8_MODEL_ROOT", "relative/models")
    with pytest.raises(ValueError, match="absolute"):
        model_root()
