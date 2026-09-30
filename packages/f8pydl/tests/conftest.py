from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def installed_model_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    # Tests use the checked-in model definitions, without guessing runtime paths.
    root = Path(__file__).resolve().parents[3] / "resources" / "models"
    monkeypatch.setenv("F8_MODEL_ROOT", str(root))
