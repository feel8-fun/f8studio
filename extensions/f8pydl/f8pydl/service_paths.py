from __future__ import annotations

from pathlib import Path

from f8pysdk.resource_paths import model_root


def default_weights_dir() -> Path:
    return model_root() / "onnx"


def resolve_user_path(raw: str) -> Path:
    """Empty selects installed models; explicit relative user paths use the cwd."""
    return Path(raw).expanduser().resolve() if raw.strip() else default_weights_dir()
