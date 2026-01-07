from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .discovery import DiscoveryError, find_service_dirs, load_operator_specs, load_service_spec
from .service_operator_registry import ServiceOperatorSpecRegistry
from .service_registry import ServiceSpecRegistry


def _default_roots() -> list[Path]:
    """
    Default discovery roots.

    - `F8_SERVICE_DISCOVERY_DIRS` (os.pathsep-separated) overrides.
    - otherwise, use `<repo>/services` if it exists (when running from source).
    """
    env = (os.environ.get("F8_SERVICE_DISCOVERY_DIRS") or "").strip()
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p.strip()]

    # Try to locate repo root when running from source.
    try:
        repo_root = Path(__file__).resolve().parents[4]  # .../packages/pyengineqt
        candidate = repo_root / "services"
        if candidate.is_dir():
            return [candidate]
    except Exception:
        pass
    return []


def load_discovery_into_registries(*, roots: list[Path] | None = None, overwrite: bool = True) -> list[str]:
    """
    Load `service.yml` + optional `operators.yml` into in-process registries.

    Returns the list of discovered serviceClass entries.
    """
    roots = roots if roots is not None else _default_roots()
    svc_registry = ServiceSpecRegistry.instance()
    op_registry = ServiceOperatorSpecRegistry.instance()

    found: list[str] = []
    for service_dir in find_service_dirs(roots):
        try:
            svc = load_service_spec(service_dir)
        except DiscoveryError:
            continue
        try:
            svc_registry.register(svc, overwrite=overwrite)
        except Exception:
            pass
        found.append(str(svc.serviceClass))

        try:
            ops = load_operator_specs(service_dir, service_class=str(svc.serviceClass))
        except DiscoveryError:
            ops = []
        if ops:
            try:
                op_registry.register_many(str(svc.serviceClass), ops, overwrite=overwrite)
            except Exception:
                # best-effort; per-spec errors already validated in loader.
                pass
    return found


def default_discovery_roots() -> list[Path]:
    return _default_roots()

