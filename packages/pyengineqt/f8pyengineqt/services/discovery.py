from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from f8pysdk import F8OperatorSpec, F8ServiceSpec, F8ServiceEntry, F8ServiceDescribe
# from .service_entry import F8ServiceDescribeEntryLaunch


@dataclass(frozen=True)
class DiscoveryPaths:
    roots: list[Path]


class DiscoveryError(Exception):
    pass


def _read_yaml(path: Path) -> Any:
    try:
        raw = path.read_text("utf-8")
    except Exception as exc:
        raise DiscoveryError(f"Failed to read {path}: {exc}") from exc
    try:
        return yaml.safe_load(raw) if raw.strip() else None
    except Exception as exc:
        raise DiscoveryError(f"Failed to parse YAML {path}: {exc}") from exc


def find_service_dirs(roots: Iterable[Path]) -> list[Path]:
    """
    Find directories containing a `service.yml`.

    Discovery structure:
      <root>/<serviceClass>/service.yml
      <root>/<serviceClass>/operators.yml   (optional)

    Note:
    - `serviceClass` is slash-separated (eg. `f8/engine`), so services may live
      in nested directories like `<root>/f8/engine/service.yml`.
    """
    found: list[Path] = []
    for root in roots:
        r = Path(root)
        if not r.exists() or not r.is_dir():
            continue
        try:
            for svc_file in sorted(r.rglob("service.yml")):
                if svc_file.is_file():
                    found.append(svc_file.parent)
        except Exception:
            # Fallback to the immediate children layout.
            for child in sorted(r.iterdir()):
                if not child.is_dir():
                    continue
                if (child / "service.yml").is_file():
                    found.append(child)
    return found


def load_service_spec(service_dir: Path) -> F8ServiceSpec:
    data = _read_yaml(Path(service_dir) / "service.yml")
    if not isinstance(data, dict):
        raise DiscoveryError(f"{service_dir}/service.yml must be a YAML mapping")
    try:
        return F8ServiceSpec.model_validate(data)
    except Exception as exc:
        raise DiscoveryError(f"Invalid service.yml in {service_dir}: {exc}") from exc


def load_service_entry(service_dir: Path) -> F8ServiceEntry:
    """
    Load a minimal discovery entry from `service.yml`.

    Supported forms:
    - `schemaVersion: f8serviceEntry/1` + `launch: {...}`
    - shorthand: `command: ...` + optional `args/env/workdir` (will be mapped into `launch`)
    """
    data = _read_yaml(Path(service_dir) / "service.yml")
    if not isinstance(data, dict):
        raise DiscoveryError(f"{service_dir}/service.yml must be a YAML mapping")

    # Shorthand mapping.
    if "launch" not in data and "command" in data:
        launch = F8ServiceEntry.model_validate(
            {
                "command": data.get("command"),
                "args": data.get("args") or [],
                "env": data.get("env") or {},
                "workdir": data.get("workdir") or "./",
            }
        )
        data = dict(data)
        data["launch"] = launch.model_dump(mode="json")

    try:
        return F8ServiceEntry.model_validate(data)
    except Exception as exc:
        raise DiscoveryError(f"Invalid service entry in {service_dir}: {exc}") from exc


def load_operator_specs(service_dir: Path, *, service_class: str) -> list[F8OperatorSpec]:
    """
    Load operator specs from `<service_dir>/operators.yml` if present.

    `operators.yml` can be:
    - a YAML list of operator spec mappings
    - a YAML mapping with key `operators: [...]`

    Each operator spec may omit `serviceClass`; it will be filled from `service.yml`.
    """
    path = Path(service_dir) / "operators.yml"
    if not path.is_file():
        return []

    data = _read_yaml(path)
    items: list[Any] = []
    if data is None:
        return []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        ops = data.get("operators")
        if isinstance(ops, list):
            items = ops
        else:
            # allow single-spec mapping for convenience
            items = [data]
    else:
        raise DiscoveryError(f"{path} must be a YAML list or mapping")

    specs: list[F8OperatorSpec] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        merged = dict(item)
        merged.setdefault("serviceClass", service_class)
        try:
            spec = F8OperatorSpec.model_validate(merged)
        except Exception as exc:
            raise DiscoveryError(f"Invalid operator spec in {path}: {exc}") from exc
        if str(getattr(spec, "serviceClass", "") or "").strip() != str(service_class).strip():
            raise DiscoveryError(f"{path}: operator spec serviceClass must match service.yml ({service_class})")
        specs.append(spec)

    # stable ordering (nice palette)
    try:
        specs.sort(key=lambda s: str(getattr(s, "operatorClass", "") or ""))
    except Exception:
        pass
    return specs
