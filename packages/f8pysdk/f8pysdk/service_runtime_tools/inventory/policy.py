from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger(__name__)

SERVICE_DISCOVERY_POLICY_SCHEMA_VERSION = "f8serviceDiscoveryPolicy/1"
SERVICE_DISCOVERY_POLICY_ENV = "F8_SERVICE_DISCOVERY_POLICY"
DISABLED_SERVICE_CLASSES_ENV = "F8_DISABLED_SERVICE_CLASSES"
DEFAULT_SERVICE_DISCOVERY_POLICY_REL_PATH = Path("config") / "service_discovery_policy.yml"
_YAML_SAFE_LOADER = yaml.SafeLoader


@dataclass(frozen=True)
class ServiceDiscoveryPolicy:
    disabled_service_classes: tuple[str, ...] = ()


def split_service_class_values(values: Sequence[str]) -> tuple[str, ...]:
    service_classes: list[str] = []
    for value in values:
        for comma_part in str(value or "").split(","):
            for pathsep_part in comma_part.split(os.pathsep):
                item = pathsep_part.strip()
                if item:
                    service_classes.append(item)
    return tuple(service_classes)


def merge_disabled_service_classes(
    *,
    policy: ServiceDiscoveryPolicy | None = None,
    explicit_service_classes: Sequence[str] | None = None,
    include_env: bool = True,
) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    sources: list[Sequence[str]] = []
    if policy is not None:
        sources.append(policy.disabled_service_classes)
    if explicit_service_classes is not None:
        sources.append(explicit_service_classes)
    if include_env:
        sources.append((os.environ.get(DISABLED_SERVICE_CLASSES_ENV) or "",))

    for source in sources:
        for service_class in split_service_class_values(source):
            if service_class in seen:
                continue
            seen.add(service_class)
            merged.append(service_class)
    return tuple(merged)


def repo_root_from_path(start_path: Path) -> Path | None:
    for candidate in (Path(start_path).resolve(), *Path(start_path).resolve().parents):
        if (candidate / "pixi.toml").is_file() and (candidate / "config").is_dir():
            return candidate
    return None


def default_service_discovery_policy_path(*, start_path: Path | None = None) -> Path | None:
    env_path_raw = str(os.environ.get(SERVICE_DISCOVERY_POLICY_ENV) or "").strip()
    if env_path_raw:
        return Path(env_path_raw).expanduser().resolve()

    base_path = Path.cwd() if start_path is None else Path(start_path)
    repo_root = repo_root_from_path(base_path)
    if repo_root is None:
        return None
    return (repo_root / DEFAULT_SERVICE_DISCOVERY_POLICY_REL_PATH).resolve()


def _required_string_list(payload: dict[str, Any], key: str, *, policy_path: Path) -> tuple[str, ...]:
    raw = payload.get(key, [])
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{policy_path}: {key} must be a list of strings")

    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(f"{policy_path}: {key}[{index}] must be a string")
        value = item.strip()
        if value:
            values.append(value)
    return tuple(values)


def load_service_discovery_policy(policy_path: Path) -> ServiceDiscoveryPolicy:
    resolved_policy_path = Path(policy_path).expanduser().resolve()
    try:
        raw_text = resolved_policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read service discovery policy {resolved_policy_path}: {exc}") from exc

    try:
        payload = yaml.load(raw_text, Loader=_YAML_SAFE_LOADER) if raw_text.strip() else {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse service discovery policy {resolved_policy_path}: {exc}") from exc

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved_policy_path}: policy file must contain a YAML mapping")

    schema_version = payload.get("schemaVersion")
    if schema_version != SERVICE_DISCOVERY_POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"{resolved_policy_path}: schemaVersion must be {SERVICE_DISCOVERY_POLICY_SCHEMA_VERSION}"
        )

    return ServiceDiscoveryPolicy(
        disabled_service_classes=_required_string_list(
            payload,
            "disabledServiceClasses",
            policy_path=resolved_policy_path,
        ),
    )


def load_default_service_discovery_policy(*, start_path: Path | None = None) -> ServiceDiscoveryPolicy:
    policy_path = default_service_discovery_policy_path(start_path=start_path)
    if policy_path is None or not policy_path.is_file():
        return ServiceDiscoveryPolicy()
    try:
        return load_service_discovery_policy(policy_path)
    except ValueError:
        logger.exception("Failed to load service discovery policy: %s", policy_path)
        return ServiceDiscoveryPolicy()


__all__ = [
    "DEFAULT_SERVICE_DISCOVERY_POLICY_REL_PATH",
    "DISABLED_SERVICE_CLASSES_ENV",
    "SERVICE_DISCOVERY_POLICY_ENV",
    "SERVICE_DISCOVERY_POLICY_SCHEMA_VERSION",
    "ServiceDiscoveryPolicy",
    "default_service_discovery_policy_path",
    "load_default_service_discovery_policy",
    "load_service_discovery_policy",
    "merge_disabled_service_classes",
    "split_service_class_values",
]
