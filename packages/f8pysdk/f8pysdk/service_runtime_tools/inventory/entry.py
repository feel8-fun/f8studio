from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
import msgspec

from f8pysdk.codec import dump_json, validate_as
from f8pysdk.specs import F8ServiceEntry, F8ServiceLaunchSpec

_YAML_SAFE_LOADER = yaml.SafeLoader
logger = logging.getLogger(__name__)
_ENTRY_PATH_ERRORS = (OSError, RuntimeError, TypeError, ValueError)
_ENTRY_YAML_PARSE_ERRORS = (TypeError, ValueError, yaml.YAMLError)
_ENTRY_YAML_READ_ERRORS = (OSError, UnicodeError)
_ENTRY_CANDIDATE_LOAD_ERRORS = (OSError, RuntimeError, TypeError, ValueError)
_ENTRY_VALIDATION_ERRORS = (TypeError, ValueError, msgspec.ValidationError)


def _default_roots() -> list[Path]:
    env = (os.environ.get("F8_SERVICE_DISCOVERY_DIRS") or "").strip()
    if env:
        return [Path(p).expanduser().resolve() for p in env.split(os.pathsep) if p.strip()]

    try:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "config" / "services"
            if candidate.is_dir():
                return [candidate.resolve()]
    except _ENTRY_PATH_ERRORS as exc:
        logger.debug("default service discovery root probe failed", exc_info=exc)
    return []


def default_discovery_roots() -> list[Path]:
    return _default_roots()


def _read_yaml(path: Path) -> Any:
    try:
        raw = path.read_text("utf-8")
    except _ENTRY_YAML_READ_ERRORS as exc:
        raise ValueError(f"Failed to read {path}: {exc}") from exc
    try:
        return yaml.load(raw, Loader=_YAML_SAFE_LOADER) if raw.strip() else None
    except _ENTRY_YAML_PARSE_ERRORS as exc:
        raise ValueError(f"Failed to parse YAML {path}: {exc}") from exc


def _platform_service_yml_names() -> list[str]:
    if os.name == "nt" or sys.platform.startswith("win"):
        return ["service.win.yml"]
    if sys.platform.startswith("darwin"):
        return ["service.mac.yml"]
    return ["service.linux.yml"]


def _service_yml_candidates(service_dir: Path) -> list[Path]:
    platform_names = _platform_service_yml_names()
    all_platform_names = ["service.win.yml", "service.linux.yml", "service.mac.yml"]
    fallback_names: list[str] = []
    for name in all_platform_names:
        if name not in platform_names:
            fallback_names.append(name)
    ordered_names = platform_names + ["service.yml"] + fallback_names
    return [service_dir / name for name in ordered_names]


def find_service_dirs(roots: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        resolved_root = Path(root).expanduser()
        if not resolved_root.is_absolute():
            resolved_root = (Path.cwd() / resolved_root).resolve()
        else:
            resolved_root = resolved_root.resolve()

        if not resolved_root.exists() or not resolved_root.is_dir():
            continue
        try:
            service_file_names = {"service.yml", "service.win.yml", "service.linux.yml", "service.mac.yml"}
            for svc_file in resolved_root.rglob("*"):
                if svc_file.name in service_file_names and svc_file.is_file():
                    found.add(svc_file.parent.resolve())
        except _ENTRY_PATH_ERRORS as exc:
            logger.debug("recursive service discovery failed root=%s; falling back to direct children", resolved_root, exc_info=exc)
            for child in sorted(resolved_root.iterdir()):
                if not child.is_dir():
                    continue
                if any(
                    (child / name).is_file()
                    for name in ("service.yml", "service.win.yml", "service.linux.yml", "service.mac.yml")
                ):
                    found.add(child.resolve())
    return sorted(found)


def _absolutize_entry_paths(entry: F8ServiceEntry, *, service_dir: Path) -> F8ServiceEntry:
    launch = entry.launch
    workdir_raw = str(launch.workdir or "./")
    workdir_path = Path(workdir_raw).expanduser()
    if not workdir_path.is_absolute():
        workdir_path = (service_dir / workdir_path).resolve()
    else:
        workdir_path = workdir_path.resolve()

    command = launch.command
    command_raw = str(command or "").strip()
    try:
        command_path = Path(command_raw).expanduser()
        looks_like_path = bool(command_raw) and (
            "/" in command_raw or "\\" in command_raw or command_raw.startswith(".") or bool(command_path.suffix)
        )
        if looks_like_path and not command_path.is_absolute():
            command = str((workdir_path / command_path).resolve())
    except _ENTRY_PATH_ERRORS as exc:
        logger.debug("service entry command path absolutization failed command=%s", command_raw, exc_info=exc)

    absolute_launch = F8ServiceLaunchSpec(
        command=command,
        args=launch.args,
        env=launch.env,
        workdir=str(workdir_path),
    )
    return F8ServiceEntry(
        launch=absolute_launch,
        schemaVersion=entry.schemaVersion,
        serviceClass=entry.serviceClass,
        label=entry.label,
        version=entry.version,
        describeArgs=entry.describeArgs,
        timeoutMs=entry.timeoutMs,
    )


def load_service_entry(service_dir: Path) -> F8ServiceEntry:
    service_dir = Path(service_dir).resolve()

    def _try_load_candidate(candidate: Path) -> dict[str, Any] | None:
        if not candidate.is_file():
            return None
        obj = _read_yaml(candidate)
        if not isinstance(obj, dict):
            raise ValueError(f"{candidate} must be a YAML mapping")

        if candidate.name != "service.yml":
            try:
                launch = obj.get("launch") if isinstance(obj.get("launch"), dict) else {}
                command_raw = str((launch or {}).get("command") or "").strip()
                workdir_raw = str((launch or {}).get("workdir") or "./").strip() or "./"
                command_path = Path(command_raw)
                if command_raw and not command_path.is_absolute() and (
                    "/" in command_raw or "\\" in command_raw or command_path.suffix
                ):
                    workdir_path = Path(workdir_raw).expanduser()
                    if not workdir_path.is_absolute():
                        workdir_path = (service_dir / workdir_path).resolve()
                    else:
                        workdir_path = workdir_path.resolve()
                    resolved_command = (workdir_path / command_path).resolve()
                    if not resolved_command.is_file():
                        return None
            except _ENTRY_PATH_ERRORS as exc:
                logger.debug("platform service entry command probe failed path=%s", candidate, exc_info=exc)
        return obj

    data: Any | None = None
    candidates = _service_yml_candidates(service_dir)
    for candidate in candidates:
        try:
            loaded = _try_load_candidate(candidate)
        except _ENTRY_CANDIDATE_LOAD_ERRORS as exc:
            raise ValueError(str(exc)) from exc
        if loaded is not None:
            data = loaded
            break

    if data is None:
        tried = ", ".join(str(path.name) for path in candidates)
        raise ValueError(f"{service_dir} is missing a service entry YAML (tried: {tried})")

    if "launch" not in data and "command" in data:
        launch = validate_as(
            F8ServiceEntry,
            {
                "command": data.get("command"),
                "args": data.get("args") or [],
                "env": data.get("env") or {},
                "workdir": data.get("workdir") or "./",
            },
        )
        data = dict(data)
        data["launch"] = dump_json(launch)

    try:
        entry = validate_as(F8ServiceEntry, data)
    except _ENTRY_VALIDATION_ERRORS as exc:
        raise ValueError(f"Invalid service entry in {service_dir}: {exc}") from exc

    return _absolutize_entry_paths(entry, service_dir=service_dir)


__all__ = ["default_discovery_roots", "find_service_dirs", "load_service_entry"]
