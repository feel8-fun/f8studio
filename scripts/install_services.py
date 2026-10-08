"""Prepare descriptions and migrate resources from an explicit service index.

Normal Studio startup never invokes this tool or downloads model files.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import sys
import subprocess
import tempfile
import tomllib

import msgspec

from f8pysdk._specs.builtin_fields import normalize_describe_payload_dict
from f8pysdk.codec import copy_model, validate_as
from f8pysdk.monitoring import validate_describe_monitor_contract
from f8pysdk.specs import F8ServiceDescribe, F8ServiceEntry
from f8pysdk.resource_paths import service_config_root
from f8pysdk.service_runtime_tools.inventory.describe import _extract_last_json_obj
from f8pysdk.service_runtime_tools.inventory.index import default_service_index, index_paths, indexed_entry, read_service_index


REPO_ROOT = Path(__file__).resolve().parents[1]


def file_digest(path: Path) -> bytes:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").digest()


def copy_verified(source: Path, target: Path) -> None:
    """Preserve existing resources, refusing collisions instead of overwriting."""
    digest = file_digest(source)
    if target.exists():
        if file_digest(target) != digest:
            raise ValueError(f"Resource conflict: {source} -> {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copy2(source, temporary_path)
        if file_digest(temporary_path) != digest:
            raise ValueError(f"Resource copy checksum mismatch: {source}")
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)


def migrate_resources(legacy_root: Path, model_root: Path) -> int:
    count = 0
    for relative, destination in (
        ("f8/dl/weights", "onnx"),
        ("f8/mp/pose/models", "mediapipe"),
        ("f8/cvkit/tracking/models", "tracking"),
    ):
        source_dir = legacy_root / relative
        if not source_dir.is_dir():
            continue
        for source in sorted(source_dir.rglob("*")):
            if not source.is_file() or source.name.startswith("."):
                continue
            copy_verified(source, model_root / destination / source.relative_to(source_dir))
            count += 1
    return count


def migrate_runtime_layout(legacy_root: Path, installation_root: Path) -> int:
    """Copy known runtime bundles and preserve legacy UI configuration."""
    count = 0
    for source_name, bundle in (
        ("audiocap", "f8.audiocap"), ("cppengine", "f8.cppengine"),
        ("cvkit", "f8.cvkit"), ("implayer", "f8.implayer"),
        ("screencap", "f8.screencap"),
    ):
        for platform in ("linux", "win", "mac", "macos"):
            source_dir = legacy_root / "f8" / source_name / platform
            if not source_dir.is_dir():
                continue
            for source in sorted(source_dir.rglob("*")):
                if source.is_file():
                    target = installation_root / "runtime" / "bundles" / bundle / "0.0.1" / platform / source.relative_to(source_dir)
                    copy_verified(source, target)
                    count += 1
    config = legacy_root / "f8" / "implayer" / "imgui.ini"
    if config.is_file():
        copy_verified(config, service_config_root() / "f8.implayer" / "imgui.ini")
        count += 1
    return count


def pixi_environment(entry: F8ServiceEntry) -> str | None:
    """Require an explicit, declared task/environment before executing any service."""
    if entry.launch.command not in {"pixi", "pixi.exe"}:
        return None
    args = entry.launch.args or []
    if len(args) != 4 or args[0] != "run" or args[1] not in {"-e", "--environment"}:
        raise ValueError(f"{entry.serviceClass}: expected pixi run -e ENV TASK, got {args!r}")
    environment, task = args[2:]
    manifest_path = Path(entry.launch.workdir or ".") / "pixi.toml"
    with manifest_path.open("rb") as source:
        manifest = tomllib.load(source)
    environments = manifest.get("environments", {})
    if environment not in environments:
        raise ValueError(f"{entry.serviceClass}: unknown Pixi environment {environment!r}")
    definition = environments[environment]
    features = definition if isinstance(definition, list) else definition.get("features", [])
    tasks = set(manifest.get("tasks", {}))
    for feature in features:
        tasks.update(manifest.get("feature", {}).get(feature, {}).get("tasks", {}))
    if task not in tasks:
        raise ValueError(f"{entry.serviceClass}: task {task!r} is not available in {environment!r}")
    return environment


def description_entry(entry: F8ServiceEntry, *, build_check: bool = False) -> F8ServiceEntry:
    environment = pixi_environment(entry)
    if environment is None or not build_check:
        return entry
    workspace = Path(entry.launch.workdir or '.').resolve()
    manifest = tomllib.loads((workspace / 'pixi.toml').read_text(encoding='utf-8'))
    tasks = dict(manifest.get('tasks', {}))
    definition = manifest['environments'][environment]
    features = definition if isinstance(definition, list) else definition.get('features', [])
    for feature in features:
        tasks.update(manifest.get('feature', {}).get(feature, {}).get('tasks', {}))
    task = tasks[(entry.launch.args or [])[-1]]
    command = task.get('cmd') if isinstance(task, dict) else task
    if not isinstance(command, str):
        raise ValueError(f'{entry.serviceClass}: description task requires an explicit command')
    args = shlex.split(command)
    if len(args) < 3 or args[:2] != ['python', '-m']:
        raise ValueError(f'{entry.serviceClass}: build-check descriptions require a python -m entrypoint')
    env = dict(entry.launch.env or {})
    previous_path = env.get('PYTHONPATH', os.environ.get('PYTHONPATH', ''))
    env['PYTHONPATH'] = str(workspace) + (os.pathsep + previous_path if previous_path else '')
    # Only the explicit CI/source-description path uses this interpreter.
    # Normal installation and execution retain the extension's own workspace.
    return copy_model(entry, update={'launch': copy_model(entry.launch, update={
        'command': sys.executable, 'args': args[1:], 'workdir': str(REPO_ROOT), 'env': env,
    })})


def describe_service(entry: F8ServiceEntry) -> object:
    args = list(entry.launch.args or [])
    if entry.launch.command in {"pixi", "pixi.exe"}:
        # Dependency installation is a separate phase, outside the describe timeout.
        args[1:1] = ["--frozen", "--no-install"]
    command = [entry.launch.command, *args, *(entry.describeArgs or ["--describe"])]
    env = os.environ.copy()
    if entry.launch.command in {'pixi', 'pixi.exe'}:
        for key in ('PYTHONPATH', 'PYTHONHOME', 'PIXI_PROJECT_MANIFEST', 'PIXI_ENVIRONMENT_NAME'):
            env.pop(key, None)
    env.update(entry.launch.env or {})
    print(f"Describe {entry.serviceClass}: {command!r}", flush=True)
    try:
        proc = subprocess.run(
            command, cwd=entry.launch.workdir, env=env, capture_output=True, text=True,
            timeout=max(30.0, float(entry.timeoutMs or 4000) / 1000), check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Describe failed for {entry.serviceClass}; cwd={entry.launch.workdir}; "
            f"command={command!r}; {exc}\nstdout: {exc.stdout}\nstderr: {exc.stderr}"
        ) from exc
    return _extract_last_json_obj(proc.stdout)


def install(index_path: Path, *, refresh: bool, service_classes: set[str],
            python_only: bool = False, no_install: bool = False, build_check: bool = False, native_only: bool = False,
            validate_only: bool = False) -> int:
    index_path = index_path.resolve()
    index = read_service_index(index_path)
    known = {item.serviceClass for item in index.services}
    if service_classes - known:
        raise ValueError(f"Unregistered services: {sorted(service_classes - known)}")
    selected: list[tuple[F8ServiceEntry, Path]] = []
    environments: dict[Path, set[str]] = {}
    for item in index.services:
        if service_classes and item.serviceClass not in service_classes:
            continue
        entry = indexed_entry(index_path, index, item)
        if entry is None:
            continue
        python_service = (pixi_environment(entry) is not None
                          or entry.launch.command in {'python', 'python.exe', sys.executable})
        entry = description_entry(entry, build_check=build_check)
        target = index_paths(index_path, index, item).package_path(item.describe, relative_to=index_path.parent)
        environment = pixi_environment(entry)
        if native_only and python_service:
            continue
        if python_only and not python_service:
            continue
        selected.append((entry, target))
        if environment is not None:
            root = Path(entry.launch.workdir or ".").resolve()
            environments.setdefault(root, set()).add(environment)
    # Validate every selected binding before installing or running anything.
    if not no_install:
        for root, names in sorted(environments.items()):
            command = ["pixi", "install", "--locked", "--manifest-path", str(root / "pixi.toml")]
            for name in sorted(names):
                command.extend(["-e", name])
            print(f"Prepare service environments: {command!r}", flush=True)
            subprocess.run(command, cwd=root, check=True)
    outputs: list[tuple[Path, bytes]] = []
    for entry, target in selected:
        if target.is_file() and not refresh and pixi_environment(entry) is None:
            raw = msgspec.json.decode(target.read_bytes())
        else:
            raw = describe_service(entry)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid describe object: {entry.serviceClass}")
        payload = normalize_describe_payload_dict(raw)
        validate_describe_monitor_contract(payload)
        describe = validate_as(F8ServiceDescribe, payload)
        if describe.service.serviceClass != entry.serviceClass:
            raise ValueError(f"Description class mismatch: {entry.serviceClass}")
        outputs.append((target, msgspec.json.format(msgspec.json.encode(describe), indent=2) + b"\n"))
    # Validate the entire selection before publishing any new descriptions.
    if validate_only:
        return len(outputs)
    for target, content in outputs:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        try:
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)
    return len(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--refresh", action="store_true", help="Regenerate descriptions by running registered services")
    parser.add_argument("--native-only", action="store_true", help="Refresh native descriptions, reusing checked Python descriptions")
    parser.add_argument("--python-only", action="store_true", help="Check Pixi services before native compilation")
    parser.add_argument("--build-check", action="store_true", help="Use the consolidated CI environment for descriptions")
    parser.add_argument("--no-install", action="store_true", help="Use environments already prepared by CI")
    parser.add_argument("--service-class", action="append", default=[])
    parser.add_argument("--migrate-resources", type=Path, metavar="OLD_SERVICES_DIR",
                        help="Copy and checksum legacy models; keep original files")
    parser.add_argument("--migrate-layout", type=Path, metavar="OLD_SERVICES_DIR",
                        help="Copy legacy executable bundles and preserve user configuration")
    args = parser.parse_args()
    path = (args.index or default_service_index()).resolve()
    if args.migrate_layout is not None:
        count = migrate_runtime_layout(args.migrate_layout, path.parent.parent)
        print(f"Verified {count} runtime/config files; originals preserved")
    if args.migrate_resources is not None:
        index = read_service_index(path)
        count = migrate_resources(args.migrate_resources, index_paths(path, index).model_root)
        print(f"Verified {count} resource files; originals preserved")
    count = install(path, refresh=args.refresh, service_classes=set(args.service_class),
                    python_only=args.python_only, no_install=args.no_install, build_check=args.build_check, native_only=args.native_only)
    print(f"Installed {count} service descriptions from {path}")


if __name__ == "__main__":
    main()
