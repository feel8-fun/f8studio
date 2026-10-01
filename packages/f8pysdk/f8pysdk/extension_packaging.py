"""Build an extension ZIP from an independent service source tree and wheel/runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import msgspec
import yaml

from .codec import copy_model, validate_as
from .extension_spec import ExtensionCatalog
from .monitoring import validate_describe_monitor_contract
from .service_runtime_tools.inventory.index import indexed_entry, read_service_index
from .specs import F8ServiceDescribe, F8ServiceEntry


def validate_package(source: Path) -> ExtensionCatalog:
    catalog = msgspec.json.decode((source / 'extension.json').read_bytes(), type=ExtensionCatalog)
    index = read_service_index(source / 'config/service-index.json')
    owners = [service for extension in catalog.extensions for service in extension.service_classes]
    if not catalog.extensions or len(owners) != len(set(owners)) or set(owners) != {item.serviceClass for item in index.services}:
        raise ValueError('Extension ownership must match the service index exactly')
    ids = [extension.extension_id for extension in catalog.extensions]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate extension ID')
    for item in index.services:
        for relative in item.manifests.values():
            path = (source / 'config' / relative).resolve()
            if not path.is_relative_to(source.resolve()) or not path.is_file():
                raise ValueError(f'Missing or unsafe service metadata: {relative}')
        for relative in item.manifests.values():
            entry = validate_as(F8ServiceEntry, yaml.safe_load((source / 'config' / relative).read_text()))
            if entry.serviceClass != item.serviceClass:
                raise ValueError(f'Service manifest disagrees with index: {relative}')
            if entry.launch.command in {'pixi', 'pixi.exe'}:
                raise ValueError(f'Extension must declare its own module/executable, not a superbuild task: {relative}')
        describe_path = (source / 'config' / item.describe).resolve()
        if not describe_path.is_relative_to(source.resolve()):
            raise ValueError(f'Unsafe service description path: {item.describe}')
        if describe_path.is_file():
            payload = json.loads(describe_path.read_bytes())
            describe = validate_as(F8ServiceDescribe, payload)
            if describe.service.serviceClass != item.serviceClass:
                raise ValueError(f'Service description disagrees with index: {item.serviceClass}')
            validate_describe_monitor_contract(payload)
    return catalog


def _extract_wheel(wheel: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            path = destination / info.filename
            if not path.resolve().is_relative_to(destination.resolve()) or '\\' in info.filename:
                raise ValueError(f'Unsafe wheel path: {info.filename}')
            if '.data' in Path(info.filename).parts[0]:
                raise ValueError(f'Extension wheel must contain importable modules only: {info.filename}')
        archive.extractall(destination)


def build_extension(source: Path, output: Path, *, wheel: Path | None = None, runtime_root: Path | None = None) -> Path:
    source = source.resolve()
    catalog = validate_package(source)
    kinds = {extension.runtime.kind for extension in catalog.extensions}
    if kinds <= {'workspace', 'shared'}:
        python_package = True
    elif kinds == {'native'}:
        python_package = False
    else:
        raise ValueError('This builder accepts native services or Python services sharing official presets')
    if python_package != (wheel is not None):
        raise ValueError('Python extensions require --wheel; native extensions must not supply one')
    if python_package and any(extension.runtime.environment is None for extension in catalog.extensions):
        raise ValueError('Python extensions must declare an official preset environment')
    with tempfile.TemporaryDirectory(prefix='f8-extension-') as temporary:
        stage = Path(temporary)
        shutil.copytree(source / 'config', stage / 'config')
        if (source / 'resources').is_dir():
            shutil.copytree(source / 'resources', stage / 'resources')
        published = copy_model(catalog, update={
            'preinstalled': (),
            'extensions': tuple(copy_model(extension, update={
                'runtime': copy_model(extension.runtime, update={'kind': 'shared'})
            }) if extension.runtime.kind == 'workspace' else extension for extension in catalog.extensions),
        })
        (stage / 'config/extensions.json').write_bytes(msgspec.json.encode(published))
        if wheel is not None:
            _extract_wheel(wheel, stage / 'python')
        else:
            if runtime_root is None:
                raise ValueError('Native extensions require --runtime-root with deployed runtime dependencies')
            index_path = stage / 'config/service-index.json'
            index = read_service_index(index_path)
            if any(sys.platform not in item.manifests and 'any' not in item.manifests for item in index.services):
                raise ValueError(f'Native service package does not support {sys.platform}')
            index = copy_model(index, update={'services': tuple(copy_model(item, update={
                'manifests': {platform: relative for platform, relative in item.manifests.items()
                              if platform in {sys.platform, 'any'}},
            }) for item in index.services)})
            index_path.write_bytes(msgspec.json.encode(index))
            copied: set[Path] = set()
            for item in index.services:
                entry = indexed_entry(stage / 'config/service-index.json', index, item)
                if entry is None:
                    continue
                if not isinstance(entry.launch.workdir, str):
                    raise ValueError(f'Missing native service workdir: {item.serviceClass}')
                directory = Path(entry.launch.workdir).resolve()
                if not directory.is_relative_to(stage / 'runtime/bundles'):
                    raise ValueError(f'Native service workdir must be inside runtime/bundles: {item.serviceClass}')
                relative = directory.relative_to(stage / 'runtime/bundles')
                original = runtime_root.resolve() / relative
                if not original.resolve().is_relative_to(runtime_root.resolve()):
                    raise ValueError(f'Unsafe native runtime path: {original}')
                if directory not in copied:
                    shutil.copytree(original, directory)
                    copied.add(directory)
        _refresh_describes(stage, python_package=python_package)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_archive = output.with_suffix('.zip.tmp')
        with zipfile.ZipFile(temporary_archive, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:
                    archive.write(path, path.relative_to(stage).as_posix())
        temporary_archive.replace(output)
    with output.open('rb') as archive_file:
        digest = hashlib.file_digest(archive_file, 'sha256').hexdigest()
    output.with_suffix('.zip.sha256').write_text(f'{digest}  {output.name}\n')
    return output


def _refresh_describes(stage: Path, *, python_package: bool) -> None:
    index_path = stage / 'config/service-index.json'
    index = read_service_index(index_path)
    for item in index.services:
        entry = indexed_entry(index_path, index, item)
        if entry is None:
            continue
        if not isinstance(entry.launch.workdir, str):
            raise ValueError(f'Missing service workdir: {item.serviceClass}')
        if python_package:
            args = entry.launch.args or []
            if entry.launch.command != 'python' or len(args) != 2 or args[0] != '-m':
                raise ValueError(f'Python service must launch python -m module: {item.serviceClass}')
            command = [sys.executable, '-I', '-c',
                       'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); '
                       'runpy.run_module(sys.argv.pop(1), run_name="__main__")',
                       str(stage / 'python'), args[1], '--describe']
        else:
            command = [entry.launch.command, *(entry.launch.args or []), '--describe']
        process = subprocess.run(command, cwd=entry.launch.workdir, env={**os.environ, **(entry.launch.env or {})},
                                 capture_output=True, text=True, timeout=120)
        if process.returncode:
            raise RuntimeError(f'{item.serviceClass} --describe failed ({process.returncode}):\n{process.stderr}')
        payload = json.loads(process.stdout)
        describe = validate_as(F8ServiceDescribe, payload)
        if describe.service.serviceClass != item.serviceClass:
            raise ValueError(f'Built service class mismatch: {item.serviceClass}')
        validate_describe_monitor_contract(payload)
        describe_path = index_path.parent / item.describe
        describe_path.parent.mkdir(parents=True, exist_ok=True)
        describe_path.write_bytes(msgspec.json.encode(describe))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wheel', type=Path)
    parser.add_argument('--wheel-dir', type=Path)
    parser.add_argument('--runtime-root', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.wheel_dir is not None:
        wheels = tuple(args.wheel_dir.glob('*.whl'))
        if args.wheel is not None or len(wheels) != 1:
            parser.error('--wheel-dir must contain exactly one extension wheel')
        args.wheel = wheels[0]
    if args.check:
        validate_package(args.source)
    elif args.output is not None:
        print(build_extension(args.source, args.output, wheel=args.wheel, runtime_root=args.runtime_root))
    else:
        parser.error('--output or --check is required')


if __name__ == '__main__':
    main()
