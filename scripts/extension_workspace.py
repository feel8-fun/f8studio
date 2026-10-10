"""Generate development registrations from extension-owned declarations."""
from __future__ import annotations

import argparse
import ast
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
import tomllib

import msgspec
import yaml

from f8pysdk.codec import copy_model, validate_as
from f8pysdk.extension_packaging import validate_package
from f8pysdk.extension_spec import ExtensionCatalog
from f8pysdk.release_spec import RuntimeCatalog, RuntimeDefinition
from f8pysdk.service_runtime_tools.inventory.index import IndexedService, ServiceIndex, read_service_index
from f8pysdk.service_paths import ServicePaths
from f8pysdk.specs import F8ServiceEntry

REPO_ROOT = Path(__file__).resolve().parents[1]


def workspace_root(root: Path = REPO_ROOT) -> Path:
    return root / 'build/workspace'


def workspace_index(root: Path = REPO_ROOT) -> Path:
    return workspace_root(root) / 'config/service-index.json'


@dataclass(frozen=True)
class SourcePackage:
    package: str
    extension_ids: tuple[str, ...]
    python_modules: tuple[str, ...]


def source_packages(root: Path = REPO_ROOT) -> tuple[SourcePackage, ...]:
    workspace = tomllib.loads((root / 'config/extension-workspace.toml').read_text())
    packages = tuple(SourcePackage(item['package'], tuple(item['extension_ids']),
                                   tuple(item.get('python_modules', [item['package']])))
                     for item in workspace['extensions'])
    names = [item.package for item in packages]
    ids = [extension for item in packages for extension in item.extension_ids]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)) or set(names) & set(workspace['core']):
        raise ValueError('Core and extension packages must have unique, disjoint ownership')
    for name in names:
        if Path(name).name != name or not name.startswith('f8'):
            raise ValueError(f'Unsafe source package name: {name}')
    return packages


def _workspace_reference(root: Path, paths: ServicePaths, reference: str, *, relative_to: Path) -> str:
    path = paths.package_path(reference, relative_to=relative_to)
    return '${F8_PACKAGE_ROOT}/' + path.relative_to(root.resolve()).as_posix()


def compose_catalog(root: Path = REPO_ROOT) -> ExtensionCatalog:
    workspace = tomllib.loads((root / 'config/extension-workspace.toml').read_text())
    manifests = []
    for package in source_packages(root):
        source = root / 'extensions' / package.package
        catalog = validate_package(source)
        if set(package.extension_ids) != {item.extension_id for item in catalog.extensions}:
            raise ValueError(f'Package extension IDs disagree with workspace: {package.package}')
        paths = ServicePaths.for_index(source / 'config/service-index.json')
        for manifest in catalog.extensions:
            tools = tuple(copy_model(tool, update={
                'workdir': _workspace_reference(root, paths, tool.workdir, relative_to=source),
                'command': _workspace_reference(root, paths, tool.command,
                                                relative_to=paths.package_path(tool.workdir, relative_to=source))
                if manifest.runtime.kind == 'native' else tool.command,
                'args': tuple(_workspace_reference(root, paths, arg, relative_to=source)
                              if arg.startswith('${F8_PACKAGE_ROOT}') else arg for arg in tool.args),
            }) for tool in manifest.tools)
            skills = tuple(copy_model(skill, update={'path': _workspace_reference(root, paths, skill.path, relative_to=source)})
                           for skill in manifest.skills)
            resources = tuple(copy_model(asset, update={'path': _workspace_reference(root, paths, asset.path, relative_to=source)})
                              for asset in manifest.resources)
            manifests.append(copy_model(manifest, update={'tools': tools, 'skills': skills, 'resources': resources}))
    classes = [name for item in manifests for name in item.service_classes]
    ids = {item.extension_id for item in manifests}
    preinstalled = tuple(workspace.get('preinstalled', []))
    if len(classes) != len(set(classes)) or set(preinstalled) - ids:
        raise ValueError('Duplicate service ownership or unknown preinstalled extension')
    return ExtensionCatalog(schema_version='f8extensionCatalog/1', extensions=tuple(manifests), preinstalled=preinstalled)


def _python_task(source: Path, entry: F8ServiceEntry) -> str:
    definition = tomllib.loads((source / 'pixi.toml').read_text())
    tasks = dict(definition.get('tasks', {}))
    for feature in definition.get('feature', {}).values():
        tasks.update(feature.get('tasks', {}))
    expected = ['python', *(entry.launch.args or [])]
    matches = []
    for name, task in tasks.items():
        command = task.get('cmd') if isinstance(task, dict) else task
        if isinstance(command, str) and shlex.split(command) == expected:
            matches.append(name)
    if len(matches) != 1:
        raise ValueError(f'{entry.serviceClass}: expected one Pixi task for {expected}, found {matches}')
    return matches[0]


def generated_config(root: Path = REPO_ROOT) -> dict[Path, bytes]:
    root = root.resolve()
    output = workspace_root(root)
    config = output / 'config'
    result: dict[Path, bytes] = {}
    catalog = compose_catalog(root)
    owners = {name: manifest for manifest in catalog.extensions for name in manifest.service_classes}
    registrations: list[IndexedService] = []
    runtimes: list[RuntimeDefinition] = []
    sources: dict[str, str] = {}
    workspace = tomllib.loads((root / 'config/extension-workspace.toml').read_text())
    for runtime in workspace.get('runtimes', []):
        runtimes.append(RuntimeDefinition(runtime_id=runtime['environment'],
                                         manifest='${F8_PACKAGE_ROOT}/' + runtime['package'] + '/pixi.toml'))
    for package in source_packages(root):
        source = root / 'extensions' / package.package
        declared = validate_package(source)
        for manifest in declared.extensions:
            sources[manifest.extension_id] = '${F8_PACKAGE_ROOT}/extensions/' + package.package
            for directory in manifest.model_directories:
                for metadata in sorted((source / 'resources/models' / directory).glob('*.yaml')):
                    target = output / 'resources/models' / directory / metadata.name
                    data = metadata.read_bytes()
                    if target in result and result[target] != data:
                        raise ValueError(f'Conflicting model metadata: {target.name}')
                    result[target] = data
        index_path = source / 'config/service-index.json'
        if not index_path.is_file():
            continue
        index = read_service_index(index_path)
        paths = ServicePaths.for_index(index_path)
        for item in index.services:
            manifest = owners[item.serviceClass]
            manifests: dict[str, str] = {}
            for platform, reference in item.manifests.items():
                original = paths.package_path(reference, relative_to=index_path.parent)
                entry = validate_as(F8ServiceEntry, yaml.safe_load(original.read_text()))
                if entry.launch.command == 'python':
                    if manifest.runtime.environment is None:
                        raise ValueError(f'Missing workspace environment for {item.serviceClass}')
                    launch = copy_model(entry.launch, update={
                        'command': 'pixi', 'args': ['run', '-e', manifest.runtime.environment, _python_task(source, entry)],
                        'workdir': '${F8_PACKAGE_ROOT}/extensions/' + package.package,
                    })
                    entry = copy_model(entry, update={'launch': launch})
                target = config / 'services' / item.serviceClass / original.name
                result[target] = yaml.safe_dump(msgspec.to_builtins(entry), sort_keys=False).encode()
                manifests[platform] = '${F8_PACKAGE_ROOT}/' + target.relative_to(root).as_posix()
            bundle_roots = {platform: reference.replace('${F8_PACKAGE_ROOT}/runtime/', '${F8_PACKAGE_ROOT}/build/workspace/runtime/')
                            for platform, reference in item.bundleRoots.items()}
            describe = f'${{F8_PACKAGE_ROOT}}/build/workspace/runtime/bundles/{item.serviceClass}/{manifest.version}/describe.json'
            registrations.append(IndexedService(serviceClass=item.serviceClass, manifests=manifests,
                                                describe=describe, bundleRoots=bundle_roots))
    names = [runtime.runtime_id for runtime in runtimes]
    if len(names) != len(set(names)):
        raise ValueError('Workspace runtime aliases must be unique; extension environment names remain independent')
    result[config / 'extensions.json'] = _json(catalog)
    result[config / 'extension-sources.json'] = _json(sources)
    result[config / 'service-index.json'] = _json(ServiceIndex(
        schemaVersion='f8serviceIndex/1', services=tuple(registrations), modelRoot='${F8_MODEL_ROOT}', packageRoot=str(root),
    ))
    result[config / 'runtime-environments.json'] = _json(RuntimeCatalog(schema_version='f8runtimeCatalog/1', runtimes=tuple(runtimes)))
    return result


def _json(value: object) -> bytes:
    return msgspec.json.format(msgspec.json.encode(value), indent=2) + b'\n'


def sync_workspace(root: Path = REPO_ROOT) -> Path:
    contents = generated_config(root)
    output = workspace_root(root)
    for directory in (output / 'config/services', output / 'resources'):
        if directory.exists():
            shutil.rmtree(directory)
    for path, content in contents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return workspace_index(root)


def check_workspace(root: Path = REPO_ROOT) -> None:
    compose_catalog(root)
    workspace = tomllib.loads((root / 'config/extension-workspace.toml').read_text())
    extension_modules = {module for item in source_packages(root) for module in item.python_modules}
    roots = [root / 'packages' / package for package in workspace['core']]
    roots.extend(root / 'extensions' / package for package in workspace.get('applications', []))
    roots.append(root / 'platform')
    for source_root in roots:
        for source in source_root.rglob('*.py'):
            if any(part in {'node_modules', '.pixi', '.git', '__pycache__', 'tests', '.sdk', '.platform', '.media-dependency', '.engine', '.diagnostics', 'build'} for part in source.parts):
                continue
            tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                if any(name.split('.')[0] in extension_modules for name in names):
                    raise ValueError(f'Core must not import an extension implementation: {source}:{node.lineno}')
    for path, content in generated_config(root).items():
        if not path.is_file() or path.read_bytes() != content:
            raise ValueError(f'Development registration is stale: {path}; run extension_workspace.py sync')



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'sync'])
    args = parser.parse_args()
    if args.command == 'check':
        check_workspace()
    elif args.command == 'sync':
        print(sync_workspace())


if __name__ == '__main__':
    main()
