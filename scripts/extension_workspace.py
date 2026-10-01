"""Validate extension ownership and synchronize the superbuild catalog."""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import tomllib

import msgspec

from f8pysdk.extension_packaging import validate_package
from f8pysdk.extension_spec import ExtensionCatalog

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourcePackage:
    package: str
    extension_ids: tuple[str, ...]


def source_packages(root: Path = REPO_ROOT) -> tuple[SourcePackage, ...]:
    workspace = tomllib.loads((root / 'config/extension-workspace.toml').read_text())
    packages = tuple(SourcePackage(item['package'], tuple(item['extension_ids'])) for item in workspace['extensions'])
    names = [item.package for item in packages]
    ids = [extension for item in packages for extension in item.extension_ids]
    if len(names) != len(set(names)) or len(ids) != len(set(ids)) or set(names) & set(workspace['core']):
        raise ValueError('Core and extension packages must have unique, disjoint ownership')
    for name in names:
        if Path(name).name != name or not name.startswith('f8'):
            raise ValueError(f'Unsafe source package name: {name}')
    return packages


def compose_catalog(root: Path = REPO_ROOT) -> ExtensionCatalog:
    previous = msgspec.json.decode((root / 'config/extensions.json').read_bytes(), type=ExtensionCatalog)
    manifests = []
    for package in source_packages(root):
        source = root / 'extensions' / package.package
        catalog = validate_package(source)
        if set(package.extension_ids) != {item.extension_id for item in catalog.extensions}:
            raise ValueError(f'Package extension IDs disagree with workspace: {package.package}')
        manifests.extend(catalog.extensions)
    order = {item.extension_id: position for position, item in enumerate(previous.extensions)}
    manifests.sort(key=lambda item: (order.get(item.extension_id, len(order)), item.extension_id))
    classes = [name for item in manifests for name in item.service_classes]
    ids = {item.extension_id for item in manifests}
    if len(classes) != len(set(classes)) or set(previous.preinstalled) - ids:
        raise ValueError('Duplicate service ownership or unknown preinstalled extension')
    return ExtensionCatalog(schema_version='f8extensionCatalog/1', extensions=tuple(manifests), preinstalled=previous.preinstalled)


def check_workspace(root: Path = REPO_ROOT) -> None:
    expected = compose_catalog(root)
    actual = msgspec.json.decode((root / 'config/extensions.json').read_bytes(), type=ExtensionCatalog)
    if expected != actual:
        raise ValueError('Extension catalog is stale; run extension_workspace.py sync')
    workspace = tomllib.loads((root / 'config/extension-workspace.toml').read_text())
    extension_modules = {item.package for item in source_packages(root)}
    for package in workspace['core']:
        for source in (root / 'packages' / package / package).rglob('*.py'):
            tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                if any(name.split('.')[0] in extension_modules for name in names):
                    raise ValueError(f'Core must not import an extension implementation: {source}:{node.lineno}')



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'sync'])
    args = parser.parse_args()
    if args.command == 'check':
        check_workspace()
    elif args.command == 'sync':
        (REPO_ROOT / 'config/extensions.json').write_bytes(msgspec.json.format(msgspec.json.encode(compose_catalog()), indent=2) + b'\n')


if __name__ == '__main__':
    main()
