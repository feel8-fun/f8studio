"""Validate extension metadata against an existing Python environment."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path

import msgspec
from packaging.requirements import Requirement
from packaging.tags import parse_tag
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from .errors import InvalidRequestError
from .extension_models import ExtensionManifest


class InstalledDistribution(msgspec.Struct, frozen=True):
    version: str
    requires: tuple[str, ...]
    extras: tuple[str, ...]


class RuntimeProbe(msgspec.Struct, frozen=True, rename='camel'):
    python_version: str
    markers: dict[str, str]
    distributions: dict[str, InstalledDistribution]
    modules: tuple[str, ...]
    wheel_tags: tuple[str, ...] = ()


def check_dependencies(requirements: tuple[str, ...], probe: RuntimeProbe) -> None:
    installed = {canonicalize_name(name): value for name, value in probe.distributions.items()}
    checked: set[tuple[str, frozenset[str]]] = set()

    def check(requirement: Requirement, parent_extras: frozenset[str]) -> None:
        if requirement.marker is not None and not any(
            requirement.marker.evaluate({**probe.markers, 'extra': extra})
            for extra in (parent_extras | {''})
        ):
            return
        if requirement.url is not None:
            raise InvalidRequestError(f'Shared runtimes do not accept URL dependencies: {requirement}')
        name = canonicalize_name(requirement.name)
        distribution = installed.get(name)
        if distribution is None or not requirement.specifier.contains(distribution.version, prereleases=True):
            found = distribution.version if distribution is not None else 'not installed'
            raise InvalidRequestError(f'Selected runtime cannot satisfy {requirement} (found: {found}); '
                                      'publish this extension with an independent Pixi runtime')
        extras = frozenset(canonicalize_name(extra) for extra in requirement.extras)
        missing_extras = extras - {canonicalize_name(extra) for extra in distribution.extras}
        if missing_extras:
            raise InvalidRequestError(f'{requirement.name} does not provide extras {sorted(missing_extras)}')
        identity = (name, extras)
        if identity in checked:
            return
        checked.add(identity)
        for dependency in distribution.requires:
            check(Requirement(dependency), extras)

    for value in requirements:
        check(Requirement(value), frozenset())


def validate_shared_package(manifest: ExtensionManifest, python_root: Path, probe: RuntimeProbe) -> None:
    python_constraints = [manifest.runtime.requires_python] if manifest.runtime.requires_python else []
    dependencies = list(manifest.runtime.dependencies)
    installed_names = {canonicalize_name(name) for name in probe.distributions}
    for distribution in metadata.distributions(path=[str(python_root)]):
        wheel_metadata = distribution.read_text('WHEEL')
        if wheel_metadata is not None:
            tags = {str(tag) for line in wheel_metadata.splitlines() if line.startswith('Tag: ')
                    for tag in parse_tag(line.removeprefix('Tag: '))}
            if not tags or not tags.intersection(probe.wheel_tags):
                raise InvalidRequestError('Extension wheel is incompatible with the selected interpreter/ABI/platform')
        name = distribution.metadata.get('Name')
        if name and canonicalize_name(name) in installed_names:
            raise InvalidRequestError(f'Extension contains {name}, which would replace a runtime package')
        requires_python = distribution.metadata.get('Requires-Python')
        if requires_python:
            python_constraints.append(requires_python)
        dependencies.extend(distribution.requires or [])
    for constraint in python_constraints:
        if not SpecifierSet(constraint).contains(probe.python_version, prereleases=True):
            raise InvalidRequestError(f'Selected Python {probe.python_version} does not satisfy {constraint}; '
                                      'use an independent Pixi runtime')
    base_modules = {module.casefold() for module in probe.modules}
    for path in python_root.iterdir():
        module = path.name.split('.')[0]
        if ((path.is_dir() and '.' not in path.name or path.suffix in {'.py', '.pyd', '.so'})
                and module.casefold() in base_modules):
            raise InvalidRequestError(f'Extension module {module} would shadow an runtime or standard Python module')
    check_dependencies(tuple(dependencies), probe)
