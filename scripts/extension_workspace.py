"""Coordinate extension sources and export independent repositories for migration."""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
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


def _source_files(root: Path, package: str) -> tuple[tuple[Path, Path], ...]:
    source = root / 'extensions' / package
    # A real submodule owns its own tracked files; a directory is tracked by the superbuild.
    if (source / '.git').exists():
        base = source
        prefix = Path('.')
    else:
        base = root
        prefix = Path('extensions') / package
    output = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z', '--', str(prefix)], cwd=base)
    paths = [Path(value.decode()) for value in output.split(b'\0') if value]
    return tuple((base / path, path.relative_to(prefix)) for path in paths)


def export_repository(package: SourcePackage, destination: Path, *, sdk_ref: str, root: Path = REPO_ROOT, initialize_git: bool = False) -> None:
    if destination.exists():
        raise FileExistsError(f'Export destination already exists: {destination}')
    source = root / 'extensions' / package.package
    validate_package(source)
    destination.mkdir(parents=True)
    for original, relative in _source_files(root, package.package):
        if original.is_symlink():
            raise ValueError(f'Source export must not follow symlinks: {original}')
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    python_package = (source / 'pyproject.toml').is_file()
    (destination / 'pixi.toml').write_text(_pixi_manifest(package.package, python_package=python_package))
    if python_package:
        shutil.copy2(root / 'ruff.toml', destination / 'ruff.toml')
    workflow = destination / '.github/workflows/quality.yml'
    workflow.parent.mkdir(parents=True)
    workflow.write_text(_workflow(sdk_ref, python_package=python_package, has_tests=(source / 'tests').is_dir()))
    with (destination / '.gitignore').open('a') as ignore:
        ignore.write('\n.sdk/\n')
    (destination / 'DEVELOPMENT.md').write_text(f'''# {package.package}

This repository owns its source, tests, `extension.json`, service manifests and model metadata.
The Studio superbuild can check it out unchanged at `extensions/{package.package}`.

The SDK source revision used by CI is `{sdk_ref}`. Checkout `feel8-fun/f8studio` at that
revision into `.sdk` before `pixi install`; only the public SDK is a runtime dependency.
Use the commands in `.github/workflows/quality.yml` for local build/test parity.

Python releases contain this package's wheel contents and reuse the official environment
named by `extension.json`. Native releases contain deployed executables and their runtime
libraries. `python -m f8pysdk.extension_packaging` verifies the real `--describe` entrypoints,
then produces an importable ZIP and its SHA-256. No editable source path is included.

The initial source export is a snapshot. Original commit history remains in the superbuild;
use `git subtree split --prefix=extensions/{package.package}` if full package history is needed.
''')
    validate_package(destination)
    # Resolve each repository's own lock against the public SDK, not the entire
    # superbuild dependency set. CI checks out the pinned SDK into the same path.
    sdk = destination / '.sdk/packages/f8pysdk'
    shutil.copytree(root / 'packages/f8pysdk', sdk,
                    ignore=shutil.ignore_patterns('.git', '.pixi', '__pycache__', '*.egg-info', 'build', 'dist'))
    subprocess.run(['pixi', 'lock', '--manifest-path', str(destination / 'pixi.toml')], cwd=destination, check=True)
    if initialize_git:
        subprocess.run(['git', 'init', '-q', '-b', 'main', str(destination)], check=True)
        subprocess.run(['git', 'add', '.'], cwd=destination, check=True)
        subprocess.run(['git', 'commit', '-q', '-m', f'Initialize independent {package.package} extension'], cwd=destination, check=True)


def _pixi_manifest(package: str, *, python_package: bool) -> str:
    own = f'{package} = {{path = ".", editable = true}}\n' if python_package else ''
    native = ''
    if not python_package:
        native = '''conan = ">=2.23,<3"
cmake = ">=3.30,<5"
ninja = ">=1.13,<2"
pkg-config = ">=0.29,<1"
libzenohc = ">=1.9.0,<1.10"
libzenohcxx = ">=1.9.0,<1.10"
gtest = ">=1.17,<2"
'''
        if package in {'f8audiocap', 'f8implayer'}:
            native += 'sdl3 = ">=3.2,<4"\n'
        if package == 'f8implayer':
            native += 'yt-dlp = ">=2025.8,<2027"\n'
        native += '\n[target.linux-64.dependencies]\ngxx_linux-64 = ">=11,<12"\n'
        if package == 'f8screencap':
            native += 'xorg-libx11 = ">=1.8,<2"\nxorg-libxrandr = ">=1.5,<2"\nxorg-libxext = ">=1.3,<2"\nxorg-xorgproto = ">=2025.1,<2027"\n'
        if package == 'f8implayer':
            native += 'mpv = ">=0.39,<1"\nlibgl-devel = ">=1.7,<2"\nlibegl-devel = ">=1.7,<2"\n'
    return f'''[workspace]
name = "{package}"
channels = ["conda-forge"]
platforms = ["win-64", {{name = "linux-glibc228", platform = "linux-64", glibc = "2.28"}}]

[dependencies]
python = ">=3.14.3,<3.15"
pip = ">=26,<27"
pytest = ">=8.4.2,<9"
{native}

[pypi-dependencies]
f8pysdk = {{path = ".sdk/packages/f8pysdk", editable = false}}
hatchling = ">=1.28,<2"
basedpyright = ">=1.31,<2"
ruff = ">=0.12,<0.13"
{own}
[tasks]
check = "python -m f8pysdk.extension_packaging --check"
'''


def _workflow(sdk_ref: str, *, python_package: bool, has_tests: bool) -> str:
    if not sdk_ref or any(char not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._/-' for char in sdk_ref):
        raise ValueError('SDK revision must be a Git commit, tag or branch')
    prefix = f'''name: extension
on: [push, pull_request, workflow_dispatch]
permissions:
  contents: read
concurrency:
  group: extension-${{{{ github.ref }}}}
  cancel-in-progress: true
jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-24.04, windows-2022]
    runs-on: ${{{{ matrix.os }}}}
    steps:
      - uses: actions/checkout@v5
      - uses: actions/checkout@v5
        with:
          repository: feel8-fun/f8studio
          ref: {sdk_ref}
          path: .sdk
      - uses: prefix-dev/setup-pixi@v0.9.3
        with:
          pixi-version: v0.81.0
          cache: true
      - run: pixi run check
'''
    if python_package:
        body = '      - run: pixi run python -m pytest tests -q\n' if has_tests else ''
        body += '''      - run: pixi run basedpyright -p pyrightconfig.json
      - run: pixi run ruff check .
      - run: pixi run python -m pip wheel --no-deps --no-build-isolation -w build/wheels .
      - run: pixi run python -m f8pysdk.extension_packaging --wheel-dir build/wheels --output build/extension.zip
'''
    else:
        body = '''      - uses: actions/cache/restore@v5
        id: conan-cache
        with:
          path: build/conan-cache
          key: conan-${{ runner.os }}-${{ hashFiles('conan.lock', '.sdk/conan.lock', 'pixi.lock') }}
      - name: Prepare locked SDK and extension dependencies
        env:
          CONAN_HOME: ${{ github.workspace }}/build/conan-cache
        run: |
          pixi run conan profile detect --force
          pixi run conan install .sdk -of build/sdk-deps -s build_type=Release -s compiler.cppstd=17 -o with_extensions=False --build=missing --lockfile .sdk/conan.lock --lockfile-partial
          pixi run conan install . -of build/deps -s build_type=Release -s compiler.cppstd=17 --build=missing --lockfile conan.lock
      - uses: actions/cache/save@v5
        if: steps.conan-cache.outputs.cache-hit != 'true'
        with:
          path: build/conan-cache
          key: ${{ steps.conan-cache.outputs.cache-primary-key }}
      - name: Build and install the communication SDK
        shell: bash
        run: |
          f8_sdk_toolchain=$(pixi run python -c "from pathlib import Path; candidates = list(Path('build/sdk-deps').rglob('conan_toolchain.cmake')); assert len(candidates) == 1, candidates; print(candidates[0].resolve().as_posix())")
          pixi run cmake -S .sdk -B build/sdk "-DCMAKE_TOOLCHAIN_FILE=$f8_sdk_toolchain" -DF8_EXTENSION_PACKAGES= -DF8_BUILD_SDK_DEMO=OFF "-DF8_PIXI_CPP_ENV_DIR=${{ github.workspace }}/.pixi/envs/default" "-DCMAKE_INSTALL_PREFIX=${{ github.workspace }}/build/sdk-install" -DCMAKE_BUILD_TYPE=Release
          pixi run cmake --build build/sdk --config Release --parallel 2
          pixi run cmake --install build/sdk --config Release
      - name: Build only this extension
        run: |
          pixi run cmake -S . -B build/native -DCMAKE_TOOLCHAIN_FILE=${{ github.workspace }}/build/deps/conan_toolchain.cmake "-DCMAKE_PREFIX_PATH=${{ github.workspace }}/build/sdk-install;${{ github.workspace }}/.pixi/envs/default;${{ github.workspace }}/.pixi/envs/default/Library" -DCMAKE_BUILD_TYPE=Release
          pixi run cmake --build build/native --config Release --parallel 2
          pixi run ctest --test-dir build/native -C Release --output-on-failure
      - run: pixi run python -m f8pysdk.extension_packaging --runtime-root build/native/runtime/bundles --output build/extension.zip
'''
    return prefix + body + '''      - uses: actions/upload-artifact@v4
        with:
          name: extension-${{ runner.os }}
          path: |
            build/extension.zip
            build/extension.zip.sha256
          if-no-files-found: error
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'sync', 'export'])
    parser.add_argument('--package', action='append', default=[])
    parser.add_argument('--output-dir', type=Path, default=REPO_ROOT / 'build/extension-repositories')
    parser.add_argument('--sdk-ref', help='SDK commit/tag to use in independent repository CI; required for export')
    parser.add_argument('--git', action='store_true', help='Initialize and commit each exported local repository')
    args = parser.parse_args()
    if args.command == 'check':
        check_workspace()
    elif args.command == 'sync':
        (REPO_ROOT / 'config/extensions.json').write_bytes(msgspec.json.format(msgspec.json.encode(compose_catalog()), indent=2) + b'\n')
    else:
        if not args.sdk_ref:
            parser.error('export requires --sdk-ref (publish the SDK changes before pinning its commit)')
        packages = source_packages()
        unknown = set(args.package) - {package.package for package in packages}
        if unknown:
            parser.error(f'Unknown source packages: {sorted(unknown)}')
        for package in packages:
            if args.package and package.package not in args.package:
                continue
            destination = args.output_dir.resolve() / package.package
            export_repository(package, destination, sdk_ref=args.sdk_ref, initialize_git=args.git)
            print(destination)


if __name__ == '__main__':
    main()
