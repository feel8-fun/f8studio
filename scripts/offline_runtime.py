"""Build the offline base runtime from release wheels and the release lock."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tomllib
import urllib.request

import yaml

from f8pysdk.service_paths import ServicePaths

VERSION = '0.7.11'
DIGESTS = {
    'pixi-pack-x86_64-pc-windows-msvc.exe': '12dde5b363fb7c5fb6215e0747638b0486a0bd3f47ef653b23ce020bc1936263',
    'pixi-unpack-x86_64-pc-windows-msvc.exe': '4b27cae647100dae4a321954461fe5863d25aa3eb9a919fb302f20509090bc72',
    'pixi-pack-x86_64-unknown-linux-musl': '4f2e6b70433c846974b9bd1356456335028afbe23f1fdf856ffa022ed691fc22',
    'pixi-unpack-x86_64-unknown-linux-musl': '8191f586b734e634e2f1644e553dcb8e07718d0bafbfefb65c5e6e22b4d484b7',
}


def tool(name: str, cache: Path) -> Path:
    suffix = 'x86_64-pc-windows-msvc.exe' if os.name == 'nt' else 'x86_64-unknown-linux-musl'
    asset = f'{name}-{suffix}'
    destination = cache / VERSION / asset
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.is_file():
        temporary = destination.with_suffix('.download')
        urllib.request.urlretrieve(f'https://github.com/Quantco/pixi-pack/releases/download/v{VERSION}/{asset}', temporary)
        temporary.replace(destination)
    with destination.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != DIGESTS[asset]:
        raise ValueError(f'Checksum mismatch for {destination}; remove it and retry')
    destination.chmod(0o755)
    return destination


def rewrite_base_services(root: Path, *, windows: bool, preset: str = 'standard') -> None:
    if preset not in {'standard', 'core'}:
        raise ValueError(f'Unknown distribution preset: {preset}')
    manifest = tomllib.loads((root / 'pixi.toml').read_text())
    tasks: dict[str, str] = {}
    for feature in manifest['environments']['studio-runtime']['features']:
        tasks.update(manifest['feature'][feature].get('tasks', {}))
    index_path = root / 'config/service-index.json'
    index = json.loads(index_path.read_text())
    paths = ServicePaths.for_index(index_path)
    catalog_path = root / 'config/extensions.json'
    catalog = json.loads(catalog_path.read_text())
    environments: dict[str, str] = {}
    for item in index['services']:
        for relative in item['manifests'].values():
            path = paths.package_path(relative, relative_to=index_path.parent)
            document = yaml.safe_load(path.read_text())
            launch = document['launch']
            if launch['command'] not in {'pixi', 'pixi.exe'}:
                continue
            args = launch['args']
            if len(args) != 4 or args[:2] != ['run', '-e']:
                raise ValueError(f'Unsupported release launch: {path}: {args}')
            previous_environment = environments.setdefault(item['serviceClass'], args[2])
            if previous_environment != args[2]:
                raise ValueError(f'Platform environments disagree for {item["serviceClass"]}')
            if args[2] != 'studio-runtime':
                continue
            command = shlex.split(tasks[args[3]])
            if command[:2] != ['python', '-m']:
                raise ValueError(f'Base task must be an explicit Python module: {command}')
            launch['command'] = './env/python.exe' if windows else './env/bin/python'
            launch['args'] = ['-I', *command[1:]]
            launch['workdir'] = '${F8_PACKAGE_ROOT}'
            path.write_text(yaml.safe_dump(document, sort_keys=False), encoding='utf-8')
    preinstalled: list[str] = []
    for extension in catalog['extensions']:
        names = {environments[name] for name in extension['serviceClasses'] if name in environments}
        if len(names) > 1:
            raise ValueError(f'Extension {extension["extensionId"]} references multiple runtimes')
        if not names:
            if extension.get('runtime', {}).get('kind') != 'bundled':
                extension['runtime'] = {'kind': 'native'}
        else:
            environment = names.pop()
            extension['runtime'] = ({'kind': 'bundled'} if environment == 'studio-runtime'
                                    else {'kind': 'pixi', 'environment': environment})
        if extension['runtime']['kind'] != 'pixi' and preset == 'standard':
            preinstalled.append(extension['extensionId'])
    catalog['preinstalled'] = preinstalled
    catalog_path.write_text(json.dumps(catalog, indent=2) + '\n')


def bundle_base_runtime(root: Path, *, cache: Path, preset: str = 'standard') -> None:
    pack = tool('pixi-pack', cache / 'tools')
    unpack = tool('pixi-unpack', cache / 'tools')
    destination = root / 'offline'
    destination.mkdir()
    subprocess.run([str(pack), '-e', 'studio-runtime', '-p', 'win-64' if os.name == 'nt' else 'linux-glibc228',
                    '-o', str(destination / 'base-runtime.tar'), '--use-cache', str(cache / 'packages'),
                    str(root / 'pixi.toml')], check=True)
    shutil.copy2(unpack, destination / ('pixi-unpack.exe' if os.name == 'nt' else 'pixi-unpack'))
    rewrite_base_services(root, windows=os.name == 'nt', preset=preset)
