"""Install all checkout environments except the GPU runtime used for inference."""
from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]


def ci_environments(manifest_path: Path) -> list[str]:
    with manifest_path.open('rb') as source:
        manifest = tomllib.load(source)
    names = []
    for name, definition in manifest['environments'].items():
        features = definition if isinstance(definition, list) else definition['features']
        if 'onnx' not in features:
            names.append(name)
    if 'onnx-describe' not in names:
        raise ValueError('CI requires the lightweight onnx-describe environment')
    return sorted(names)


def main() -> None:
    manifest = REPO_ROOT / 'pixi.toml'
    names = ci_environments(manifest)
    print('Install CI environments (no GPU inference): ' + ', '.join(names), flush=True)
    command = ['pixi', 'install', '--locked', '--manifest-path', str(manifest)]
    for name in names:
        command.extend(['-e', name])
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == '__main__':
    main()
