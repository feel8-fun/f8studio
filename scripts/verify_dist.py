"""Verify the actual offline archive in a relocated directory, without installing dependencies."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]


def verify_distribution(root: Path) -> None:
    root = root.resolve()
    env = {key: value for key, value in os.environ.items()
           if key not in {'PYTHONPATH', 'PYTHONHOME', 'CONDA_PREFIX', 'PIXI_ENVIRONMENT_NAME'}}
    env['F8_SERVICE_INDEX'] = str(root / 'config/service-index.json')
    env['F8_MODEL_ROOT'] = str(root / 'resources/models')
    # Eliminate checkout/Pixi tool directories from PATH. Activation supplies bundled DLLs.
    if os.name == 'nt':
        system = Path(os.environ['SystemRoot'])
        env['PATH'] = os.pathsep.join([str(system / 'System32'), str(system)])
        launcher = ['cmd.exe', '/d', '/c', str(root / 'f8studio.cmd')]
    else:
        env['PATH'] = '/usr/bin:/bin'
        launcher = [str(root / 'f8studio')]
    started = time.monotonic()
    subprocess.run([*launcher, '--help'], cwd=root, env=env, check=True, timeout=300)
    marker = root / '.runtime-location'
    prepared = marker.stat().st_mtime_ns
    subprocess.run([*launcher, '--help'], cwd=root, env=env, check=True, timeout=30)
    if marker.stat().st_mtime_ns != prepared:
        raise RuntimeError('Second launch unpacked the runtime again')
    probe = REPO_ROOT / 'scripts/quality/check_offline_runtime.py'
    if os.name == 'nt':
        script = root / 'verify-runtime.cmd'
        script.write_text('@echo off\r\ncall activate.bat\r\nif errorlevel 1 exit /b %errorlevel%\r\n'
                          f'env\\python.exe -I "{probe}" "%CD%"\r\nexit /b %errorlevel%\r\n')
        command = ['cmd.exe', '/d', '/c', str(script)]
    else:
        command = ['sh', '-c', '. "$1/activate.sh"; exec "$1/env/bin/python" -I "$2" "$1"',
                   'verify-runtime', str(root), str(probe)]
    subprocess.run(command, cwd=root, env=env, check=True, timeout=180)
    print(f'Offline release verification passed in {time.monotonic() - started:.1f}s')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='f8 offline release ') as temporary:
        output = Path(temporary)
        shutil.unpack_archive(args.archive.resolve(), output)
        roots = list(output.glob('*/offline/base-runtime.tar'))
        if len(roots) != 1:
            raise ValueError('Expected exactly one offline base runtime in archive')
        verify_distribution(roots[0].parent.parent)


if __name__ == '__main__':
    main()
