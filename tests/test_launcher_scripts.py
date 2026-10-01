from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def offline_release(tmp_path: Path) -> Path:
    release = tmp_path / 'release with spaces'
    (release / 'offline').mkdir(parents=True)
    for name in ('f8studio', 'install_env.sh'):
        shutil.copy2(Path('scripts/launchers') / name, release / name)
        (release / name).chmod(0o755)
    unpack = release / 'offline/pixi-unpack'
    unpack.write_text('''#!/bin/sh
set -eu
printf 'unpacked\n' >> unpack-count
mkdir -p env/bin
printf '\n' > activate.sh
cat > env/bin/python <<'PYTHON'
#!/bin/sh
if [ "$2" = '-c' ]; then exit 0; fi
printf '%s\n' "$@" > arguments
PYTHON
chmod +x env/bin/python
''')
    unpack.chmod(0o755)
    return release


@pytest.mark.skipif(os.name == 'nt', reason='Shell launcher; Windows is exercised by offline dist CI')
def test_offline_launcher_unpacks_once_forwards_args_and_recovers_after_move(offline_release: Path) -> None:
    release = offline_release
    subprocess.run([str(release / 'f8studio'), '--web-dist', 'assets with spaces'], check=True)
    assert (release / 'arguments').read_text().splitlines() == [
        '-I', '-m', 'f8studio_server', '--tray', '--open-browser', '--web-dist', 'assets with spaces']
    subprocess.run([str(release / 'f8studio'), '--help'], check=True)
    assert (release / 'unpack-count').read_text().splitlines() == ['unpacked']
    moved = release.with_name('moved release')
    release.rename(moved)
    subprocess.run([str(moved / 'f8studio'), '--help'], check=True)
    assert (moved / 'unpack-count').read_text().splitlines() == ['unpacked', 'unpacked']


@pytest.mark.skipif(os.name == 'nt', reason='Shell launcher')
def test_unpack_failure_is_not_marked_ready(offline_release: Path) -> None:
    (offline_release / 'offline/pixi-unpack').write_text('#!/bin/sh\nexit 23\n')
    result = subprocess.run([str(offline_release / 'f8studio')])
    assert result.returncode == 23
    assert not (offline_release / '.runtime-location').exists()
    assert not (offline_release / '.runtime-install-lock').exists()
    assert not (offline_release / 'arguments').exists()


@pytest.mark.skipif(os.name == 'nt', reason='Shell launcher')
def test_concurrent_install_does_not_modify_runtime(offline_release: Path) -> None:
    (offline_release / '.runtime-install-lock').mkdir()
    result = subprocess.run([str(offline_release / 'f8studio')])
    assert result.returncode == 2
    assert not (offline_release / 'unpack-count').exists()
