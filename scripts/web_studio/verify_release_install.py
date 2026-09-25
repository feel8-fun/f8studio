from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_SOURCE_DIR = REPO_ROOT / "packages" / "f8studio_web" / "dist"
WEB_PACKAGE_DIR = REPO_ROOT / "packages" / "f8studio_server" / "f8studio_server" / "web_dist"
PACKAGE_DIRS = (
    REPO_ROOT / "packages" / "f8pysdk",
    REPO_ROOT / "packages" / "f8studio_core",
    REPO_ROOT / "packages" / "f8media_protocol",
    REPO_ROOT / "packages" / "f8media_gateway",
    REPO_ROOT / "external" / "f8unitymods",
    REPO_ROOT / "packages" / "f8studio_server",
)


def _run(command: list[str], *, cwd: Path = REPO_ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True, env=os.environ.copy())


def _stage_web_bundle() -> None:
    npm_command = "npm.cmd" if os.name == "nt" else "npm"
    _run([npm_command, "--prefix", "packages/f8studio_web", "ci"])
    _run([npm_command, "--prefix", "packages/f8studio_web", "run", "build"])
    if not (WEB_SOURCE_DIR / "index.html").is_file():
        raise FileNotFoundError(f"Web build did not produce {WEB_SOURCE_DIR / 'index.html'}")
    WEB_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    for child in WEB_PACKAGE_DIR.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    shutil.copytree(WEB_SOURCE_DIR, WEB_PACKAGE_DIR, dirs_exist_ok=True)


def _build_wheels(wheels_dir: Path) -> tuple[Path, ...]:
    wheels_dir.mkdir(parents=True, exist_ok=True)
    for package_dir in PACKAGE_DIRS:
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--quiet",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheels_dir),
                str(package_dir),
            ]
        )
    wheels = tuple(sorted(wheels_dir.glob("*.whl")))
    if len(wheels) != len(PACKAGE_DIRS):
        raise RuntimeError(f"expected {len(PACKAGE_DIRS)} wheels, found {len(wheels)}")
    return wheels


def _assert_wheel_contents(wheels: tuple[Path, ...]) -> None:
    server_wheels = tuple(path for path in wheels if path.name.startswith("f8studio_server-"))
    if len(server_wheels) != 1:
        raise RuntimeError("expected exactly one f8studio-server wheel")
    with zipfile.ZipFile(server_wheels[0]) as archive:
        names = tuple(archive.namelist())
    if "f8studio_server/web_dist/index.html" not in names:
        raise RuntimeError("f8studio-server wheel does not contain the Web Studio entry page")
    forbidden = tuple(
        name for name in names if "qt" in Path(name).name.lower() and Path(name).suffix.lower() in {".dll", ".so", ".dylib"}
    )
    if forbidden:
        raise RuntimeError("server wheel contains Qt libraries: " + ", ".join(forbidden))


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _verify_installed_runtime(venv_dir: Path, wheels: tuple[Path, ...], work_dir: Path) -> None:
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
    python_executable = _venv_python(venv_dir)
    _run(
        [
            str(python_executable),
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--no-index",
            *(str(wheel) for wheel in wheels),
        ],
        cwd=work_dir,
    )
    smoke_code = """
from importlib import metadata
from pathlib import Path
import sys
import tempfile

from fastapi.testclient import TestClient
import f8media_gateway
import f8media_protocol
import f8pysdk
import f8studio_core
import f8studio_server
import f8unitymods_setup
from f8media_gateway.service import InProcessMediaGateway
from f8studio_server.app import create_app, default_web_dist

prefix = Path(sys.prefix).resolve()
modules = (f8media_gateway, f8media_protocol, f8pysdk, f8studio_core, f8studio_server, f8unitymods_setup)
for module in modules:
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(prefix):
        raise RuntimeError(f'{module.__name__} was not imported from the release environment: {module_path}')

forbidden = {'nodegraphqt', 'pyqt5', 'pyqt6', 'pyside2', 'pyside6', 'pyqtgraph', 'qtpy'}
installed = {str(dist.metadata['Name'] or '').lower() for dist in metadata.distributions()}
if installed & forbidden:
    raise RuntimeError(f'Qt distributions present: {sorted(installed & forbidden)}')

web_dist = default_web_dist()
if not web_dist.is_relative_to(prefix) or not (web_dist / 'index.html').is_file():
    raise RuntimeError(f'embedded Web bundle is unavailable: {web_dist}')
with tempfile.TemporaryDirectory(prefix='f8studio-wheel-smoke-') as data_dir:
    app = create_app(data_dir=Path(data_dir), service_roots=(), media_gateway=InProcessMediaGateway())
    with TestClient(app) as client:
        health = client.get('/api/health')
        root = client.get('/')
    if health.status_code != 200:
        raise RuntimeError(f'health request failed: {health.status_code} {health.text}')
    if root.status_code != 200 or '<div id="root"></div>' not in root.text:
        raise RuntimeError(f'embedded Web root failed: {root.status_code}')
print(f'non-editable release smoke passed: {web_dist}')
"""
    _run([str(python_executable), "-P", "-c", smoke_code], cwd=work_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify non-editable Web Studio release wheels.")
    parser.add_argument("--keep", action="store_true", help="Keep temporary wheel and venv output under build/.")
    args = parser.parse_args()
    _stage_web_bundle()
    if args.keep:
        work_dir = REPO_ROOT / "build" / "release-smoke"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        wheels = _build_wheels(work_dir / "wheels")
        _assert_wheel_contents(wheels)
        _verify_installed_runtime(work_dir / "venv", wheels, work_dir)
        print(f"release smoke artifacts: {work_dir}")
        return 0

    with tempfile.TemporaryDirectory(prefix="f8studio-release-smoke-") as temp_dir:
        work_dir = Path(temp_dir)
        wheels = _build_wheels(work_dir / "wheels")
        _assert_wheel_contents(wheels)
        _verify_installed_runtime(work_dir / "venv", wheels, work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
