"""Build release wheels from disposable copies, never from editable source trees."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


def build_wheels(
    package_dirs: list[Path], *, wheels_dir: Path, staging_dir: Path, web_bundle: Path
) -> None:
    if not (web_bundle / "index.html").is_file():
        raise FileNotFoundError(f"Build Web Studio first: {web_bundle / 'index.html'}")
    for output in (wheels_dir, staging_dir):
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
    for package_dir in package_dirs:
        staged = staging_dir / package_dir.name
        shutil.copytree(
            package_dir, staged,
            ignore=shutil.ignore_patterns(
                ".git", ".pixi", ".sdk", ".platform", ".media-dependency", "__pycache__", "*.pyc", "*.egg-info",
                ".pytest_cache", ".mypy_cache", "node_modules", "build", "dist", "web_dist",
            ),
        )
        if package_dir.name == "f8studio_server":
            shutil.copytree(web_bundle, staged / "f8studio_server" / "web_dist")
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
             "--wheel-dir", str(wheels_dir.resolve()), str(staged.resolve())],
            check=True, cwd=staging_dir,
        )
