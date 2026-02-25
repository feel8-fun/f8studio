#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKFILE_PATH = REPO_ROOT / "conan.lock"
USER_PRESETS_PATH = REPO_ROOT / "CMakeUserPresets.json"
GENERATED_PRESETS_PATH = REPO_ROOT / "build" / "Release" / "generators" / "CMakePresets.json"
PIXI_TOML_PATH = REPO_ROOT / "pixi.toml"
PIXI_LOCK_PATH = REPO_ROOT / "pixi.lock"

PYTHON_PACKAGE_DIRS: dict[str, str] = {
    "f8pysdk": "packages/f8pysdk",
    "f8pyengine": "packages/f8pyengine",
    "f8pystudio": "packages/f8pystudio",
    "f8pydl": "packages/f8pydl",
    "f8pymppose": "packages/f8pymppose",
}

CPP_DEPLOY_TARGETS: tuple[str, ...] = (
    "f8implayer_service_deploy_runtime",
    "f8cvkit_template_match_service_deploy_runtime",
    "f8cvkit_dense_optflow_service_deploy_runtime",
    "f8cvkit_tracking_service_deploy_runtime",
    "f8cvkit_video_stab_service_deploy_runtime",
    "f8screencap_service_deploy_runtime",
    "f8audiocap_service_deploy_runtime",
)


def _run(command: list[str]) -> None:
    ccache_tmp_dir = REPO_ROOT / ".ccache-tmp"
    ccache_tmp_dir.mkdir(parents=True, exist_ok=True)

    command_env = os.environ.copy()
    command_env["CCACHE_TEMPDIR"] = str(ccache_tmp_dir)

    subprocess.run(command, check=True, cwd=REPO_ROOT, env=command_env)


def _bootstrap() -> None:
    if USER_PRESETS_PATH.is_file():
        USER_PRESETS_PATH.unlink()

    if not LOCKFILE_PATH.is_file():
        raise FileNotFoundError(
            "Missing conan.lock at repository root. Run `python scripts/cpp_ci.py lock-refresh` first."
        )

    _run(["conan", "profile", "detect", "--force"])
    _run(["conan", "export", "./conan_recipes/libmpv_recipe"])
    _run(["conan", "export", "./conan_recipes/ytdlp_recipe"])
    _run(
        [
            "conan",
            "install",
            ".",
            "-of",
            ".",
            "-s",
            "build_type=Release",
            "--build=missing",
            "--lockfile",
            "conan.lock",
            "--lockfile-partial",
        ]
    )

    if not GENERATED_PRESETS_PATH.is_file():
        raise FileNotFoundError(
            "Expected Conan-generated preset file is missing: build/Release/generators/CMakePresets.json"
        )


def _configure() -> None:
    _run(
        [
            "cmake",
            "--preset",
            "conan-release",
            "-DF8_DEPLOY_SERVICE_CLEAN=OFF",
            "-DF8_DEPLOY_SERVICE_RUNTIME_POST_BUILD=OFF",
        ]
    )


def _build() -> None:
    _run(["cmake", "--build", "--preset", "conan-release", "--parallel"])


def _lock_refresh() -> None:
    _run(["conan", "profile", "detect", "--force"])
    _run(["conan", "export", "./conan_recipes/libmpv_recipe"])
    _run(["conan", "export", "./conan_recipes/ytdlp_recipe"])
    _run(["conan", "lock", "create", ".", "--lockfile-out", "conan.lock", "--build=missing"])


def _platform_info() -> tuple[str, str, str]:
    if os.name == "nt":
        return ("windows-x86_64", "win", "zip")
    if sys.platform.startswith("linux"):
        return ("linux-x86_64", "linux", "gztar")
    raise RuntimeError(f"Unsupported platform for dist packaging: {sys.platform}")


def _normalize_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _find_wheel_for_distribution(wheels_dir: Path, distribution: str) -> Path:
    normalized = _normalize_dist_name(distribution)
    wheels = sorted(wheels_dir.glob("*.whl"))
    for wheel in wheels:
        wheel_name = wheel.name.lower()
        if wheel_name.startswith(f"{normalized}-"):
            return wheel
    raise FileNotFoundError(f"Wheel for distribution '{distribution}' was not found in {wheels_dir}")


def _build_python_wheels(wheels_dir: Path) -> dict[str, str]:
    if wheels_dir.exists():
        shutil.rmtree(wheels_dir)
    wheels_dir.mkdir(parents=True, exist_ok=True)

    for package_dir in PYTHON_PACKAGE_DIRS.values():
        _run(
            [
                "python",
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheels_dir),
                package_dir,
            ]
        )

    dependency_to_wheel: dict[str, str] = {}
    for dependency_name in PYTHON_PACKAGE_DIRS:
        wheel_path = _find_wheel_for_distribution(wheels_dir, dependency_name)
        dependency_to_wheel[dependency_name] = f"wheels/{wheel_path.name}"
    return dependency_to_wheel


def _render_dist_pixi_toml(dependency_to_wheel: dict[str, str]) -> str:
    pixi_text = PIXI_TOML_PATH.read_text(encoding="utf-8")
    for dependency_name, wheel_rel_path in dependency_to_wheel.items():
        pattern = re.compile(
            rf'^{dependency_name}\s*=\s*\{{\s*path\s*=\s*"[^"]+"\s*,\s*editable\s*=\s*true\s*\}}\s*$',
            flags=re.MULTILINE,
        )
        replacement = f'{dependency_name} = {{ path = "{wheel_rel_path}" }}'
        pixi_text, replacement_count = pattern.subn(replacement, pixi_text, count=1)
        if replacement_count != 1:
            raise ValueError(
                f"Expected exactly one editable path dependency entry for '{dependency_name}' in pixi.toml"
            )
    return pixi_text


def _deploy_cpp_runtime() -> None:
    _run(["cmake", "--build", "--preset", "conan-release", "--target", *CPP_DEPLOY_TARGETS, "--parallel"])


def _dist_runtime() -> None:
    if not GENERATED_PRESETS_PATH.is_file():
        _bootstrap()

    _configure()
    _deploy_cpp_runtime()

    platform_tag, platform_dir, archive_format = _platform_info()
    dist_base_dir = REPO_ROOT / "build" / "dist"
    dist_name = f"f8runtime-{platform_tag}"
    dist_dir = dist_base_dir / dist_name

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    (dist_dir / "services").mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "services", dist_dir / "services", dirs_exist_ok=True)

    wheels_dir = dist_dir / "wheels"
    dependency_to_wheel = _build_python_wheels(wheels_dir)
    dist_pixi_text = _render_dist_pixi_toml(dependency_to_wheel)
    dist_manifest_path = dist_dir / "pixi.toml"
    dist_manifest_path.write_text(dist_pixi_text, encoding="utf-8")
    _run(["pixi", "lock", "--manifest-path", str(dist_manifest_path), "--no-install"])

    readme_text = (
        "# f8 Runtime Dist\n\n"
        "This bundle contains:\n"
        "- pixi.toml + pixi.lock\n"
        "- services/**\n"
        "- Python wheels for local non-editable install\n\n"
        "Bootstrap:\n"
        "1. Install Pixi.\n"
        "2. `pixi install --frozen -e default`\n"
        "3. Run your service command via `pixi run ...`.\n\n"
        f"Platform runtime binaries are under `services/**/{platform_dir}`.\n"
    )
    (dist_dir / "README.md").write_text(readme_text, encoding="utf-8")

    archive_path = shutil.make_archive(
        base_name=str(dist_base_dir / dist_name),
        format=archive_format,
        root_dir=dist_base_dir,
        base_dir=dist_name,
    )
    print(f"dist archive: {archive_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C++ CI entrypoint for Conan + CMake.")
    parser.add_argument(
        "command",
        choices=("bootstrap", "configure", "build", "lock-refresh", "dist-runtime"),
        help="Action to run.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "bootstrap":
        _bootstrap()
    elif args.command == "configure":
        _configure()
    elif args.command == "build":
        _build()
    elif args.command == "lock-refresh":
        _lock_refresh()
    elif args.command == "dist-runtime":
        _dist_runtime()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
