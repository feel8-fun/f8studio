#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKFILE_PATH = REPO_ROOT / "conan.lock"
USER_PRESETS_PATH = REPO_ROOT / "CMakeUserPresets.json"
GENERATED_PRESETS_PATH = REPO_ROOT / "build" / "Release" / "generators" / "CMakePresets.json"


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C++ CI entrypoint for Conan + CMake.")
    parser.add_argument(
        "command",
        choices=("bootstrap", "configure", "build", "lock-refresh"),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
