#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKFILE_PATH = REPO_ROOT / "conan.lock"
USER_PRESETS_PATH = REPO_ROOT / "CMakeUserPresets.json"
GENERATED_PRESETS_PATH = REPO_ROOT / "build" / "Release" / "generators" / "CMakePresets.json"
GENERATED_PRESETS_FALLBACK_PATH = REPO_ROOT / "build" / "generators" / "CMakePresets.json"


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
    _run(
        [
            "conan",
            "install",
            ".",
            "-of",
            ".",
            "-s",
            "build_type=Release",
            "-s",
            "compiler.cppstd=17",
            "--build=missing",
            "--lockfile",
            "conan.lock",
            "--lockfile-partial",
        ]
    )

    if not GENERATED_PRESETS_PATH.is_file() and not GENERATED_PRESETS_FALLBACK_PATH.is_file():
        raise FileNotFoundError(
            "Expected Conan-generated preset file is missing: "
            "build/Release/generators/CMakePresets.json or build/generators/CMakePresets.json"
        )


def _resolve_conan_presets() -> tuple[str, str]:
    for presets_path in (GENERATED_PRESETS_PATH, GENERATED_PRESETS_FALLBACK_PATH):
        if not presets_path.is_file():
            continue
        presets = json.loads(presets_path.read_text(encoding="utf-8"))
        configure_presets = presets.get("configurePresets", [])
        build_presets = presets.get("buildPresets", [])

        configure_preset_names = {
            preset.get("name") for preset in configure_presets if isinstance(preset.get("name"), str)
        }
        build_preset_names = {preset.get("name") for preset in build_presets if isinstance(preset.get("name"), str)}

        configure_preset_name = ""
        if "conan-default" in configure_preset_names:
            configure_preset_name = "conan-default"
        elif "conan-release" in configure_preset_names:
            configure_preset_name = "conan-release"
        elif len(configure_preset_names) == 1:
            configure_preset_name = next(iter(configure_preset_names))

        build_preset_name = ""
        if "conan-release" in build_preset_names:
            build_preset_name = "conan-release"
        elif "conan-default" in build_preset_names:
            build_preset_name = "conan-default"
        elif configure_preset_name and configure_preset_name in build_preset_names:
            build_preset_name = configure_preset_name
        elif len(build_preset_names) == 1:
            build_preset_name = next(iter(build_preset_names))

        if configure_preset_name and build_preset_name:
            return (configure_preset_name, build_preset_name)

        raise FileNotFoundError(
            "Unable to resolve Conan CMake presets from generated CMakePresets.json. "
            f"configurePresets={sorted(configure_preset_names)}, buildPresets={sorted(build_preset_names)}"
        )
    raise FileNotFoundError("Unable to resolve Conan CMake preset paths from generated CMakePresets.json")


def _configure() -> None:
    configure_preset_name, _ = _resolve_conan_presets()
    _run(
        [
            "cmake",
            "--preset",
            configure_preset_name,
            "-DF8_DEPLOY_SERVICE_CLEAN=OFF",
            "-DF8_DEPLOY_SERVICE_RUNTIME_POST_BUILD=OFF",
        ]
    )


def _build() -> None:
    _, build_preset_name = _resolve_conan_presets()
    _run(["cmake", "--build", "--preset", build_preset_name, "--parallel"])


def _lock_refresh() -> None:
    _run(["conan", "profile", "detect", "--force"])
    if LOCKFILE_PATH.is_file():
        LOCKFILE_PATH.unlink()
    _run(
        [
            "conan",
            "lock",
            "create",
            ".",
            "-s",
            "compiler.cppstd=17",
            "--lockfile-out",
            "conan.lock",
            "--build=missing",
        ]
    )


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
