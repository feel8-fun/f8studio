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
PIXI_CPP_ENV_PATH = (REPO_ROOT / ".pixi" / "envs" / "cpp").resolve()


def _run(command: list[str]) -> None:
    ccache_tmp_dir = REPO_ROOT / ".ccache-tmp"
    ccache_tmp_dir.mkdir(parents=True, exist_ok=True)

    command_env = os.environ.copy()
    command_env["CCACHE_TEMPDIR"] = str(ccache_tmp_dir)
    if os.name == "nt":
        pixi_cuda_root = PIXI_CPP_ENV_PATH / "Library"
        pixi_cuda_bin = pixi_cuda_root / "bin"
        existing_path = command_env.get("PATH", "")
        command_env["PATH"] = str(pixi_cuda_bin) + os.pathsep + existing_path
        command_env["CUDA_PATH"] = str(pixi_cuda_root)
        command_env["CUDA_HOME"] = str(pixi_cuda_root)
        command_env["CUDA_TOOLKIT_ROOT_DIR"] = str(pixi_cuda_root)
        command_env["CUDAToolkit_ROOT"] = str(pixi_cuda_root)
        for key in list(command_env.keys()):
            if key.startswith("CUDA_PATH_V"):
                del command_env[key]

    subprocess.run(command, check=True, cwd=REPO_ROOT, env=command_env)


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def _enforce_pixi_nvcc() -> None:
    nvcc_proc = _run_capture(["nvcc", "--version"])
    if nvcc_proc.returncode != 0:
        raise RuntimeError(
            "CUDA compiler check failed: `nvcc --version` is unavailable in current shell. "
            "Run `pixi install -e cpp` and execute builds via `pixi run -e cpp ...`."
        )

    if os.name == "nt":
        path_proc = _run_capture(["where", "nvcc"])
    else:
        path_proc = _run_capture(["which", "-a", "nvcc"])
    if path_proc.returncode != 0:
        raise RuntimeError(
            "CUDA compiler path check failed: unable to resolve `nvcc` path. "
            "Run `pixi install -e cpp` and execute builds via `pixi run -e cpp ...`."
        )

    nvcc_paths: list[Path] = []
    for raw_line in path_proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        candidate = Path(line).expanduser().resolve()
        nvcc_paths.append(candidate)

    if not nvcc_paths:
        raise RuntimeError(
            "CUDA compiler path check failed: `where/which nvcc` returned no usable paths. "
            "Run `pixi install -e cpp` and execute builds via `pixi run -e cpp ...`."
        )

    first_nvcc = nvcc_paths[0]
    if PIXI_CPP_ENV_PATH in first_nvcc.parents:
        return

    paths_text = ", ".join(str(path) for path in nvcc_paths)
    raise RuntimeError(
        "CUDA compiler resolution is ambiguous or not pixi-first. "
        f"Expected first `nvcc` under `{PIXI_CPP_ENV_PATH}`, got: {paths_text}. "
        "Run `pixi install -e cpp` and ensure `pixi run -e cpp ...` is used."
    )


def _conan_msvc_settings_args() -> list[str]:
    if os.name != "nt":
        return []
    # Make profile resolution deterministic inside pixi shells where `conan profile detect`
    # may not infer compiler settings.
    return [
        "-s:h",
        "os=Windows",
        "-s:h",
        "arch=x86_64",
        "-s:h",
        "compiler=msvc",
        "-s:h",
        "compiler.version=194",
        "-s:h",
        "compiler.runtime=dynamic",
        "-s:h",
        "compiler.runtime_type=Release",
        "-s:h",
        "compiler.cppstd=17",
        "-s:b",
        "os=Windows",
        "-s:b",
        "arch=x86_64",
        "-s:b",
        "compiler=msvc",
        "-s:b",
        "compiler.version=194",
        "-s:b",
        "compiler.runtime=dynamic",
        "-s:b",
        "compiler.runtime_type=Release",
        "-s:b",
        "compiler.cppstd=17",
    ]


def _bootstrap() -> None:
    if USER_PRESETS_PATH.is_file():
        USER_PRESETS_PATH.unlink()

    if not LOCKFILE_PATH.is_file():
        raise FileNotFoundError(
            "Missing conan.lock at repository root. Run `python scripts/cpp_ci.py lock-refresh` first."
        )

    _enforce_pixi_nvcc()
    _run(
        [
            "conan",
            "install",
            ".",
            "-of",
            ".",
            "-s",
            "build_type=Release",
            *_conan_msvc_settings_args(),
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
    if LOCKFILE_PATH.is_file():
        LOCKFILE_PATH.unlink()
    _run(
        [
            "conan",
            "lock",
            "create",
            ".",
            *_conan_msvc_settings_args(),
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
