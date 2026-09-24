#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKFILE_PATH = REPO_ROOT / "conan.lock"
USER_PRESETS_PATH = REPO_ROOT / "CMakeUserPresets.json"
DEFAULT_GENERATED_PRESETS_PATH = REPO_ROOT / "build" / "Release" / "generators" / "CMakePresets.json"
GENERATED_PRESETS_PATH = DEFAULT_GENERATED_PRESETS_PATH
GENERATED_PRESET_CANDIDATES = (
    REPO_ROOT / "build" / "generators" / "CMakePresets.json",
    DEFAULT_GENERATED_PRESETS_PATH,
)
CONAN_CONFIGURE_PRESET_NAME = "conan-release"
CONAN_BUILD_PRESET_NAME = "conan-release"


@dataclass(frozen=True)
class ConanPresetSelection:
    configure_preset_name: str
    build_preset_name: str


def _pixi_cpp_env_path() -> Path:
    return REPO_ROOT / ".pixi" / "envs" / "cpp"


def _cpp_tool(name: str) -> str:
    tool_path = _pixi_cpp_env_path() / "bin" / name
    if tool_path.is_file():
        return str(tool_path)
    return name


def _run(command: list[str], *, use_pixi_cpp_paths: bool = False, use_host_pkg_config: bool = False) -> None:
    ccache_tmp_dir = REPO_ROOT / ".ccache-tmp"
    ccache_tmp_dir.mkdir(parents=True, exist_ok=True)

    command_env = os.environ.copy()
    command_env["CCACHE_TEMPDIR"] = str(ccache_tmp_dir)
    if use_host_pkg_config:
        _prepend_path_list(command_env, "PATH", [Path("/usr/bin"), Path("/usr/local/bin")])
    if use_pixi_cpp_paths:
        _apply_pixi_cpp_env(command_env)

    subprocess.run(command, check=True, cwd=REPO_ROOT, env=command_env)


def _prepend_path_list(env: dict[str, str], name: str, paths: list[Path]) -> None:
    existing_value = env.get(name, "")
    path_values = [str(path) for path in paths if path.exists()]
    if existing_value:
        path_values.append(existing_value)
    if path_values:
        env[name] = os.pathsep.join(path_values)


def _apply_pixi_cpp_env(env: dict[str, str]) -> None:
    pixi_cpp_env = _pixi_cpp_env_path()
    if not pixi_cpp_env.is_dir():
        return

    pixi_prefixes = [pixi_cpp_env]
    pixi_library_dirs = [pixi_cpp_env / "lib"]
    pixi_include_dirs = [pixi_cpp_env / "include"]
    pixi_pkg_config_dirs = [
        pixi_cpp_env / "lib" / "pkgconfig",
        pixi_cpp_env / "share" / "pkgconfig",
    ]
    pixi_bin_dirs = [pixi_cpp_env / "bin"]

    pixi_windows_prefix = pixi_cpp_env / "Library"
    if pixi_windows_prefix.is_dir():
        pixi_prefixes.insert(0, pixi_windows_prefix)
        pixi_library_dirs.insert(0, pixi_windows_prefix / "lib")
        pixi_include_dirs.insert(0, pixi_windows_prefix / "include")
        pixi_pkg_config_dirs.insert(0, pixi_windows_prefix / "share" / "pkgconfig")
        pixi_pkg_config_dirs.insert(0, pixi_windows_prefix / "lib" / "pkgconfig")
        pixi_bin_dirs.insert(0, pixi_windows_prefix / "bin")

    _prepend_path_list(env, "PATH", pixi_bin_dirs)
    _prepend_path_list(env, "CMAKE_PREFIX_PATH", pixi_prefixes)
    _prepend_path_list(env, "CMAKE_LIBRARY_PATH", pixi_library_dirs)
    _prepend_path_list(env, "CMAKE_INCLUDE_PATH", pixi_include_dirs)
    _prepend_path_list(
        env,
        "PKG_CONFIG_PATH",
        pixi_pkg_config_dirs,
    )


def _repo_relative_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _generated_preset_path_from_user_presets() -> Path | None:
    if not USER_PRESETS_PATH.is_file():
        return None

    user_presets = json.loads(USER_PRESETS_PATH.read_text(encoding="utf-8"))
    include_entries = user_presets.get("include")
    if not isinstance(include_entries, list):
        return None

    for include_entry in include_entries:
        if not isinstance(include_entry, str):
            continue
        include_path = (REPO_ROOT / include_entry).resolve()
        if include_path.name == "CMakePresets.json" and include_path.is_file():
            return include_path
    return None


def _generated_presets_path() -> Path:
    if GENERATED_PRESETS_PATH != DEFAULT_GENERATED_PRESETS_PATH and GENERATED_PRESETS_PATH.is_file():
        return GENERATED_PRESETS_PATH

    user_preset_path = _generated_preset_path_from_user_presets()
    if user_preset_path is not None:
        return user_preset_path

    for candidate_path in GENERATED_PRESET_CANDIDATES:
        if candidate_path.is_file():
            return candidate_path

    checked_paths = ", ".join(_repo_relative_path(candidate_path) for candidate_path in GENERATED_PRESET_CANDIDATES)
    raise FileNotFoundError(f"Expected Conan-generated preset file is missing. Checked: {checked_paths}")


def _cmake_build_directory() -> Path:
    return _generated_presets_path().parent.parent


def _pixi_gtest_cmake_directory() -> Path | None:
    pixi_cpp_env = _pixi_cpp_env_path()
    candidates = (
        pixi_cpp_env / "lib" / "cmake" / "GTest",
        pixi_cpp_env / "Library" / "lib" / "cmake" / "GTest",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _bootstrap() -> None:
    if USER_PRESETS_PATH.is_file():
        USER_PRESETS_PATH.unlink()

    if not LOCKFILE_PATH.is_file():
        raise FileNotFoundError(
            "Missing conan.lock at repository root. Run `python scripts/cpp_ci.py lock-refresh` first."
        )

    _run([_cpp_tool("conan"), "profile", "detect", "--force"], use_host_pkg_config=True)
    _run(
        [
            _cpp_tool("conan"),
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
        ],
        use_host_pkg_config=True,
    )

    _generated_presets_path()


def _load_generated_presets() -> dict[str, object]:
    generated_presets_path = _generated_presets_path()
    return json.loads(generated_presets_path.read_text(encoding="utf-8"))


def _select_conan_release_presets() -> ConanPresetSelection:
    presets = _load_generated_presets()
    configure_presets = presets.get("configurePresets", [])
    build_presets = presets.get("buildPresets", [])

    configure_preset_names = {
        preset.get("name") for preset in configure_presets if isinstance(preset, dict) and isinstance(preset.get("name"), str)
    }
    build_preset_names = {
        preset.get("name") for preset in build_presets if isinstance(preset, dict) and isinstance(preset.get("name"), str)
    }

    for build_preset in build_presets:
        if not isinstance(build_preset, dict):
            continue
        build_preset_name = build_preset.get("name")
        if build_preset_name != CONAN_BUILD_PRESET_NAME:
            continue

        configure_preset_name = build_preset.get("configurePreset")
        if not isinstance(configure_preset_name, str):
            configure_preset_name = CONAN_CONFIGURE_PRESET_NAME
        if configure_preset_name not in configure_preset_names:
            raise FileNotFoundError(
                "Generated Conan release build preset references a missing configure preset. "
                f"configurePreset={configure_preset_name!r}, configurePresets={sorted(configure_preset_names)}"
            )
        return ConanPresetSelection(
            configure_preset_name=configure_preset_name,
            build_preset_name=CONAN_BUILD_PRESET_NAME,
        )

    raise FileNotFoundError(
        "Generated Conan presets must define the canonical release build preset. "
        f"configurePresets={sorted(configure_preset_names)}, buildPresets={sorted(build_preset_names)}"
    )


def _configure() -> None:
    _configure_release(build_tests=False)


def _configure_release(*, build_tests: bool) -> None:
    conan_presets = _select_conan_release_presets()
    build_tests_value = "ON" if build_tests else "OFF"
    configure_args = [
        _cpp_tool("cmake"),
        "--preset",
        conan_presets.configure_preset_name,
        f"-DBUILD_TESTS={build_tests_value}",
        "-DF8_DEPLOY_SERVICE_CLEAN=OFF",
        "-DF8_DEPLOY_SERVICE_RUNTIME_POST_BUILD=OFF",
        f"-DPKG_CONFIG_EXECUTABLE={_cpp_tool('pkg-config')}",
    ]
    gtest_cmake_directory = _pixi_gtest_cmake_directory()
    if build_tests and gtest_cmake_directory is not None:
        configure_args.append(f"-DGTest_DIR={gtest_cmake_directory}")
    _run(
        configure_args,
        use_pixi_cpp_paths=True,
    )


def _build() -> None:
    conan_presets = _select_conan_release_presets()
    _run([_cpp_tool("cmake"), "--build", "--preset", conan_presets.build_preset_name, "--parallel"], use_pixi_cpp_paths=True)
    _run(
        [
            _cpp_tool("cmake"),
            "--build",
            "--preset",
            conan_presets.build_preset_name,
            "--target",
            "f8_deploy_all_runtime",
            "--parallel",
        ],
        use_pixi_cpp_paths=True,
    )


def _build_target(target: str) -> None:
    target_s = str(target or "").strip()
    if not target_s:
        raise ValueError("target must be non-empty")
    conan_presets = _select_conan_release_presets()
    _run(
        [
            _cpp_tool("cmake"),
            "--build",
            "--preset",
            conan_presets.build_preset_name,
            "--target",
            target_s,
            "--parallel",
        ],
        use_pixi_cpp_paths=True,
    )


def _test() -> None:
    _configure_release(build_tests=True)
    _build_target("f8cppsdk_tests")
    _run(
        [
            _cpp_tool("ctest"),
            "--test-dir",
            str(_cmake_build_directory()),
            "--output-on-failure",
            "-R",
            "f8cppsdk_tests",
            "-C",
            "Release",
        ],
        use_pixi_cpp_paths=True,
    )


def _lock_refresh() -> None:
    _run([_cpp_tool("conan"), "profile", "detect", "--force"], use_host_pkg_config=True)
    if LOCKFILE_PATH.is_file():
        LOCKFILE_PATH.unlink()
    _run(
        [
            _cpp_tool("conan"),
            "lock",
            "create",
            ".",
            "-s",
            "compiler.cppstd=17",
            "--lockfile-out",
            "conan.lock",
            "--build=missing",
        ],
        use_host_pkg_config=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C++ CI entrypoint for Conan + CMake.")
    parser.add_argument(
        "command",
        choices=("bootstrap", "configure", "build", "build-target", "test", "lock-refresh"),
        help="Action to run.",
    )
    parser.add_argument("target", nargs="?", help="CMake target for the build-target command.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "bootstrap":
        _bootstrap()
    elif args.command == "configure":
        _configure()
    elif args.command == "build":
        _build()
    elif args.command == "build-target":
        if not args.target:
            raise SystemExit("Missing target for build-target")
        _build_target(str(args.target))
    elif args.command == "test":
        _test()
    elif args.command == "lock-refresh":
        _lock_refresh()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
