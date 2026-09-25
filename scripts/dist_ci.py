#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PIXI_TOML_PATH = REPO_ROOT / "pixi.toml"
CPP_USER_PRESETS_PATH = REPO_ROOT / "CMakeUserPresets.json"
DEFAULT_CPP_PRESET_PATH = REPO_ROOT / "build" / "Release" / "generators" / "CMakePresets.json"
CPP_PRESET_PATH = DEFAULT_CPP_PRESET_PATH
CPP_PRESET_CANDIDATES = (
    REPO_ROOT / "build" / "generators" / "CMakePresets.json",
    DEFAULT_CPP_PRESET_PATH,
)
CPP_BUILD_PRESET_NAME = "conan-release"
LOCAL_EDITABLE_PATH_PREFIXES = ("packages/", "external/f8unitymods")
# C++ runtime deploy targets are owned by CMake's f8_deploy_all_runtime aggregator.
CPP_DEPLOY_ALL_TARGET = "f8_deploy_all_runtime"
LAUNCHER_ENVIRONMENT_NAME = "launcher"
LAUNCHER_RUNTIME_FEATURE = "launcher-runtime"
DEV_RUNTIME_ENVIRONMENT_NAME = "default"
DIST_RUNTIME_ENVIRONMENT_NAME = "studio-runtime"
WEB_BUNDLE_SOURCE = REPO_ROOT / "packages" / "f8studio_web" / "dist"
WEB_BUNDLE_PACKAGE_DIR = REPO_ROOT / "packages" / "f8studio_server" / "f8studio_server" / "web_dist"


@dataclass(frozen=True)
class LocalEditablePackage:
    package_dir: str
    feature_name: str


def _run(command: list[str]) -> None:
    ccache_tmp_dir = REPO_ROOT / ".ccache-tmp"
    ccache_tmp_dir.mkdir(parents=True, exist_ok=True)

    command_env = os.environ.copy()
    command_env["CCACHE_TEMPDIR"] = str(ccache_tmp_dir)

    subprocess.run(command, check=True, cwd=REPO_ROOT, env=command_env)


def _platform_info() -> tuple[str, str]:
    if os.name == "nt":
        return ("windows-x86_64", "win")
    if sys.platform.startswith("linux"):
        return ("linux-x86_64", "linux")
    raise RuntimeError(f"Unsupported platform for dist packaging: {sys.platform}")


def _repo_relative_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _cpp_preset_path_from_user_presets() -> Path | None:
    if not CPP_USER_PRESETS_PATH.is_file():
        return None

    user_presets = json.loads(CPP_USER_PRESETS_PATH.read_text(encoding="utf-8"))
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


def _resolve_cpp_preset_path() -> Path:
    if CPP_PRESET_PATH != DEFAULT_CPP_PRESET_PATH and CPP_PRESET_PATH.is_file():
        return CPP_PRESET_PATH

    user_preset_path = _cpp_preset_path_from_user_presets()
    if user_preset_path is not None:
        return user_preset_path

    for candidate_path in CPP_PRESET_CANDIDATES:
        if candidate_path.is_file():
            return candidate_path

    checked_paths = ", ".join(_repo_relative_path(candidate_path) for candidate_path in CPP_PRESET_CANDIDATES)
    raise FileNotFoundError(f"Expected Conan-generated preset file is missing. Checked: {checked_paths}")


def _normalize_dist_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _find_wheel_for_distribution(wheels_dir: Path, distribution: str) -> Path:
    normalized = _normalize_dist_name(distribution)
    wheels = sorted(wheels_dir.glob("*.whl"))
    for wheel in wheels:
        wheel_distribution_token = wheel.name.split("-", 1)[0]
        if _normalize_dist_name(wheel_distribution_token) == normalized:
            return wheel
    raise FileNotFoundError(f"Wheel for distribution '{distribution}' was not found in {wheels_dir}")


def _discover_local_editable_packages(
    *,
    pixi_toml_path: Path = PIXI_TOML_PATH,
    repo_root: Path = REPO_ROOT,
    allowed_feature_names: set[str] | None = None,
) -> dict[str, LocalEditablePackage]:
    with pixi_toml_path.open("rb") as pixi_file:
        manifest = tomllib.load(pixi_file)

    feature_table = manifest.get("feature")
    if not isinstance(feature_table, dict):
        raise ValueError(f"Expected [feature] table in {pixi_toml_path}")

    packages: dict[str, LocalEditablePackage] = {}
    for feature_name, feature_spec in feature_table.items():
        if not isinstance(feature_name, str):
            raise ValueError(f"Feature name must be a string in {pixi_toml_path}")
        if not isinstance(feature_spec, dict):
            continue
        if allowed_feature_names is not None and feature_name not in allowed_feature_names:
            continue

        pypi_dependencies = feature_spec.get("pypi-dependencies")
        if not isinstance(pypi_dependencies, dict):
            continue

        for dependency_name, dependency_spec in pypi_dependencies.items():
            if not isinstance(dependency_name, str):
                raise ValueError(f"Dependency name must be a string in feature '{feature_name}'")
            if not isinstance(dependency_spec, dict):
                continue

            dependency_path = dependency_spec.get("path")
            editable_value = dependency_spec.get("editable")
            if not isinstance(dependency_path, str) or editable_value is not True:
                continue
            if not dependency_path.startswith(LOCAL_EDITABLE_PATH_PREFIXES):
                continue
            if dependency_name in packages:
                first_feature = packages[dependency_name].feature_name
                raise ValueError(
                    "Duplicate local editable package dependency "
                    f"'{dependency_name}' in features '{first_feature}' and '{feature_name}'"
                )

            package_dir = repo_root / dependency_path
            if not package_dir.is_dir():
                raise FileNotFoundError(
                    f"Package directory for '{dependency_name}' was not found: {package_dir}"
                )

            pyproject_path = package_dir / "pyproject.toml"
            if not pyproject_path.is_file():
                raise FileNotFoundError(
                    f"pyproject.toml for '{dependency_name}' was not found: {pyproject_path}"
                )

            packages[dependency_name] = LocalEditablePackage(
                package_dir=dependency_path,
                feature_name=feature_name,
            )

    return packages


def _discover_local_editable_package_dirs(
    *,
    pixi_toml_path: Path = PIXI_TOML_PATH,
    repo_root: Path = REPO_ROOT,
    allowed_feature_names: set[str] | None = None,
) -> dict[str, str]:
    packages = _discover_local_editable_packages(
        pixi_toml_path=pixi_toml_path,
        repo_root=repo_root,
        allowed_feature_names=allowed_feature_names,
    )
    return {
        dependency_name: package.package_dir
        for dependency_name, package in packages.items()
    }


def _discover_environment_feature_names(
    *,
    environment_names: list[str],
    pixi_toml_path: Path = PIXI_TOML_PATH,
) -> list[str]:
    with pixi_toml_path.open("rb") as pixi_file:
        manifest = tomllib.load(pixi_file)

    feature_table = manifest.get("feature")
    if not isinstance(feature_table, dict):
        raise ValueError(f"Expected [feature] table in {pixi_toml_path}")

    environments_table = manifest.get("environments")
    if not isinstance(environments_table, dict):
        raise ValueError(f"Expected [environments] table in {pixi_toml_path}")

    ordered_feature_names: list[str] = []
    seen_feature_names: set[str] = set()
    for environment_name in environment_names:
        environment_spec = environments_table.get(environment_name)
        if not isinstance(environment_spec, dict):
            raise ValueError(f"Environment '{environment_name}' was not found in {pixi_toml_path}")

        features = environment_spec.get("features")
        if not isinstance(features, list):
            raise ValueError(
                f"Environment '{environment_name}' must define a list of features in {pixi_toml_path}"
            )

        for feature_name in features:
            if not isinstance(feature_name, str):
                raise ValueError(
                    f"Environment '{environment_name}' contains a non-string feature in {pixi_toml_path}"
                )
            if feature_name not in feature_table:
                raise ValueError(
                    f"Environment '{environment_name}' references undefined feature '{feature_name}' "
                    f"in {pixi_toml_path}"
                )
            if feature_name in seen_feature_names:
                continue
            seen_feature_names.add(feature_name)
            ordered_feature_names.append(feature_name)

    return ordered_feature_names


def _discover_launcher_runtime_environments(*, pixi_toml_path: Path = PIXI_TOML_PATH) -> list[str]:
    with pixi_toml_path.open("rb") as pixi_file:
        manifest = tomllib.load(pixi_file)

    environments_table = manifest.get("environments")
    if not isinstance(environments_table, dict):
        raise ValueError(f"Expected [environments] table in {pixi_toml_path}")

    runtime_environment_names: list[str] = []
    for environment_name, environment_spec in environments_table.items():
        if not isinstance(environment_name, str):
            raise ValueError(f"Environment name must be a string in {pixi_toml_path}")
        if not isinstance(environment_spec, dict):
            continue
        features = environment_spec.get("features")
        if not isinstance(features, list):
            continue
        if LAUNCHER_RUNTIME_FEATURE in features:
            runtime_environment_names.append(environment_name)

    if not runtime_environment_names:
        raise ValueError(
            f"No runtime environments were marked with feature '{LAUNCHER_RUNTIME_FEATURE}' in {pixi_toml_path}"
        )

    return runtime_environment_names


def _split_manifest_sections(pixi_text: str) -> list[tuple[str | None, str]]:
    section_matches = list(re.finditer(r"(?m)^\[([^\[\]\n]+)\]\s*$", pixi_text))
    if not section_matches:
        return [(None, pixi_text)]

    sections: list[tuple[str | None, str]] = []
    if section_matches[0].start() > 0:
        sections.append((None, pixi_text[: section_matches[0].start()]))

    for index, match in enumerate(section_matches):
        section_name = match.group(1)
        section_end = section_matches[index + 1].start() if index + 1 < len(section_matches) else len(pixi_text)
        sections.append((section_name, pixi_text[match.start() : section_end]))

    return sections


def _feature_name_from_section(section_name: str) -> str | None:
    if not section_name.startswith("feature."):
        return None
    parts = section_name.split(".")
    if len(parts) < 2 or parts[1] == "":
        return None
    return parts[1]


def _rewrite_service_entry_environment_args(
    service_text: str,
    *,
    source_environment_name: str,
    target_environment_name: str,
) -> str:
    pattern = re.compile(
        r'(\bargs:\s*\[\s*["\']run["\']\s*,\s*["\']-e["\']\s*,\s*["\'])'
        + re.escape(source_environment_name)
        + r'(["\'])'
    )
    return pattern.sub(r"\1" + target_environment_name + r"\2", service_text)


def _rewrite_dist_service_entries(services_root: Path) -> list[Path]:
    rewritten_paths: list[Path] = []
    for service_entry_path in sorted(services_root.rglob("service*.yml")):
        original_text = service_entry_path.read_text(encoding="utf-8")
        rewritten_text = _rewrite_service_entry_environment_args(
            original_text,
            source_environment_name=DEV_RUNTIME_ENVIRONMENT_NAME,
            target_environment_name=DIST_RUNTIME_ENVIRONMENT_NAME,
        )
        if rewritten_text == original_text:
            continue
        service_entry_path.write_text(rewritten_text, encoding="utf-8")
        rewritten_paths.append(service_entry_path)
    return rewritten_paths


def _copy_dist_config(dist_dir: Path) -> Path | None:
    config_root = REPO_ROOT / "config"
    if not config_root.is_dir():
        return None
    dist_config_root = dist_dir / "config"
    shutil.copytree(config_root, dist_config_root, dirs_exist_ok=True)
    return dist_config_root


def _bundle_unitymods_assets(dist_dir: Path, *, build_assets: bool = True) -> Path | None:
    if os.name != "nt":
        return None
    output_dir = dist_dir / "unitymods"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "unitymods_ci.py"),
        "bundle",
        "--output",
        str(output_dir),
    ]
    if not build_assets:
        command.append("--skip-build")
    _run(command)
    return output_dir


def _stage_web_bundle() -> Path:
    _run(["pixi", "run", "--frozen", "-e", "web-studio", "npm", "--prefix", "packages/f8studio_web", "ci"])
    _run(["pixi", "run", "--frozen", "-e", "web-studio", "studio_web_build"])
    index_path = WEB_BUNDLE_SOURCE / "index.html"
    if not index_path.is_file():
        raise FileNotFoundError(f"Web Studio build did not produce {index_path}")
    WEB_BUNDLE_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    for child in WEB_BUNDLE_PACKAGE_DIR.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    shutil.copytree(WEB_BUNDLE_SOURCE, WEB_BUNDLE_PACKAGE_DIR, dirs_exist_ok=True)
    return WEB_BUNDLE_PACKAGE_DIR


def _build_python_wheels(wheels_dir: Path, dependency_to_package_dir: dict[str, str]) -> dict[str, str]:
    if wheels_dir.exists():
        shutil.rmtree(wheels_dir)
    wheels_dir.mkdir(parents=True, exist_ok=True)

    for package_dir in dependency_to_package_dir.values():
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
    for dependency_name in dependency_to_package_dir:
        wheel_path = _find_wheel_for_distribution(wheels_dir, dependency_name)
        dependency_to_wheel[dependency_name] = f"wheels/{wheel_path.name}"
    return dependency_to_wheel


def _filter_dist_environments(pixi_text: str, runtime_environment_names: list[str]) -> str:
    start_match = re.search(r"(?m)^\[environments\]\s*$", pixi_text)
    if start_match is None:
        raise ValueError("Expected [environments] table in pixi.toml")

    next_section_match = re.search(r"(?m)^\[[^\[\]\n].*\]\s*$", pixi_text[start_match.end() :])
    if next_section_match is None:
        section_end = len(pixi_text)
    else:
        section_end = start_match.end() + next_section_match.start()

    section_text = pixi_text[start_match.end() : section_end]
    runtime_environment_set = set(runtime_environment_names)
    kept_environment_names: list[str] = []
    filtered_lines: list[str] = []

    for line in section_text.splitlines(keepends=True):
        stripped_line = line.strip()
        if stripped_line == "" or stripped_line.startswith("#"):
            filtered_lines.append(line)
            continue

        env_match = re.match(r"^([A-Za-z0-9_.-]+)\s*=", stripped_line)
        if env_match is None:
            raise ValueError(f"Unsupported [environments] entry format: {stripped_line}")
        environment_name = env_match.group(1)

        if environment_name in runtime_environment_set:
            filtered_lines.append(line)
            kept_environment_names.append(environment_name)

    missing_runtime_environments = [
        environment_name
        for environment_name in runtime_environment_names
        if environment_name not in kept_environment_names
    ]
    if missing_runtime_environments:
        raise ValueError(
            "Failed to retain runtime environments in dist manifest: "
            + ", ".join(missing_runtime_environments)
        )

    filtered_section_text = "".join(filtered_lines)
    return pixi_text[: start_match.end()] + filtered_section_text + pixi_text[section_end:]


def _filter_dist_feature_sections(pixi_text: str, runtime_feature_names: list[str]) -> str:
    runtime_feature_name_set = set(runtime_feature_names)
    filtered_sections: list[str] = []
    retained_feature_names: set[str] = set()

    for section_name, section_text in _split_manifest_sections(pixi_text):
        if section_name is None:
            filtered_sections.append(section_text)
            continue

        feature_name = _feature_name_from_section(section_name)
        if feature_name is None:
            filtered_sections.append(section_text)
            continue
        if feature_name not in runtime_feature_name_set:
            continue

        retained_feature_names.add(feature_name)
        filtered_sections.append(section_text)

    missing_runtime_features = [
        feature_name for feature_name in runtime_feature_names if feature_name not in retained_feature_names
    ]
    if missing_runtime_features:
        raise ValueError(
            "Failed to retain runtime features in dist manifest: " + ", ".join(missing_runtime_features)
        )

    return "".join(filtered_sections)


def _remove_dist_pixi_build_preview(pixi_text: str) -> str:
    pattern = re.compile(
        r'(?m)^preview\s*=\s*\[\s*["\']pixi-build["\']\s*\]\s*(?:\r?\n|$)'
    )
    return pattern.sub("", pixi_text, count=1)


def _render_dist_pixi_toml(
    dependency_to_wheel: dict[str, str],
    runtime_environment_names: list[str],
    runtime_feature_names: list[str],
) -> str:
    pixi_text = PIXI_TOML_PATH.read_text(encoding="utf-8")
    for dependency_name in dependency_to_wheel:
        pattern = re.compile(
            rf'^{re.escape(dependency_name)}\s*=\s*\{{\s*path\s*=\s*"[^"]+"\s*,\s*editable\s*=\s*true\s*\}}\s*$',
            flags=re.MULTILINE,
        )
        pixi_text, replacement_count = pattern.subn("", pixi_text, count=1)
        if replacement_count != 1:
            raise ValueError(
                f"Expected exactly one editable path dependency entry for '{dependency_name}' in pixi.toml"
            )
    pixi_text = _remove_dist_pixi_build_preview(pixi_text)
    pixi_text = _filter_dist_environments(pixi_text, runtime_environment_names)
    return _filter_dist_feature_sections(pixi_text, runtime_feature_names)


def _launcher_binary_name() -> str:
    if os.name == "nt":
        return "f8studio.exe"
    return "f8studio"


def _env_install_script_name() -> str:
    if os.name == "nt":
        return "install_env.bat"
    return "install_env.sh"


def _runtime_environment_wheels(
    *,
    runtime_environment_names: list[str],
    packages: dict[str, LocalEditablePackage],
    dependency_to_wheel: dict[str, str],
    pixi_toml_path: Path = PIXI_TOML_PATH,
) -> dict[str, list[str]]:
    with pixi_toml_path.open("rb") as pixi_file:
        manifest = tomllib.load(pixi_file)
    environments_table = manifest.get("environments")
    if not isinstance(environments_table, dict):
        raise ValueError(f"Expected [environments] table in {pixi_toml_path}")

    environment_to_wheels: dict[str, list[str]] = {}
    installed_dependencies: set[str] = set()
    for environment_name in runtime_environment_names:
        environment_spec = environments_table.get(environment_name)
        if not isinstance(environment_spec, dict):
            raise ValueError(f"Environment '{environment_name}' was not found in {pixi_toml_path}")
        feature_names = environment_spec.get("features")
        if not isinstance(feature_names, list) or not all(
            isinstance(feature_name, str) for feature_name in feature_names
        ):
            raise ValueError(
                f"Environment '{environment_name}' must define string features in {pixi_toml_path}"
            )

        feature_name_set = set(feature_names)
        wheel_paths: list[str] = []
        for dependency_name, package in packages.items():
            if package.feature_name not in feature_name_set:
                continue
            wheel_path = dependency_to_wheel.get(dependency_name)
            if wheel_path is None:
                raise ValueError(f"Wheel mapping is missing dependency '{dependency_name}'")
            wheel_paths.append(wheel_path)
            installed_dependencies.add(dependency_name)
        environment_to_wheels[environment_name] = wheel_paths

    missing_dependencies = sorted(set(packages) - installed_dependencies)
    if missing_dependencies:
        raise ValueError(
            "Local packages are not assigned to a runtime environment: "
            + ", ".join(missing_dependencies)
        )
    return environment_to_wheels


def _env_install_script_text(
    runtime_environment_names: list[str],
    environment_to_wheels: dict[str, list[str]] | None = None,
) -> str:
    install_command_parts = ["pixi", "install"]
    for environment_name in runtime_environment_names:
        install_command_parts.extend(["-e", environment_name])
    install_command = " ".join(install_command_parts)
    wheel_install_commands: list[str] = []
    for environment_name in runtime_environment_names:
        wheel_paths = (environment_to_wheels or {}).get(environment_name, [])
        if not wheel_paths:
            continue
        quoted_wheel_paths = " ".join(f'"{wheel_path}"' for wheel_path in wheel_paths)
        wheel_install_commands.append(
            "pixi run -e "
            + environment_name
            + " python -m pip install --no-deps --no-index "
            + quoted_wheel_paths
        )

    if os.name == "nt":
        commands = [install_command, *wheel_install_commands]
        return (
            "@echo off\r\n"
            "setlocal\r\n"
            "cd /d \"%~dp0\"\r\n"
            "if errorlevel 1 exit /b %errorlevel%\r\n"
            + "".join(command + "\r\nif errorlevel 1 exit /b %errorlevel%\r\n" for command in commands)
        )
    script_text = (
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        "SCRIPT_DIR=$(CDPATH= cd -- \"$(dirname -- \"$0\")\" && pwd)\n"
        "cd \"$SCRIPT_DIR\"\n"
        f"{install_command}\n"
    )
    if wheel_install_commands:
        script_text += "\n".join(wheel_install_commands) + "\n"
    return script_text


def _write_env_install_script(
    dist_dir: Path,
    runtime_environment_names: list[str],
    environment_to_wheels: dict[str, list[str]],
) -> Path:
    script_path = dist_dir / _env_install_script_name()
    script_path.write_text(
        _env_install_script_text(runtime_environment_names, environment_to_wheels),
        encoding="utf-8",
    )
    if os.name != "nt":
        script_path.chmod(0o755)
    return script_path


def _is_running_inside_pixi_environment(environment_name: str) -> bool:
    if os.environ.get("PIXI_ENVIRONMENT_NAME") != environment_name:
        return False
    project_root = os.environ.get("PIXI_PROJECT_ROOT")
    if project_root is None:
        return False
    return Path(project_root).resolve() == REPO_ROOT.resolve()


def _bundle_studio_launcher(dist_dir: Path) -> None:
    if _is_running_inside_pixi_environment(LAUNCHER_ENVIRONMENT_NAME):
        print("Building studio launcher in current launcher Pixi environment")
        _run(["python", "scripts/build_studio_launcher.py"])
    else:
        print("Building studio launcher via isolated launcher Pixi environment")
        _run(["pixi", "run", "--frozen", "-e", LAUNCHER_ENVIRONMENT_NAME, "build_studio_launcher"])

    launcher_path = REPO_ROOT / "build" / "dist" / _launcher_binary_name()
    if not launcher_path.is_file():
        raise FileNotFoundError(f"Expected launcher binary was not produced: {launcher_path}")

    bundled_launcher_path = dist_dir / launcher_path.name
    shutil.copy2(launcher_path, bundled_launcher_path)
    if os.name != "nt":
        bundled_launcher_path.chmod(0o755)


def _build_cpp_runtime() -> None:
    def _require_conan_release_build_preset() -> None:
        cpp_preset_path = _resolve_cpp_preset_path()
        presets = json.loads(cpp_preset_path.read_text(encoding="utf-8"))
        build_presets = presets.get("buildPresets", [])
        build_preset_names = {
            preset.get("name") for preset in build_presets if isinstance(preset, dict) and isinstance(preset.get("name"), str)
        }
        if CPP_BUILD_PRESET_NAME not in build_preset_names:
            raise FileNotFoundError(
                "Generated Conan presets must define the canonical release build preset. "
                f"buildPresets={sorted(build_preset_names)}"
            )

    try:
        _resolve_cpp_preset_path()
    except FileNotFoundError:
        _run(["pixi", "run", "--frozen", "-e", "cpp", "cpp_bootstrap"])

    _require_conan_release_build_preset()
    _run(["pixi", "run", "--frozen", "-e", "cpp", "cpp_configure_release"])
    deploy_build_command = [
        "pixi",
        "run",
        "--frozen",
        "-e",
        "cpp",
        "cmake",
        "--build",
        "--preset",
        CPP_BUILD_PRESET_NAME,
        "--target",
        CPP_DEPLOY_ALL_TARGET,
        "--parallel",
    ]
    try:
        _run(deploy_build_command)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to build C++ deploy target '{CPP_DEPLOY_ALL_TARGET}'. "
            "Ensure each service is registered via f8_deploy_service_runtime(...) in CMake."
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build full runtime distribution bundle (CI packaging path).")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Also emit compressed archive in build/dist (zip on Windows, tar.gz on Linux).",
    )
    parser.add_argument(
        "--reuse-unitymods-assets",
        action="store_true",
        help="Bundle previously built f8unitymods dist assets instead of rebuilding them.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    _build_cpp_runtime()
    _stage_web_bundle()

    platform_tag, platform_dir = _platform_info()
    dist_base_dir = REPO_ROOT / "build" / "dist"
    dist_name = f"f8studio-{platform_tag}"
    dist_dir = dist_base_dir / dist_name

    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True, exist_ok=True)

    (dist_dir / "services").mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_ROOT / "services", dist_dir / "services", dirs_exist_ok=True)
    _rewrite_dist_service_entries(dist_dir / "services")
    _copy_dist_config(dist_dir)
    _bundle_unitymods_assets(
        dist_dir,
        build_assets=not bool(args.reuse_unitymods_assets),
    )

    wheels_dir = dist_dir / "wheels"
    runtime_environment_names = _discover_launcher_runtime_environments()
    runtime_feature_names = _discover_environment_feature_names(environment_names=runtime_environment_names)
    local_packages = _discover_local_editable_packages(
        allowed_feature_names=set(runtime_feature_names),
    )
    dependency_to_package_dir = {
        dependency_name: package.package_dir
        for dependency_name, package in local_packages.items()
    }
    dependency_to_wheel = _build_python_wheels(wheels_dir, dependency_to_package_dir)
    environment_to_wheels = _runtime_environment_wheels(
        runtime_environment_names=runtime_environment_names,
        packages=local_packages,
        dependency_to_wheel=dependency_to_wheel,
    )
    dist_pixi_text = _render_dist_pixi_toml(
        dependency_to_wheel,
        runtime_environment_names,
        runtime_feature_names,
    )
    dist_manifest_path = dist_dir / "pixi.toml"
    dist_manifest_path.write_text(dist_pixi_text, encoding="utf-8")

    source_lock_path = REPO_ROOT / "pixi.lock"
    if not source_lock_path.is_file():
        raise FileNotFoundError(f"Root Pixi lockfile was not found: {source_lock_path}")
    shutil.copy2(source_lock_path, dist_dir / "pixi.lock")
    _run(["pixi", "lock", "--manifest-path", str(dist_manifest_path), "--no-install"])
    _bundle_studio_launcher(dist_dir)
    env_install_script_path = _write_env_install_script(
        dist_dir,
        runtime_environment_names,
        environment_to_wheels,
    )

    readme_text = (
        "# f8 Runtime Dist\n\n"
        "This bundle contains:\n"
        "- pixi.toml + pixi.lock\n"
        "- services/**\n"
        "- Python wheels for local non-editable install\n\n"
        "- Web Studio production assets embedded in the f8studio-server wheel\n\n"
        "- Windows Unity modding installer/exporter assets under unitymods/\n\n"
        "- Studio launcher executable at dist root\n\n"
        "Bootstrap:\n"
        "1. Install Pixi.\n"
        f"2. Run `{env_install_script_path.name}` in dist root.\n"
        "3. Start Studio via launcher (`./f8studio` on Linux/macOS, `f8studio.exe` on Windows),\n"
        "   or run your service command via `pixi run ...`.\n\n"
        f"Platform runtime binaries are under `services/**/{platform_dir}`.\n"
    )
    (dist_dir / "README.md").write_text(readme_text, encoding="utf-8")

    if args.archive:
        archive_format = "zip" if os.name == "nt" else "gztar"
        archive_path = shutil.make_archive(
            base_name=str(dist_base_dir / dist_name),
            format=archive_format,
            root_dir=dist_base_dir,
            base_dir=dist_name,
        )
        print(f"dist archive: {archive_path}")
    print(f"dist directory: {dist_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
