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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sdk/python'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'launcher'))
from release_wheels import build_wheels
from f8platform.environment_definitions import read_manifest, selected_lock, write_manifest
from f8platform.runtime_sources import read_runtime_sources
from f8pysdk.codec import copy_model
from f8pysdk.extension_spec import ExtensionCatalog
from f8pysdk.service_runtime_tools.inventory.index import IndexedService, read_service_index
from extension_workspace import workspace_root
import msgspec


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
LOCAL_EDITABLE_PATH_PREFIXES = ("launcher/", "extensions/", "sdk/")
# C++ runtime deploy targets are owned by CMake's f8_deploy_all_runtime aggregator.
CPP_DEPLOY_ALL_TARGET = "f8_deploy_all_runtime"
LAUNCHER_RUNTIME_FEATURE = "launcher-runtime"
# Dev service entries may target either the full dev environment or the slim
# web-studio runtime; both collapse onto the single dist runtime environment.
DEV_RUNTIME_ENVIRONMENT_NAMES = ("default", "web-studio-runtime")
DIST_RUNTIME_ENVIRONMENT_NAME = "studio-runtime"
WEB_BUNDLE_SOURCE = REPO_ROOT / "extensions/f8webstudio/build/web-studio"


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
    document = yaml.compose(service_text)
    if not isinstance(document, yaml.MappingNode):
        raise ValueError("service entry must be a YAML mapping")
    for key, launch in document.value:
        if not isinstance(key, yaml.ScalarNode) or key.value != "launch":
            continue
        if not isinstance(launch, yaml.MappingNode):
            raise ValueError("service launch must be a YAML mapping")
        fields = {key.value: value for key, value in launch.value if isinstance(key, yaml.ScalarNode)}
        command = fields.get("command")
        if not isinstance(command, yaml.ScalarNode) or command.value not in {"pixi", "pixi.exe"}:
            continue
        args = fields.get("args")
        if not isinstance(args, yaml.SequenceNode):
            raise ValueError("pixi launch args must be a YAML sequence")
        previous = ""
        for argument in args.value:
            if not isinstance(argument, yaml.ScalarNode):
                raise ValueError("pixi launch args must contain strings")
            value = argument.value
            replacement = None
            if previous in {"-e", "--environment"} and value == source_environment_name:
                replacement = target_environment_name
            elif value == f"--environment={source_environment_name}":
                replacement = f"--environment={target_environment_name}"
            if replacement is not None:
                return (service_text[:argument.start_mark.index] + json.dumps(replacement)
                        + service_text[argument.end_mark.index:])
            previous = value
    return service_text


def _rewrite_dist_service_entries(services_root: Path) -> list[Path]:
    rewritten_paths: list[Path] = []
    for service_entry_path in sorted(services_root.rglob("service*.yml")):
        original_text = service_entry_path.read_text(encoding="utf-8")
        rewritten_text = original_text
        for source_environment_name in DEV_RUNTIME_ENVIRONMENT_NAMES:
            rewritten_text = _rewrite_service_entry_environment_args(
                rewritten_text,
                source_environment_name=source_environment_name,
                target_environment_name=DIST_RUNTIME_ENVIRONMENT_NAME,
            )
        if rewritten_text == original_text:
            continue
        service_entry_path.write_text(rewritten_text, encoding="utf-8")
        rewritten_paths.append(service_entry_path)
    return rewritten_paths


def _validate_dist_service_environments(services_root: Path, runtime_environment_names: list[str]) -> None:
    allowed = frozenset(runtime_environment_names)
    problems: list[str] = []
    for path in sorted(services_root.rglob("service*.yml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"{path}: service entry must be a mapping")
        launch = document.get("launch", {})
        if not isinstance(launch, dict):
            raise ValueError(f"{path}: launch must be a mapping")
        if launch.get("command") not in {"pixi", "pixi.exe"}:
            continue
        args = launch.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError(f"{path}: launch args must be strings")
        environments: list[str] = []
        for index, argument in enumerate(args):
            if argument in {"-e", "--environment"}:
                if index + 1 == len(args):
                    raise ValueError(f"{path}: missing environment argument")
                environments.append(args[index + 1])
            elif argument.startswith("--environment="):
                environments.append(argument.partition("=")[2])
        if not environments:
            problems.append(f"{path}: pixi launch requires an explicit shipped environment")
        for environment in environments:
            if environment not in allowed:
                problems.append(f"{path}: pixi environment '{environment}'")
    if problems:
        raise ValueError(
            "Dist service entries reference environments that are not shipped "
            f"(shipped: {sorted(allowed)}):\n" + "\n".join(problems)
        )


def _copy_dist_services(dist_dir: Path) -> None:
    # Only installed versioned artifacts are distributable; migration backups are local.
    shutil.copytree(workspace_root(REPO_ROOT) / "runtime" / "bundles", dist_dir / "runtime" / "bundles", dirs_exist_ok=True)


def _copy_dist_config(dist_dir: Path) -> Path | None:
    config_root = workspace_root(REPO_ROOT) / "config"
    if not config_root.is_dir():
        return None
    dist_config_root = dist_dir / "config"
    shutil.copytree(config_root, dist_config_root, dirs_exist_ok=True)
    (dist_config_root / 'extension-sources.json').unlink(missing_ok=True)
    index_path = dist_config_root / 'service-index.json'
    index = read_service_index(index_path)
    if index.packageRoot is not None:
        prefix = '${F8_PACKAGE_ROOT}/build/workspace/'
        services = tuple(IndexedService(
            serviceClass=item.serviceClass,
            manifests={name: path.replace(prefix, '${F8_PACKAGE_ROOT}/') for name, path in item.manifests.items()},
            describe=item.describe.replace(prefix, '${F8_PACKAGE_ROOT}/'),
            bundleRoots={name: path.replace(prefix, '${F8_PACKAGE_ROOT}/') for name, path in item.bundleRoots.items()},
        ) for item in index.services)
        index_path.write_bytes(msgspec.json.encode(copy_model(index, update={'services': services, 'packageRoot': None})))
        catalog = msgspec.json.decode((config_root / 'extensions.json').read_bytes(), type=ExtensionCatalog)
        for manifest in catalog.extensions:
            for asset in (*manifest.skills, *manifest.resources):
                relative = asset.path.removeprefix('${F8_PACKAGE_ROOT}/')
                target = dist_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / relative, target)
    models = workspace_root(REPO_ROOT) / 'resources/models'
    if models.is_dir():
        shutil.copytree(models, dist_dir / 'resources/models', dirs_exist_ok=True)
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
    _run(["pixi", "run", "--frozen", "-e", "build-check", "npm", "--prefix", "extensions/f8webstudio/f8studio_web", "ci"])
    _run(["pixi", "run", "--frozen", "-e", "build-check", "studio_web_build"])
    index_path = WEB_BUNDLE_SOURCE / "index.html"
    if not index_path.is_file():
        raise FileNotFoundError(f"Web Studio build did not produce {index_path}")
    return WEB_BUNDLE_SOURCE


def _build_python_wheels(wheels_dir: Path, dependency_to_package_dir: dict[str, str]) -> dict[str, str]:
    build_wheels(
        [REPO_ROOT / directory for directory in dependency_to_package_dir.values()],
        wheels_dir=wheels_dir,
        staging_dir=REPO_ROOT / "build" / "wheel-staging",
        web_bundle=WEB_BUNDLE_SOURCE,
    )
    return {
        name: f"wheels/{_find_wheel_for_distribution(wheels_dir, name).name}"
        for name in dependency_to_package_dir
    }


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
        if section_name.endswith(".tasks"):
            # Only direct module entrypoints are usable without the source checkout.
            lines = section_text.splitlines(keepends=True)
            section_text = lines[0] + "".join(
                line for line in lines[1:]
                if re.match(r'^[-\w]+\s*=\s*"python -m [\w.]+(?: [^"\n]*)?"\s*$', line)
            ) + "\n"
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
        pixi_text, replacement_count = pattern.subn(f'{dependency_name} = {{ path = "{dependency_to_wheel[dependency_name]}" }}', pixi_text, count=1)
        if replacement_count != 1:
            raise ValueError(
                f"Expected exactly one editable path dependency entry for '{dependency_name}' in pixi.toml"
            )
    pixi_text = _remove_dist_pixi_build_preview(pixi_text)
    pixi_text = _filter_dist_environments(pixi_text, runtime_environment_names)
    return _filter_dist_feature_sections(pixi_text, runtime_feature_names)


def _write_dist_lock(dist_dir: Path, runtime_environment_names: list[str]) -> None:
    source_lock_path = REPO_ROOT / "pixi.lock"
    lock = yaml.safe_load(source_lock_path.read_text(encoding="utf-8"))
    # Do not seed Pixi's implicit default environment with the full dev environment.
    lock["environments"] = {
        name: spec for name, spec in lock["environments"].items()
        if name in runtime_environment_names
    }
    (dist_dir / "pixi.lock").write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    manifest = str(dist_dir / "pixi.toml")
    _run(["pixi", "lock", "--manifest-path", manifest, "--no-install"])
    _run(["pixi", "lock", "--manifest-path", manifest, "--check"])


def _build_independent_runtime_manifests(dist_dir: Path) -> list[str]:
    config = workspace_root(REPO_ROOT) / 'config'
    sources = read_runtime_sources(REPO_ROOT, catalog_path=config / 'runtime-environments.json')
    source_path = config / 'extension-sources.json'
    if source_path.is_file():
        checkouts = msgspec.json.decode(source_path.read_bytes(), type=dict[str, str])
        catalog = msgspec.json.decode((config / 'extensions.json').read_bytes(), type=ExtensionCatalog)
        for extension in catalog.extensions:
            name = extension.runtime.environment
            if name is not None:
                checkout = REPO_ROOT / checkouts[extension.extension_id].removeprefix('${F8_PACKAGE_ROOT}/')
                if name in sources and sources[name][0] != checkout:
                    raise ValueError('Integration snapshot runtime aliases collide; publish these extensions independently')
                sources[name] = checkout, None
    manifests = {name: read_manifest(root) for name, (root, _) in sources.items()}
    runtime_names = list(sources)
    base_environment = 'platform-runtime' if 'platform-runtime' in runtime_names else DIST_RUNTIME_ENVIRONMENT_NAME
    if base_environment not in runtime_names:
        raise ValueError('The runtime catalog must declare a bootstrap environment')
    packages: dict[str, str] = {}
    for name in runtime_names:
        root = sources[name][0]
        # Extension workspaces use base dependencies; Studio uses features.
        tables = [manifests[name], *manifests[name].get('feature', {}).values()]
        for table in tables:
            for dependency, specification in table.get('pypi-dependencies', {}).items():
                if not isinstance(specification, dict) or 'path' not in specification:
                    continue
                # The explicit superbuild path uses its canonical SDK checkout.
                # Standalone extension publishers continue to own their SDK input.
                source = (REPO_ROOT / 'sdk/python' if dependency == 'f8pysdk'
                          else (root / specification['path']).resolve())
                if not source.is_relative_to(REPO_ROOT.resolve()) or not (source / 'pyproject.toml').is_file():
                    raise ValueError(f'Invalid local runtime package {dependency}: {source}')
                relative = source.relative_to(REPO_ROOT).as_posix()
                if dependency in packages and packages[dependency] != relative:
                    raise ValueError(f'Runtime workspaces disagree on source package {dependency}')
                packages[dependency] = relative
    wheels = _build_python_wheels(dist_dir / 'wheels', packages)
    for name in runtime_names:
        root = sources[name][0]
        output = dist_dir / 'environment-definitions' / name
        output.mkdir(parents=True, exist_ok=True)
        manifest = manifests[name]
        for table in [manifest, *manifest.get('feature', {}).values()]:
            for dependency, specification in table.get('pypi-dependencies', {}).items():
                if isinstance(specification, dict) and 'path' in specification:
                    table['pypi-dependencies'][dependency] = {'path': '../../' + wheels[dependency]}
        write_manifest(output / 'pixi.toml', manifest)
        seed = selected_lock(root, name)
        if seed is None:
            raise ValueError(f'Official runtime {name} has no structured lock')
        (output / 'pixi.lock').write_text(yaml.safe_dump(seed, sort_keys=False), encoding='utf-8')
        _run(['pixi', 'lock', '--manifest-path', str(output / 'pixi.toml')])
        _run(['pixi', 'lock', '--manifest-path', str(output / 'pixi.toml'), '--check'])
    # Only the offline base launcher uses a top-level manifest. Optional
    # runtimes are installed from the independent references in the catalog.
    base = read_manifest(dist_dir / 'environment-definitions' / base_environment)
    for feature in [base, *base.get('feature', {}).values()]:
        for specification in feature.get('pypi-dependencies', {}).values():
            if isinstance(specification, dict) and 'path' in specification:
                specification['path'] = specification['path'].removeprefix('../../')
    write_manifest(dist_dir / 'pixi.toml', base)
    seed = yaml.safe_load((dist_dir / 'environment-definitions' / base_environment / 'pixi.lock').read_text())

    def retarget_wheels(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'pypi' and isinstance(item, str) and item.startswith('../../wheels/'):
                    value[key] = item.removeprefix('../../')
                else:
                    retarget_wheels(item)
        elif isinstance(value, list):
            for item in value:
                retarget_wheels(item)

    retarget_wheels(seed)
    (dist_dir / 'pixi.lock').write_text(yaml.safe_dump(seed, sort_keys=False), encoding='utf-8')
    _run(['pixi', 'lock', '--manifest-path', str(dist_dir / 'pixi.toml'), '--check'])
    config = dist_dir / 'config'
    config.mkdir(exist_ok=True)
    (config / 'runtime-environments.json').write_text(json.dumps({
        'schemaVersion': 'f8runtimeCatalog/1',
        'runtimes': [{'runtimeId': name, 'manifest': '${F8_PACKAGE_ROOT}/environment-definitions/' + name + '/pixi.toml'}
                     for name in runtime_names],
    }, indent=2) + '\n', encoding='utf-8')
    # Source distribution service entries launch their own shipped workspaces.
    for path in (dist_dir / 'config/services').rglob('*.yml'):
        entry = yaml.safe_load(path.read_text(encoding='utf-8'))
        launch = entry.get('launch', {})
        args = launch.get('args', [])
        if launch.get('command') in {'pixi', 'pixi.exe'} and len(args) == 4 and args[2] in runtime_names:
            launch['workdir'] = '${F8_PACKAGE_ROOT}/environment-definitions/' + args[2]
            path.write_text(yaml.safe_dump(entry, sort_keys=False), encoding='utf-8')
    return runtime_names


def build_runtime_manifest(dist_dir: Path) -> list[str]:
    """Build local runtime wheels and portable, independently locked workspaces."""
    if (workspace_root(REPO_ROOT) / 'config/runtime-environments.json').is_file():
        return _build_independent_runtime_manifests(dist_dir)
    environments = _discover_launcher_runtime_environments()
    features = _discover_environment_feature_names(environment_names=environments)
    packages = _discover_local_editable_package_dirs(allowed_feature_names=set(features))
    wheels = _build_python_wheels(dist_dir / "wheels", packages)
    (dist_dir / "pixi.toml").write_text(
        _render_dist_pixi_toml(wheels, environments, features), encoding="utf-8",
    )
    _write_dist_lock(dist_dir, environments)
    return environments


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a local integration runtime snapshot; official releases live in f8distribution.")
    parser.add_argument('--output', type=Path, default=REPO_ROOT / 'build/integration-runtime')
    parser.add_argument('--build-native', action='store_true', help='Explicitly build native services for a local integration run')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Integration output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.build_native:
        _build_cpp_runtime()
    _stage_web_bundle()
    _copy_dist_config(args.output)
    names = build_runtime_manifest(args.output)
    print('Prepared local integration environments: ' + ', '.join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
