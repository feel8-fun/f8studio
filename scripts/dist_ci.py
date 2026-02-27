#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PIXI_TOML_PATH = REPO_ROOT / "pixi.toml"
CPP_PRESET_PATH = REPO_ROOT / "build" / "Release" / "generators" / "CMakePresets.json"
CPP_PRESET_FALLBACK_PATH = REPO_ROOT / "build" / "generators" / "CMakePresets.json"

PYTHON_PACKAGE_DIRS: dict[str, str] = {
    "f8pysdk": "packages/f8pysdk",
    "f8pyengine": "packages/f8pyengine",
    "f8pyscript": "packages/f8pyscript",
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


def _platform_info() -> tuple[str, str]:
    if os.name == "nt":
        return ("windows-x86_64", "win")
    if sys.platform.startswith("linux"):
        return ("linux-x86_64", "linux")
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


def _launcher_binary_name() -> str:
    if os.name == "nt":
        return "f8studio.exe"
    return "f8studio"


def _is_running_inside_ci_env() -> bool:
    if os.environ.get("PIXI_ENVIRONMENT_NAME") != "ci":
        return False
    project_root = os.environ.get("PIXI_PROJECT_ROOT")
    if project_root is None:
        return False
    return Path(project_root).resolve() == REPO_ROOT.resolve()


def _bundle_studio_launcher(dist_dir: Path) -> None:
    if _is_running_inside_ci_env():
        print("Building studio launcher in current ci Pixi environment")
        _run(["python", "scripts/build_studio_launcher.py"])
    else:
        print("Building studio launcher via pixi ci environment")
        _run(["pixi", "run", "--frozen", "-e", "ci", "build_studio_launcher"])

    launcher_path = REPO_ROOT / "build" / "dist" / _launcher_binary_name()
    if not launcher_path.is_file():
        raise FileNotFoundError(f"Expected launcher binary was not produced: {launcher_path}")

    bundled_launcher_path = dist_dir / launcher_path.name
    shutil.copy2(launcher_path, bundled_launcher_path)
    if os.name != "nt":
        bundled_launcher_path.chmod(0o755)


def _build_cpp_runtime() -> None:
    def _resolve_conan_build_preset_name() -> str:
        for presets_path in (CPP_PRESET_PATH, CPP_PRESET_FALLBACK_PATH):
            if not presets_path.is_file():
                continue
            presets = json.loads(presets_path.read_text(encoding="utf-8"))
            build_presets = presets.get("buildPresets", [])
            configure_presets = presets.get("configurePresets", [])
            build_preset_names = {
                preset.get("name") for preset in build_presets if isinstance(preset.get("name"), str)
            }
            configure_preset_names = {
                preset.get("name") for preset in configure_presets if isinstance(preset.get("name"), str)
            }

            if "conan-release" in build_preset_names:
                return "conan-release"
            if "conan-default" in build_preset_names:
                return "conan-default"
            # Backward compatibility for generators that only emit configure preset names.
            if "conan-release" in configure_preset_names:
                return "conan-release"
            if "conan-default" in configure_preset_names:
                return "conan-default"
        raise FileNotFoundError("Unable to resolve Conan CMake build preset name from generated CMakePresets.json")

    if not CPP_PRESET_PATH.is_file() and not CPP_PRESET_FALLBACK_PATH.is_file():
        _run(["pixi", "run", "--frozen", "-e", "cpp", "cpp_bootstrap"])

    preset_name = _resolve_conan_build_preset_name()
    _run(["pixi", "run", "--frozen", "-e", "cpp", "cpp_configure_release"])
    _run(
        [
            "pixi",
            "run",
            "--frozen",
            "-e",
            "cpp",
            "cmake",
            "--build",
            "--preset",
            preset_name,
            "--target",
            *CPP_DEPLOY_TARGETS,
            "--parallel",
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build full runtime distribution bundle (CI packaging path).")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Also emit compressed archive in build/dist (zip on Windows, tar.gz on Linux).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    _build_cpp_runtime()

    platform_tag, platform_dir = _platform_info()
    dist_base_dir = REPO_ROOT / "build" / "dist"
    dist_name = f"f8studio-{platform_tag}"
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
    _bundle_studio_launcher(dist_dir)

    readme_text = (
        "# f8 Runtime Dist\n\n"
        "This bundle contains:\n"
        "- pixi.toml + pixi.lock\n"
        "- services/**\n"
        "- Python wheels for local non-editable install\n\n"
        "- Studio launcher executable at dist root\n\n"
        "Bootstrap:\n"
        "1. Install Pixi.\n"
        "2. `pixi install --frozen -e default`\n"
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
