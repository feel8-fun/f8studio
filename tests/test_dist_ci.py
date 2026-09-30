from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
from unittest import mock
from pathlib import Path


def _load_dist_ci_module() -> object:
    script_path = Path("scripts/dist_ci.py").resolve()
    spec = importlib.util.spec_from_file_location("dist_ci", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load dist_ci module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DistCiDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_dist_ci_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_pyproject(self, relative_package_dir: str, package_name: str) -> None:
        package_dir = self.root / relative_package_dir
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "pyproject.toml").write_text(
            "[project]\n"
            f'name = "{package_name}"\n'
            'version = "0.1.0"\n',
            encoding="utf-8",
        )

    def test_distribution_only_copies_installed_bundles(self) -> None:
        paths = (
            "bundles/f8.cppengine/0.0.1/linux/engine",
            "bundles/f8.cppengine/0.0.1/describe.json",
            "migration-backup/legacy-service-tree/imgui.ini",
        )
        for relative in paths:
            path = self.root / "runtime" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        dist = self.root / "dist"
        with mock.patch.object(self.module, "REPO_ROOT", self.root):
            self.module._copy_dist_services(dist)
        actual = {path.relative_to(dist / "runtime").as_posix()
                  for path in (dist / "runtime").rglob("*") if path.is_file()}
        self.assertEqual(actual, set(paths[:2]))

    def test_discover_local_editable_packages_ignores_non_packages_or_non_editable(self) -> None:
        self._write_pyproject("packages/pkg_a", "pkg-a")
        self._write_pyproject("packages/pkg_c", "pkg-c")

        pixi_toml_path = self.root / "pixi.toml"
        pixi_toml_path.write_text(
            "[feature.alpha.pypi-dependencies]\n"
            'pkg-a = { path = "packages/pkg_a", editable = true }\n'
            'pkg-b = { path = "packages/pkg_b", editable = false }\n'
            'outside = { path = "../other/pkg", editable = true }\n'
            'requests = ">=2"\n'
            "\n"
            "[feature.beta.pypi-dependencies]\n"
            'pkg-c = { path = "packages/pkg_c", editable = true }\n',
            encoding="utf-8",
        )

        dependencies = self.module._discover_local_editable_package_dirs(
            pixi_toml_path=pixi_toml_path,
            repo_root=self.root,
        )

        self.assertEqual(
            dependencies,
            {
                "pkg-a": "packages/pkg_a",
                "pkg-c": "packages/pkg_c",
            },
        )

    def test_discover_local_editable_packages_can_filter_to_runtime_features(self) -> None:
        self._write_pyproject("packages/pkg_runtime", "pkg-runtime")
        self._write_pyproject("packages/pkg_dev_only", "pkg-dev-only")

        pixi_toml_path = self.root / "pixi.toml"
        pixi_toml_path.write_text(
            "[feature.runtime.pypi-dependencies]\n"
            'pkg-runtime = { path = "packages/pkg_runtime", editable = true }\n'
            "\n"
            "[feature.dev-only.pypi-dependencies]\n"
            'pkg-dev-only = { path = "packages/pkg_dev_only", editable = true }\n',
            encoding="utf-8",
        )

        dependencies = self.module._discover_local_editable_package_dirs(
            pixi_toml_path=pixi_toml_path,
            repo_root=self.root,
            allowed_feature_names={"runtime"},
        )

        self.assertEqual(
            dependencies,
            {
                "pkg-runtime": "packages/pkg_runtime",
            },
        )

    def test_discover_local_editable_packages_fails_on_duplicate_dependency_name(self) -> None:
        self._write_pyproject("packages/pkg_a", "pkg-a")

        pixi_toml_path = self.root / "pixi.toml"
        pixi_toml_path.write_text(
            "[feature.alpha.pypi-dependencies]\n"
            'pkg-a = { path = "packages/pkg_a", editable = true }\n'
            "\n"
            "[feature.beta.pypi-dependencies]\n"
            'pkg-a = { path = "packages/pkg_a", editable = true }\n',
            encoding="utf-8",
        )

        with self.assertRaises(ValueError) as ctx:
            self.module._discover_local_editable_package_dirs(
                pixi_toml_path=pixi_toml_path,
                repo_root=self.root,
            )

        self.assertIn("Duplicate local editable package dependency 'pkg-a'", str(ctx.exception))

    def test_discover_local_editable_packages_fails_when_directory_missing(self) -> None:
        pixi_toml_path = self.root / "pixi.toml"
        pixi_toml_path.write_text(
            "[feature.alpha.pypi-dependencies]\n"
            'pkg-missing = { path = "packages/pkg_missing", editable = true }\n',
            encoding="utf-8",
        )

        with self.assertRaises(FileNotFoundError) as ctx:
            self.module._discover_local_editable_package_dirs(
                pixi_toml_path=pixi_toml_path,
                repo_root=self.root,
            )

        self.assertIn("Package directory for 'pkg-missing' was not found", str(ctx.exception))

    def test_discover_local_editable_packages_fails_when_pyproject_missing(self) -> None:
        package_dir = self.root / "packages" / "pkg_no_pyproject"
        package_dir.mkdir(parents=True, exist_ok=True)

        pixi_toml_path = self.root / "pixi.toml"
        pixi_toml_path.write_text(
            "[feature.alpha.pypi-dependencies]\n"
            'pkg-no-pyproject = { path = "packages/pkg_no_pyproject", editable = true }\n',
            encoding="utf-8",
        )

        with self.assertRaises(FileNotFoundError) as ctx:
            self.module._discover_local_editable_package_dirs(
                pixi_toml_path=pixi_toml_path,
                repo_root=self.root,
            )

        self.assertIn("pyproject.toml for 'pkg-no-pyproject' was not found", str(ctx.exception))

    def test_find_wheel_for_distribution_normalizes_hyphen_and_underscore(self) -> None:
        wheels_dir = self.root / "wheels"
        wheels_dir.mkdir(parents=True, exist_ok=True)
        wheel_path = wheels_dir / "f8studio_server-0.1.0-py3-none-any.whl"
        wheel_path.write_text("wheel-content", encoding="utf-8")

        discovered_wheel = self.module._find_wheel_for_distribution(
            wheels_dir,
            "f8studio-server",
        )

        self.assertEqual(discovered_wheel, wheel_path)

    def test_root_manifest_discovers_and_rewrites_unitymods_package(self) -> None:
        runtime_environment_names = self.module._discover_launcher_runtime_environments()
        runtime_feature_names = self.module._discover_environment_feature_names(
            environment_names=runtime_environment_names
        )
        dependencies = self.module._discover_local_editable_package_dirs(
            allowed_feature_names=set(runtime_feature_names)
        )

        self.assertEqual(dependencies["f8unitymods-setup"], "external/f8unitymods")

        rendered = self.module._render_dist_pixi_toml(
            {"f8unitymods-setup": "wheels/f8unitymods_setup-0.2.0-py3-none-any.whl"},
            runtime_environment_names,
            runtime_feature_names,
        )
        self.assertNotIn(
            'f8unitymods-setup = { path = "external/f8unitymods", editable = true }',
            rendered,
        )
        self.assertEqual(
            tomllib.loads(rendered)["feature"]["web-studio"]["pypi-dependencies"]["f8unitymods-setup"],
            {"path": "wheels/f8unitymods_setup-0.2.0-py3-none-any.whl"},
        )
        self.assertNotIn("scripts/", rendered)

    def test_ci_environment_reuses_the_runtime_python_feature(self) -> None:
        with Path("pixi.toml").open("rb") as pixi_file:
            manifest = tomllib.load(pixi_file)

        ci_environment = manifest["environments"]["ci"]
        ci_dependencies = manifest["feature"]["ci"]["dependencies"]

        self.assertIn("python", ci_environment["features"])
        self.assertNotIn("python", ci_dependencies)
        self.assertNotIn("pip", ci_dependencies)

    def test_discover_launcher_runtime_environments_from_marker_feature(self) -> None:
        pixi_toml_path = self.root / "pixi.toml"
        pixi_toml_path.write_text(
            "[environments]\n"
            'studio-runtime = { features = ["python", "web-studio", "launcher-runtime"] }\n'
            'onnx = { features = ["python", "onnx", "launcher-runtime"] }\n'
            'ci = { features = ["ci"] }\n',
            encoding="utf-8",
        )

        environments = self.module._discover_launcher_runtime_environments(pixi_toml_path=pixi_toml_path)

        self.assertEqual(environments, ["studio-runtime", "onnx"])

    def test_discover_environment_feature_names_collects_runtime_closure(self) -> None:
        pixi_toml_path = self.root / "pixi.toml"
        pixi_toml_path.write_text(
            "[feature.python]\n"
            "[feature.sdk]\n"
            "[feature.web-studio]\n"
            "[feature.onnx]\n"
            "[feature.launcher-runtime]\n"
            "\n"
            "[environments]\n"
            'studio-runtime = { features = ["python", "sdk", "web-studio", "launcher-runtime"] }\n'
            'onnx = { features = ["python", "sdk", "onnx", "launcher-runtime"] }\n',
            encoding="utf-8",
        )

        feature_names = self.module._discover_environment_feature_names(
            pixi_toml_path=pixi_toml_path,
            environment_names=["studio-runtime", "onnx"],
        )

        self.assertEqual(feature_names, ["python", "sdk", "web-studio", "launcher-runtime", "onnx"])

    def test_discover_launcher_runtime_environments_fails_without_marker(self) -> None:
        pixi_toml_path = self.root / "pixi.toml"
        pixi_toml_path.write_text(
            "[environments]\n"
            'ci = { features = ["ci"] }\n',
            encoding="utf-8",
        )

        with self.assertRaises(ValueError) as ctx:
            self.module._discover_launcher_runtime_environments(pixi_toml_path=pixi_toml_path)

        self.assertIn("launcher-runtime", str(ctx.exception))

    def test_env_install_script_text_installs_only_runtime_environments(self) -> None:
        script_text = self.module._env_install_script_text(["studio-runtime", "onnx"])

        self.assertIn("pixi install --locked -e studio-runtime -e onnx", script_text)
        self.assertNotIn("pixi install -a", script_text)
        if os.name == "nt":
            self.assertEqual(script_text.count("if errorlevel 1 exit /b %errorlevel%"), 2)

    def test_filter_dist_environments_keeps_only_runtime_environments(self) -> None:
        pixi_text = (
            "[workspace]\n"
            'name = "demo"\n'
            "\n"
            "[environments]\n"
            'studio-runtime = { features = ["python", "launcher-runtime"] }\n'
            'onnx = { features = ["python", "launcher-runtime"] }\n'
            'mediapipe = { features = ["python", "launcher-runtime"] }\n'
            'ci = { features = ["ci"] }\n'
            'launcher = { features = ["launcher"] }\n'
            "\n"
            "[tasks]\n"
            'demo = "echo ok"\n'
        )

        filtered = self.module._filter_dist_environments(pixi_text, ["studio-runtime", "onnx", "mediapipe"])

        self.assertIn('studio-runtime = { features = ["python", "launcher-runtime"] }', filtered)
        self.assertIn('onnx = { features = ["python", "launcher-runtime"] }', filtered)
        self.assertIn('mediapipe = { features = ["python", "launcher-runtime"] }', filtered)
        self.assertNotIn('ci = { features = ["ci"] }', filtered)
        self.assertNotIn('launcher = { features = ["launcher"] }', filtered)
        self.assertIn("[tasks]", filtered)

    def test_filter_dist_environments_fails_when_runtime_environment_missing(self) -> None:
        pixi_text = (
            "[environments]\n"
            'studio-runtime = { features = ["python", "launcher-runtime"] }\n'
        )

        with self.assertRaises(ValueError) as ctx:
            self.module._filter_dist_environments(pixi_text, ["studio-runtime", "onnx"])

        self.assertIn("onnx", str(ctx.exception))

    def test_filter_dist_feature_sections_removes_unused_feature_sections(self) -> None:
        pixi_text = (
            "[feature.python.dependencies]\n"
            'python = ">=3.13,<3.14"\n'
            "\n"
            "[feature.web-studio.pypi-dependencies]\n"
            'f8studio-server = { path = "packages/f8studio_server", editable = true }\n'
            "\n"
            "[feature.launcher-runtime]\n"
            "\n"
            "[feature.test.dependencies]\n"
            'pytest = ">=8"\n'
            "\n"
            "[feature.ci.tasks]\n"
            'dist_ci = "python scripts/dist_ci.py"\n'
        )

        filtered = self.module._filter_dist_feature_sections(
            pixi_text,
            ["python", "web-studio", "launcher-runtime"],
        )

        self.assertIn("[feature.python.dependencies]", filtered)
        self.assertIn("[feature.web-studio.pypi-dependencies]", filtered)
        self.assertIn("[feature.launcher-runtime]", filtered)
        self.assertNotIn("[feature.test.dependencies]", filtered)
        self.assertNotIn("[feature.ci.tasks]", filtered)

    def test_remove_dist_pixi_build_preview_keeps_other_workspace_fields(self) -> None:
        pixi_text = (
            "[workspace]\n"
            'name = "demo"\n'
            'preview = ["pixi-build"]\n'
            'platforms = ["win-64"]\n'
        )

        filtered = self.module._remove_dist_pixi_build_preview(pixi_text)

        self.assertNotIn('preview = ["pixi-build"]', filtered)
        self.assertIn('name = "demo"', filtered)
        self.assertIn('platforms = ["win-64"]', filtered)

    def test_rewrite_service_entry_environment_args_swaps_default_for_dist_runtime(self) -> None:
        service_text = (
            "launch:\n"
            '  command: pixi\n'
            '  args: ["run", "-e", "default", "f8pyengine"]\n'
        )

        rewritten = self.module._rewrite_service_entry_environment_args(
            service_text,
            source_environment_name="default",
            target_environment_name="studio-runtime",
        )

        self.assertIn('"run", "-e", "studio-runtime", "f8pyengine"', rewritten)
        self.assertNotIn('"run", "-e", "default", "f8pyengine"', rewritten)

    def test_rewrite_dist_service_entries_only_updates_default_runtime_services(self) -> None:
        services_root = self.root / "services"
        engine_service_path = services_root / "f8" / "engine" / "service.yml"
        detector_service_path = services_root / "f8" / "dl" / "detector" / "service.yml"
        engine_service_path.parent.mkdir(parents=True, exist_ok=True)
        detector_service_path.parent.mkdir(parents=True, exist_ok=True)
        engine_service_path.write_text(
            "launch:\n"
            '  command: pixi\n'
            '  args: ["run", "-e", "default", "f8pyengine"]\n',
            encoding="utf-8",
        )
        detector_service_path.write_text(
            "launch:\n"
            '  command: pixi\n'
            '  args: ["run", "-e", "onnx", "f8pydl_detector"]\n',
            encoding="utf-8",
        )

        rewritten_paths = self.module._rewrite_dist_service_entries(services_root)

        self.assertEqual(rewritten_paths, [engine_service_path])
        self.assertIn(
            '"run", "-e", "studio-runtime", "f8pyengine"',
            engine_service_path.read_text(encoding="utf-8"),
        )
        self.assertIn(
            '"run", "-e", "onnx", "f8pydl_detector"',
            detector_service_path.read_text(encoding="utf-8"),
        )

    def test_rewrite_dist_service_entries_maps_web_studio_runtime_to_dist_runtime(self) -> None:
        engine_service_path = self.root / "services" / "f8" / "engine" / "service.yml"
        engine_service_path.parent.mkdir(parents=True, exist_ok=True)
        engine_service_path.write_text(
            'launch:\n  command: pixi\n  args: ["run", "-e", "web-studio-runtime", "f8pyengine"]\n',
            encoding="utf-8",
        )

        self.module._rewrite_dist_service_entries(self.root / "services")

        self.assertIn('"run", "-e", "studio-runtime", "f8pyengine"', engine_service_path.read_text(encoding="utf-8"))

    def test_rewrite_and_validate_block_style_environment(self) -> None:
        path = self.root / "services" / "service.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("launch:\n  command: pixi\n  args:\n    - run\n    - --environment\n    - web-studio-runtime\n    - f8pyengine\n")
        self.module._rewrite_dist_service_entries(path.parent)
        self.module._validate_dist_service_environments(path.parent, ["studio-runtime"])
        self.assertIn('"studio-runtime"', path.read_text())
        with self.assertRaises(ValueError):
            self.module._validate_dist_service_environments(path.parent, ["onnx"])

    def test_validate_dist_service_environments_rejects_unshipped_environment(self) -> None:
        service_path = self.root / "services" / "f8" / "tool" / "service.yml"
        service_path.parent.mkdir(parents=True, exist_ok=True)
        service_path.write_text(
            'launch:\n  command: pixi\n  args: ["run", "-e", "web-studio", "f8tool"]\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "'web-studio'"):
            self.module._validate_dist_service_environments(self.root / "services", ["studio-runtime", "onnx"])

    def test_copy_dist_config_copies_service_index(self) -> None:
        config_root = self.root / "config"
        config_root.mkdir(parents=True, exist_ok=True)
        index_path = config_root / "service-index.json"
        index_path.write_text('{"schemaVersion":"f8serviceIndex/1","services":[],"modelRoot":"../resources/models"}\n', encoding="utf-8")
        dist_dir = self.root / "dist"
        dist_dir.mkdir(parents=True, exist_ok=True)

        with mock.patch.object(self.module, "REPO_ROOT", self.root):
            copied_root = self.module._copy_dist_config(dist_dir)

        self.assertEqual(copied_root, dist_dir / "config")
        self.assertEqual(
            (dist_dir / "config" / "service-index.json").read_text(encoding="utf-8"),
            index_path.read_text(encoding="utf-8"),
        )


class DistCiCppBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_dist_ci_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_preset_file(self, preset_name: str = "conan-release") -> Path:
        preset_path = self.root / "build" / "Release" / "generators" / "CMakePresets.json"
        preset_path.parent.mkdir(parents=True, exist_ok=True)
        preset_path.write_text(
            "{\n"
            '  "version": 3,\n'
            '  "buildPresets": [{"name": "' + preset_name + '"}],\n'
            '  "configurePresets": [{"name": "' + preset_name + '"}]\n'
            "}\n",
            encoding="utf-8",
        )
        return preset_path

    def test_build_cpp_runtime_uses_aggregate_deploy_target(self) -> None:
        preset_path = self._write_preset_file()
        recorded_commands: list[list[str]] = []

        def _record_run(command: list[str]) -> None:
            recorded_commands.append(command)

        with (
            mock.patch.object(self.module, "CPP_PRESET_PATH", preset_path),
            mock.patch.object(self.module, "_run", side_effect=_record_run),
        ):
            self.module._build_cpp_runtime()

        self.assertGreaterEqual(len(recorded_commands), 2)
        build_command = next(command for command in recorded_commands if "--build" in command)
        target_flag_index = build_command.index("--target")
        self.assertEqual(build_command[target_flag_index + 1], self.module.CPP_DEPLOY_ALL_TARGET)
        self.assertNotIn("f8implayer_service_deploy_runtime", build_command)
        self.assertNotIn("f8cvkit_template_match_service_deploy_runtime", build_command)

    def test_build_cpp_runtime_accepts_conan_user_preset_include_path(self) -> None:
        preset_path = self.root / "build" / "generators" / "CMakePresets.json"
        preset_path.parent.mkdir(parents=True, exist_ok=True)
        preset_path.write_text(
            "{\n"
            '  "version": 3,\n'
            '  "buildPresets": [{"name": "conan-release"}],\n'
            '  "configurePresets": [{"name": "conan-release"}]\n'
            "}\n",
            encoding="utf-8",
        )
        user_presets_path = self.root / "CMakeUserPresets.json"
        user_presets_path.write_text(
            "{\n"
            '  "version": 4,\n'
            '  "include": ["build/generators/CMakePresets.json"]\n'
            "}\n",
            encoding="utf-8",
        )
        recorded_commands: list[list[str]] = []

        def _record_run(command: list[str]) -> None:
            recorded_commands.append(command)

        with (
            mock.patch.object(self.module, "REPO_ROOT", self.root),
            mock.patch.object(self.module, "CPP_USER_PRESETS_PATH", user_presets_path),
            mock.patch.object(self.module, "_run", side_effect=_record_run),
        ):
            self.module._build_cpp_runtime()

        bootstrap_command = ["pixi", "run", "--frozen", "-e", "cpp", "cpp_bootstrap"]
        self.assertNotIn(bootstrap_command, recorded_commands)
        self.assertTrue(any("--build" in command for command in recorded_commands))

    def test_build_cpp_runtime_reports_actionable_error_for_missing_registration(self) -> None:
        preset_path = self._write_preset_file()

        def _fake_run(command: list[str]) -> None:
            if "--build" in command:
                raise subprocess.CalledProcessError(returncode=1, cmd=command)

        with (
            mock.patch.object(self.module, "CPP_PRESET_PATH", preset_path),
            mock.patch.object(self.module, "_run", side_effect=_fake_run),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self.module._build_cpp_runtime()

        error_text = str(ctx.exception)
        self.assertIn(self.module.CPP_DEPLOY_ALL_TARGET, error_text)
        self.assertIn("f8_deploy_service_runtime(...)", error_text)

    def test_build_cpp_runtime_requires_canonical_release_build_preset(self) -> None:
        preset_path = self._write_preset_file(preset_name="conan-default")

        with mock.patch.object(self.module, "CPP_PRESET_PATH", preset_path):
            with self.assertRaises(FileNotFoundError) as ctx:
                self.module._build_cpp_runtime()

        self.assertIn("canonical release build preset", str(ctx.exception))


class DistCiLauncherScriptTest(unittest.TestCase):
    def test_bundle_launcher_copies_script_without_running_a_compiler(self) -> None:
        module = _load_dist_ci_module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.object(module, "_run") as run:
                module._bundle_studio_launcher(output)
            run.assert_not_called()
            launcher = output / module._launcher_script_name()
            self.assertIn("studio_launch", launcher.read_text())
            if os.name != "nt":
                self.assertTrue(os.access(launcher, os.X_OK))


class DistCiUnityModsBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_dist_ci_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.dist_dir = self.root / "bundle"
        self.dist_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_bundle_unitymods_assets_runs_bundle_command_on_windows(self) -> None:
        with (
            mock.patch.object(self.module.os, "name", "nt"),
            mock.patch.object(self.module, "REPO_ROOT", self.root),
            mock.patch.object(self.module, "_run") as run_mock,
        ):
            output_dir = self.module._bundle_unitymods_assets(self.dist_dir)

        self.assertEqual(output_dir, self.dist_dir / "unitymods")
        run_mock.assert_called_once_with(
            [
                sys.executable,
                str(self.root / "scripts" / "unitymods_ci.py"),
                "bundle",
                "--output",
                str(self.dist_dir / "unitymods"),
            ]
        )

    def test_bundle_unitymods_assets_is_disabled_off_windows(self) -> None:
        with (
            mock.patch.object(self.module.os, "name", "posix"),
            mock.patch.object(self.module, "_run") as run_mock,
        ):
            output_dir = self.module._bundle_unitymods_assets(self.dist_dir)

        self.assertIsNone(output_dir)
        run_mock.assert_not_called()

    def test_bundle_unitymods_assets_can_reuse_prebuilt_assets(self) -> None:
        with (
            mock.patch.object(self.module.os, "name", "nt"),
            mock.patch.object(self.module, "REPO_ROOT", self.root),
            mock.patch.object(self.module, "_run") as run_mock,
        ):
            self.module._bundle_unitymods_assets(self.dist_dir, build_assets=False)

        run_mock.assert_called_once_with(
            [
                sys.executable,
                str(self.root / "scripts" / "unitymods_ci.py"),
                "bundle",
                "--output",
                str(self.dist_dir / "unitymods"),
                "--skip-build",
            ]
        )

if __name__ == "__main__":
    unittest.main()
