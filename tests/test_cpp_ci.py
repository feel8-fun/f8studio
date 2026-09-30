from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_cpp_ci_module() -> object:
    script_path = Path("scripts/cpp_ci.py").resolve()
    spec = importlib.util.spec_from_file_location("cpp_ci", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load cpp_ci module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CppCiBootstrapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_cpp_ci_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.lockfile_path = self.root / "conan.lock"
        self.user_presets_path = self.root / "CMakeUserPresets.json"
        self.lockfile_path.write_text("lock", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_bootstrap_accepts_conan_user_preset_include_path(self) -> None:
        generated_preset_path = self.root / "build" / "generators" / "CMakePresets.json"

        def _fake_run(
            command: list[str],
            *,
            use_pixi_cpp_paths: bool = False,
            use_host_pkg_config: bool = False,
        ) -> None:
            self.assertFalse(use_pixi_cpp_paths)
            self.assertTrue(use_host_pkg_config)
            if command[:2] != ["conan", "install"]:
                return
            generated_preset_path.parent.mkdir(parents=True, exist_ok=True)
            generated_preset_path.write_text(
                "{\n"
                '  "version": 3,\n'
                '  "buildPresets": [{"name": "conan-release"}],\n'
                '  "configurePresets": [{"name": "conan-release"}]\n'
                "}\n",
                encoding="utf-8",
            )
            self.user_presets_path.write_text(
                "{\n"
                '  "version": 4,\n'
                '  "include": ["build/generators/CMakePresets.json"]\n'
                "}\n",
                encoding="utf-8",
            )

        with (
            mock.patch.object(self.module, "REPO_ROOT", self.root),
            mock.patch.object(self.module, "LOCKFILE_PATH", self.lockfile_path),
            mock.patch.object(self.module, "USER_PRESETS_PATH", self.user_presets_path),
            mock.patch.object(self.module, "_run", side_effect=_fake_run),
        ):
            self.module._bootstrap()
            presets = self.module._load_generated_presets()

        self.assertEqual(presets["version"], 3)


class CppCiConfigureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_cpp_ci_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.user_presets_path = self.root / "CMakeUserPresets.json"
        self.generated_preset_path = self.root / "build" / "generators" / "CMakePresets.json"
        self.generated_preset_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_presets_path.write_text(
            "{\n"
            '  "version": 4,\n'
            '  "include": ["build/generators/CMakePresets.json"]\n'
            "}\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_configure_uses_release_build_preset_configure_preset(self) -> None:
        self.generated_preset_path.write_text(
            "{\n"
            '  "version": 3,\n'
            '  "configurePresets": [{"name": "conan-default"}],\n'
            '  "buildPresets": [{"name": "conan-release", "configurePreset": "conan-default"}]\n'
            "}\n",
            encoding="utf-8",
        )
        recorded_commands: list[list[str]] = []

        def _record_run(
            command: list[str],
            *,
            use_pixi_cpp_paths: bool = False,
            use_host_pkg_config: bool = False,
        ) -> None:
            self.assertTrue(use_pixi_cpp_paths)
            self.assertFalse(use_host_pkg_config)
            recorded_commands.append(command)

        with (
            mock.patch.object(self.module, "REPO_ROOT", self.root),
            mock.patch.object(self.module, "USER_PRESETS_PATH", self.user_presets_path),
            mock.patch.object(self.module, "_run", side_effect=_record_run),
        ):
            self.module._configure()
            self.module._build()

        configure_command = recorded_commands[0]
        build_command = recorded_commands[1]
        self.assertEqual(configure_command[:3], ["cmake", "--preset", "conan-default"])
        self.assertEqual(build_command[:4], ["cmake", "--build", "--preset", "conan-release"])

    def test_test_command_configures_builds_and_runs_gtest_target(self) -> None:
        self.generated_preset_path.write_text(
            "{\n"
            '  "version": 3,\n'
            '  "configurePresets": [{"name": "conan-default"}],\n'
            '  "buildPresets": [{"name": "conan-release", "configurePreset": "conan-default"}]\n'
            "}\n",
            encoding="utf-8",
        )
        recorded_commands: list[list[str]] = []

        def _record_run(
            command: list[str],
            *,
            use_pixi_cpp_paths: bool = False,
            use_host_pkg_config: bool = False,
        ) -> None:
            self.assertTrue(use_pixi_cpp_paths)
            self.assertFalse(use_host_pkg_config)
            recorded_commands.append(command)

        with (
            mock.patch.object(self.module, "REPO_ROOT", self.root),
            mock.patch.object(self.module, "USER_PRESETS_PATH", self.user_presets_path),
            mock.patch.object(self.module, "_run", side_effect=_record_run),
        ):
            self.module._test()

        self.assertEqual(len(recorded_commands), 3)
        configure_command = recorded_commands[0]
        build_command = recorded_commands[1]
        ctest_command = recorded_commands[2]

        self.assertEqual(configure_command[:3], ["cmake", "--preset", "conan-default"])
        self.assertIn("-DBUILD_TESTS=ON", configure_command)
        self.assertEqual(build_command[:4], ["cmake", "--build", "--preset", "conan-release"])
        self.assertIn("f8cppsdk_tests", build_command)
        self.assertEqual(ctest_command[:3], ["ctest", "--test-dir", str(self.root / "build")])
        self.assertIn("f8cppsdk_tests", ctest_command)


if __name__ == "__main__":
    unittest.main()
