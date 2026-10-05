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

    def test_conan_isolated_from_global_cache_and_host_path(self) -> None:
        module = self.module
        prefix = self.root / ".pixi/envs/cpp"
        (prefix / "bin").mkdir(parents=True)
        (self.root / "pixi.lock").write_text("locked toolchain", encoding="utf-8")
        with (
            mock.patch.object(module, "REPO_ROOT", self.root),
            mock.patch.dict(module.os.environ, {"CONAN_HOME": "/global/conan", "PATH": "/usr/bin"}),
            mock.patch.object(module.subprocess, "run") as run,
        ):
            module._run(["conan", "profile", "detect"])
            expected_home = module._conan_home()
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["CONAN_HOME"], str(expected_home))
        self.assertEqual(env["PATH"].split(module.os.pathsep)[0], str(prefix / "bin"))
        self.assertEqual(env["CCACHE_TEMPDIR"], str(self.root / "build/cache/ccache-tmp"))

    def test_host_compiler_is_rejected(self) -> None:
        module = self.module
        (self.root / "pixi.lock").write_text("locked toolchain", encoding="utf-8")
        with (
            mock.patch.object(module, "REPO_ROOT", self.root),
            mock.patch.object(module, "_cpp_tool", side_effect=lambda name: name),
            mock.patch.object(module.sys, "platform", "linux"),
            mock.patch.dict(module.os.environ, {"CC": "/usr/bin/gcc", "CXX": "/usr/bin/g++"}),
        ):
            with self.assertRaisesRegex(ValueError, "Compiler must belong"):
                module._conan_toolchain_args()

    def test_pixi_profile_pins_compilers_sysroot_and_binary_identity(self) -> None:
        module = self.module
        prefix = self.root / ".pixi/envs/cpp"
        (self.root / "pixi.lock").write_text("locked toolchain", encoding="utf-8")
        with (
            mock.patch.object(module, "REPO_ROOT", self.root),
            mock.patch.object(module, "_cpp_tool", side_effect=lambda name: name),
            mock.patch.dict(module.os.environ, {
                "CC": str(prefix / "bin/cc"), "CXX": str(prefix / "bin/c++"),
                "CONDA_BUILD_SYSROOT": str(prefix / "sysroot"),
            }),
        ):
            args = module._conan_toolchain_args()
            self.assertIn(f"user.f8:toolchain={module._toolchain_id()}", args)
        self.assertIn(f"tools.build:sysroot={prefix / 'sysroot'}", args)
        self.assertTrue(any(str(prefix / "bin/c++") in arg for arg in args))
        self.assertIn('tools.info.package_id:confs=["user.f8:toolchain"]', args)

    def test_host_sysroot_is_rejected_with_pixi_compilers(self) -> None:
        module = self.module
        prefix = self.root / ".pixi/envs/cpp"
        (self.root / "pixi.lock").write_text("locked toolchain", encoding="utf-8")
        with (
            mock.patch.object(module, "REPO_ROOT", self.root),
            mock.patch.object(module, "_cpp_tool", side_effect=lambda name: name),
            mock.patch.object(module.sys, "platform", "linux"),
            mock.patch.dict(module.os.environ, {
                "CC": str(prefix / "bin/cc"), "CXX": str(prefix / "bin/c++"),
                "CONDA_BUILD_SYSROOT": "/usr",
            }),
        ):
            with self.assertRaisesRegex(ValueError, "CONDA_BUILD_SYSROOT"):
                module._conan_toolchain_args()

    def test_bootstrap_accepts_conan_user_preset_include_path(self) -> None:
        generated_preset_path = self.root / "build" / "generators" / "CMakePresets.json"

        def _fake_run(
            command: list[str],
        ) -> None:
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
            mock.patch.object(self.module, "_cpp_tool", side_effect=lambda name: name),
            mock.patch.object(self.module, "_run", side_effect=_fake_run),
            mock.patch.object(self.module, "_conan_toolchain_args", return_value=[]),
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
        ) -> None:
            recorded_commands.append(command)

        with (
            mock.patch.object(self.module, "REPO_ROOT", self.root),
            mock.patch.object(self.module, "USER_PRESETS_PATH", self.user_presets_path),
            mock.patch.object(self.module, "_cpp_tool", side_effect=lambda name: name),
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
        ) -> None:
            recorded_commands.append(command)

        with (
            mock.patch.object(self.module, "REPO_ROOT", self.root),
            mock.patch.object(self.module, "USER_PRESETS_PATH", self.user_presets_path),
            mock.patch.object(self.module, "_cpp_tool", side_effect=lambda name: name),
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


class CppToolPathsTest(unittest.TestCase):
    def test_missing_pixi_tool_does_not_fall_back_to_host(self) -> None:
        module = _load_cpp_ci_module()
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(module, "_pixi_cpp_env_path", return_value=Path(directory)):
                with self.assertRaisesRegex(FileNotFoundError, "Missing Pixi cpp tool"):
                    module._cpp_tool("cmake")

    def test_tools_use_platform_specific_pixi_locations(self) -> None:
        module = _load_cpp_ci_module()
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            for platform, name, relative in (
                ('win32', 'cmake', 'Library/bin/cmake.exe'),
                ('win32', 'conan', 'Scripts/conan.exe'),
                ('linux', 'cmake', 'bin/cmake'),
            ):
                tool = prefix / relative
                tool.parent.mkdir(parents=True, exist_ok=True)
                tool.touch()
                with mock.patch.object(module, '_pixi_cpp_env_path', return_value=prefix), \
                     mock.patch.object(module.sys, 'platform', platform):
                    self.assertEqual(module._cpp_tool(name), str(tool))

    def test_run_target_uses_active_build_directory_and_platform_suffix(self) -> None:
        module = _load_cpp_ci_module()
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / 'custom build'
            for platform, suffix in (('win32', '.exe'), ('linux', '')):
                executable = build / 'bin' / ('benchmark' + suffix)
                executable.parent.mkdir(parents=True, exist_ok=True)
                executable.touch()
                with mock.patch.object(module.sys, 'platform', platform), \
                     mock.patch.object(module, '_cmake_build_directory', return_value=build), \
                     mock.patch.object(module, '_build_target') as compile_target, \
                     mock.patch.object(module, '_run') as run:
                    module._run_target('benchmark')
                compile_target.assert_called_once_with('benchmark')
                run.assert_called_once_with([str(executable)])
