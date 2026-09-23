from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_launcher_module() -> object:
    script_path = Path("scripts/f8studio_launcher.py").resolve()
    spec = importlib.util.spec_from_file_location("f8studio_launcher", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load f8studio_launcher module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LauncherEnvironmentDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_launcher_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_discover_launcher_install_environments_uses_marker_feature(self) -> None:
        (self.root / "pixi.toml").write_text(
            "[environments]\n"
            'studio-runtime = { features = ["python", "studio", "launcher-runtime"] }\n'
            'onnx = { features = ["python", "onnx", "launcher-runtime"] }\n'
            'ci = { features = ["ci"] }\n',
            encoding="utf-8",
        )

        env_names = self.module._discover_launcher_install_environments(self.root)

        self.assertEqual(env_names, ["studio-runtime", "onnx"])

    def test_discover_launcher_install_environments_fails_without_marker_feature(self) -> None:
        (self.root / "pixi.toml").write_text(
            "[environments]\n"
            'ci = { features = ["ci"] }\n',
            encoding="utf-8",
        )

        with self.assertRaises(ValueError) as ctx:
            self.module._discover_launcher_install_environments(self.root)

        self.assertIn("launcher-runtime", str(ctx.exception))

    def test_read_workspace_version_returns_workspace_version(self) -> None:
        (self.root / "pixi.toml").write_text(
            "[workspace]\n"
            'version = "0.2.0"\n',
            encoding="utf-8",
        )

        version = self.module._read_workspace_version(self.root)

        self.assertEqual(version, "0.2.0")


class LauncherInstallCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_launcher_module()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_install_workspace_environments_uses_targeted_environment_flags(self) -> None:
        with (
            mock.patch.object(
                self.module,
                "_run_subprocess_with_status",
                return_value=subprocess.CompletedProcess([], 0),
            ) as run_mock,
            mock.patch.object(
                self.module,
                "_installed_pixi_environment_names",
                return_value={"studio-runtime", "onnx"},
            ),
        ):
            ok = self.module._install_workspace_environments(
                "pixi",
                self.root,
                ["studio-runtime", "onnx"],
            )

        self.assertTrue(ok)
        run_mock.assert_called_once_with(
            ["pixi", "install", "-e", "studio-runtime", "-e", "onnx"],
            cwd=self.root,
            check=False,
            status_title="f8studio",
            status_message=self.module.SPLASH_RUNTIME_INSTALL_MESSAGE,
            status_window=None,
        )

    def test_main_launches_studio_runtime_explicitly(self) -> None:
        splash_window = mock.Mock()
        launch_proc = mock.Mock()
        with (
            mock.patch.object(self.module, "_find_workspace_root", return_value=self.root),
            mock.patch.object(self.module, "_find_pixi_executable", return_value="pixi"),
            mock.patch.object(
                self.module,
                "_discover_launcher_install_environments",
                return_value=["studio-runtime", "onnx"],
            ),
            mock.patch.object(
                self.module,
                "_installed_pixi_environment_names",
                return_value={"studio-runtime", "onnx"},
            ),
            mock.patch.object(self.module, "_create_status_window", return_value=splash_window),
            mock.patch.object(self.module, "_show_error_dialog") as error_dialog_mock,
            mock.patch.object(self.module, "_start_subprocess", return_value=launch_proc) as start_mock,
            mock.patch.object(self.module, "_complete_startup_splash", return_value=None) as complete_mock,
        ):
            exit_code = self.module.main([])

        self.assertEqual(exit_code, 0)
        error_dialog_mock.assert_not_called()
        splash_window.set_message.assert_called_once_with(
            self.module.SPLASH_LAUNCH_MESSAGE
        )
        start_mock.assert_called_once()
        start_args, start_kwargs = start_mock.call_args
        self.assertEqual(
            start_args,
            ([
                "pixi", "run", "-e", "studio-runtime", "studio_server",
                "--host", "127.0.0.1", "--port", "8210",
            ],),
        )
        self.assertEqual(start_kwargs["cwd"], self.root)
        self.assertIn("env", start_kwargs)

        complete_mock.assert_called_once()
        complete_args, complete_kwargs = complete_mock.call_args
        self.assertEqual(complete_args, (splash_window,))
        self.assertIs(complete_kwargs["launch_proc"], launch_proc)
        self.assertEqual(complete_kwargs["min_visible_s"], self.module.SPLASH_MIN_VISIBLE_S)
        self.assertEqual(complete_kwargs["fade_duration_s"], self.module.SPLASH_FADE_DURATION_S)
        self.assertEqual(complete_kwargs["health_url"], "http://127.0.0.1:8210/api/health")
        self.assertEqual(complete_kwargs["app_url"], "http://127.0.0.1:8210")
        self.assertTrue(complete_kwargs["open_browser"])

    def test_run_subprocess_hides_windows_console(self) -> None:
        with mock.patch.object(self.module.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)) as run_mock:
            self.module._run_subprocess(["pixi", "run", "studio_server"], check=False)

        expected_kwargs: dict[str, object] = {"check": False}
        if self.module.os.name == "nt":
            expected_kwargs["creationflags"] = self.module.subprocess.CREATE_NO_WINDOW
        run_mock.assert_called_once_with(["pixi", "run", "studio_server"], **expected_kwargs)

    def test_run_subprocess_with_status_waits_for_process_completion(self) -> None:
        status_window = mock.Mock()
        proc = mock.Mock()
        proc.poll.side_effect = [None, None, 0]

        with (
            mock.patch.object(self.module, "_create_status_window", return_value=status_window),
            mock.patch.object(self.module.subprocess, "Popen", return_value=proc) as popen_mock,
        ):
            completed = self.module._run_subprocess_with_status(
                ["pixi", "run", "-e", "studio-runtime", "studio_server"],
                cwd=self.root,
                check=False,
                status_title="f8studio",
                status_message="Starting Studio...",
            )

        expected_popen_kwargs: dict[str, object] = {"cwd": self.root}
        if self.module.os.name == "nt":
            expected_popen_kwargs["creationflags"] = self.module.subprocess.CREATE_NO_WINDOW
        popen_mock.assert_called_once_with(
            ["pixi", "run", "-e", "studio-runtime", "studio_server"],
            **expected_popen_kwargs,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertGreaterEqual(status_window.update.call_count, 2)
        self.assertGreaterEqual(status_window.wait.call_count, 2)
        status_window.close.assert_called_once_with()

    def test_complete_startup_splash_fades_after_minimum_visible_time(self) -> None:
        status_window = mock.Mock()
        status_window.elapsed_s.side_effect = [0.4, 1.2, 2.1]
        proc = mock.Mock()
        proc.poll.side_effect = [None, None, None, None]

        with (
            mock.patch.object(self.module, "_server_is_ready", side_effect=[False, False, True]),
            mock.patch.object(self.module, "_open_studio_browser", return_value=True) as browser_mock,
        ):
            returncode = self.module._complete_startup_splash(
                status_window,
                launch_proc=proc,
                min_visible_s=2.0,
                fade_duration_s=1.0,
                health_url="http://127.0.0.1:8210/api/health",
                app_url="http://127.0.0.1:8210",
                open_browser=True,
            )

        self.assertIsNone(returncode)
        self.assertGreaterEqual(status_window.update.call_count, 2)
        self.assertGreaterEqual(status_window.wait.call_count, 2)
        status_window.set_message.assert_called_once_with(self.module.SPLASH_READY_MESSAGE)
        status_window.fade_out.assert_called_once_with(duration_s=1.0)
        browser_mock.assert_called_once_with("http://127.0.0.1:8210")

    def test_complete_startup_splash_waits_for_health_after_minimum_visible_time(self) -> None:
        status_window = mock.Mock()
        status_window.elapsed_s.side_effect = [2.1]
        proc = mock.Mock()
        proc.poll.side_effect = [None, None, None, None]

        with mock.patch.object(self.module, "_server_is_ready", side_effect=[False, True]) as ready_mock:
            returncode = self.module._complete_startup_splash(
                status_window,
                launch_proc=proc,
                min_visible_s=2.0,
                fade_duration_s=1.0,
                health_url="http://127.0.0.1:8210/api/health",
                app_url="http://127.0.0.1:8210",
                open_browser=False,
            )

        self.assertIsNone(returncode)
        self.assertEqual(ready_mock.call_count, 2)
        status_window.wait.assert_called()
        status_window.set_message.assert_called_once_with(self.module.SPLASH_READY_MESSAGE)
        status_window.fade_out.assert_called_once_with(duration_s=1.0)

    def test_complete_startup_splash_returns_early_exit_code(self) -> None:
        status_window = mock.Mock()
        status_window.elapsed_s.side_effect = [0.2]
        proc = mock.Mock()
        proc.poll.return_value = 7

        returncode = self.module._complete_startup_splash(
            status_window,
            launch_proc=proc,
            min_visible_s=2.0,
            fade_duration_s=1.0,
            health_url="http://127.0.0.1:8210/api/health",
            app_url="http://127.0.0.1:8210",
            open_browser=False,
        )

        self.assertEqual(returncode, 7)
        status_window.close.assert_called_once_with()
        status_window.fade_out.assert_not_called()


if __name__ == "__main__":
    unittest.main()
