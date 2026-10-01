import subprocess
from pathlib import Path
from unittest import mock

import pytest

from f8studio_server.tray import stop_process, open_console


def test_tray_exit_closes_parent_pipe_before_waiting() -> None:
    process = mock.Mock(spec=subprocess.Popen)
    process.poll.return_value = None
    process.stdin = mock.Mock()
    stop_process(process)
    process.stdin.close.assert_called_once()
    process.wait.assert_called_once_with(timeout=15)
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_stuck_server_escalates_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    process = mock.Mock(spec=subprocess.Popen)
    process.pid = 123
    process.poll.return_value = None
    process.stdin = mock.Mock()
    process.wait.side_effect = [subprocess.TimeoutExpired('studio', 15),
                                subprocess.TimeoutExpired('studio', 5), 0]
    stop_process(process)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert 'did not stop' in caplog.text and 'killing PID' in caplog.text


def test_console_uses_argument_list_for_paths_with_spaces(tmp_path: Path) -> None:
    path = tmp_path / 'logs with spaces.txt'
    with mock.patch('f8studio_server.tray.os.name', 'posix'), \
         mock.patch('f8studio_server.tray.shutil.which', return_value='/usr/bin/xterm'), \
         mock.patch('f8studio_server.tray.subprocess.Popen') as popen:
        open_console(path)
    popen.assert_called_once_with(['/usr/bin/xterm', '-e', 'tail', '-n', '200', '-F', str(path)])


def test_menu_exit_leaves_process_cleanup_to_supervisor(tmp_path: Path) -> None:
    from f8studio_server.tray import run_tray

    backend = mock.MagicMock()
    icon = backend.Icon.return_value
    process = mock.Mock(spec=subprocess.Popen)
    process.stdin = mock.Mock()
    process.poll.return_value = None

    def exit_from_menu(_setup: object) -> None:
        callback = backend.MenuItem.call_args_list[-1].args[1]
        callback()
        icon.stop.assert_called_once()
        process.stdin.close.assert_not_called()

    icon.run.side_effect = exit_from_menu
    with (
        mock.patch.dict('sys.modules', {'pystray': backend}),
        mock.patch('f8studio_server.tray.sys.platform', 'darwin'),
        mock.patch('f8studio_server.tray.subprocess.Popen', return_value=process),
    ):
        run_tray(arguments=['--no-browser'], url='http://127.0.0.1:8210', data_dir=tmp_path)
    process.stdin.close.assert_called_once()
    process.wait.assert_called_once_with(timeout=15)
