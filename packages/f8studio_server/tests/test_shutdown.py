"""Real sockets are necessary: TestClient cancels leaked websocket tasks for us."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from websockets.sync.client import connect

from f8studio_server.server_instance import single_server_instance
from f8studio_server.tray import stop_process


def test_parent_exit_with_idle_browser_connections_releases_instance_lock(tmp_path: Path) -> None:
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    # Restart with exactly the same data directory and port.
    for attempt in range(2):
        log_path = tmp_path / f'server-{attempt}.log'
        with log_path.open('wb') as output:
            process = subprocess.Popen(
                [sys.executable, '-m', 'f8studio_server', '--no-browser', '--port', str(port),
                 '--exit-on-stdin-close'],
                stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT,
                env={**os.environ, 'F8STUDIO_DATA_DIR': str(tmp_path)},
            )
            try:
                deadline = time.monotonic() + 30
                while True:
                    assert process.poll() is None, log_path.read_text()
                    access_path = tmp_path / 'access.json'
                    if access_path.exists():
                        token = json.loads(access_path.read_text())['token']
                        try:
                            with urlopen(Request(f'http://127.0.0.1:{port}/api/health',
                                                 headers={'Authorization': f'Bearer {token}'}), timeout=1):
                                break
                        except (URLError, TimeoutError) as exc:
                            # Connection refusal is expected until startup binds the socket.
                            if time.monotonic() >= deadline:
                                raise AssertionError(log_path.read_text()) from exc
                    assert time.monotonic() < deadline, log_path.read_text()
                    time.sleep(0.05)
                headers = {'Authorization': f'Bearer {token}'}
                with (
                    connect(f'ws://127.0.0.1:{port}/api/events', additional_headers=headers) as events,
                    connect(f'ws://127.0.0.1:{port}/api/live', additional_headers=headers) as live,
                ):
                    events.recv(timeout=5)
                    live.recv(timeout=5)
                    assert process.stdin is not None
                    process.stdin.close()
                    assert process.wait(timeout=12) == 0, log_path.read_text()
                with single_server_instance(tmp_path / 'server.lock'):
                    pass
            finally:
                stop_process(process)
        log = log_path.read_text()
        assert 'Traceback' not in log, log
        assert 'timeout graceful shutdown exceeded' not in log, log
        # Both the server and its managed gateway must finish.
        assert log.count('Finished server process') == 2, log
