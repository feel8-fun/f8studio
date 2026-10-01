from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import logging
import os
from pathlib import Path
import signal
import subprocess
from threading import Event, RLock


logger = logging.getLogger(__name__)


class ExtensionInstallCancelled(Exception):
    pass


class InstallOperation:
    def __init__(self, extension_id: str, log_path: Path) -> None:
        self.extension_id = extension_id
        self.cancel_requested = Event()
        self._lock = RLock()
        self._detail = 'Preparing extension'
        self._process: subprocess.Popen[str] | None = None
        self._log_path = log_path

    @property
    def detail(self) -> str:
        with self._lock:
            return self._detail

    def report(self, detail: str) -> None:
        with self._lock:
            self._detail = detail

    def check_cancelled(self) -> None:
        if self.cancel_requested.is_set():
            raise ExtensionInstallCancelled

    def cancel(self, *, force: bool = False) -> None:
        self.cancel_requested.set()
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               capture_output=True, check=False, timeout=10)
            else:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            logger.debug('Extension installer exited during cancellation', exc_info=True)

    def run(self, command: Sequence[str], *, cwd: Path, timeout: float | None = None,
            env: dict[str, str] | None = None) -> str:
        self.check_cancelled()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        recent: deque[str] = deque(maxlen=8)
        output: list[str] = []
        with self._log_path.open('a', encoding='utf-8') as log:
            log.write(f'\nCommand: {list(command)!r}\n')
            with subprocess.Popen(
                command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE if timeout is not None else subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                start_new_session=os.name != 'nt',
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
            ) as process:
                with self._lock:
                    self._process = process
                try:
                    if self.cancel_requested.is_set():
                        self.cancel()
                    if timeout is not None:
                        try:
                            captured, errors = process.communicate(timeout=timeout)
                        except subprocess.TimeoutExpired:
                            self.cancel(force=True)
                            captured, errors = process.communicate()
                            log.write(captured + (errors or ''))
                            log.flush()
                            raise
                        output.append(captured)
                        log.write(captured + (errors or ''))
                        recent.extend((captured + (errors or '')).strip().splitlines()[-8:])
                    else:
                        assert process.stdout is not None
                        for line in process.stdout:
                            output.append(line)
                            log.write(line)
                            log.flush()
                            if line.strip():
                                recent.append(line.strip())
                                self.report(line.strip()[-240:])
                        process.wait()
                finally:
                    with self._lock:
                        self._process = None
        self.check_cancelled()
        if process.returncode:
            raise RuntimeError(f'Command failed ({process.returncode}): {list(command)!r}: '
                               + ' | '.join(recent) + f'; log: {self._log_path}')
        return ''.join(output)
