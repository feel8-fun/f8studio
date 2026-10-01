"""Desktop tray supervisor; closing the browser does not stop Studio."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from importlib.resources import files
import shutil
import signal
import subprocess
import sys
from threading import Thread
import webbrowser

logger = logging.getLogger(__name__)


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if process.stdin is not None:
        process.stdin.close()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        logger.warning("Studio did not stop within 15 seconds; terminating PID %s", process.pid)
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("Studio did not terminate; killing PID %s", process.pid)
            process.kill()
            process.wait()


def open_console(path: Path) -> None:
    if os.name == 'nt':
        subprocess.Popen(['powershell.exe', '-NoExit', '-NoProfile', '-Command',
                          'Get-Content -LiteralPath $env:F8_STUDIO_LOG -Tail 200 -Wait'],
                         env={**os.environ, 'F8_STUDIO_LOG': str(path)},
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:
        terminal = shutil.which('x-terminal-emulator') or shutil.which('xterm')
        if terminal:
            subprocess.Popen([terminal, '-e', 'tail', '-n', '200', '-F', str(path)])
        elif not webbrowser.open(path.as_uri()):
            raise OSError(f'No terminal or file viewer available; log: {path}')


def run_tray(*, arguments: list[str], url: str, data_dir: Path) -> None:
    # Import only for desktop mode. Linux needs GTK/AppIndicator, not the Xorg
    # backend (which cannot display a context menu).
    if sys.platform.startswith('linux'):
        os.environ.setdefault('PYSTRAY_BACKEND', 'gtk')
    try:
        import pystray
        from PIL import Image
        with files('f8studio_server').joinpath('assets', 'tray-icon.png').open('rb') as source:
            icon_image = Image.open(source).convert('RGBA')
        icon = pystray.Icon('f8studio', icon_image, 'F8 Studio')
        if not icon.HAS_MENU:
            raise RuntimeError('This desktop tray backend does not support menus')
    except (ImportError, RuntimeError, OSError):
        logger.warning('Tray unavailable; continuing in console mode', exc_info=True)
        subprocess.run([sys.executable, '-m', 'f8studio_server', *arguments], check=True)
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    log_path = data_dir / 'studio-console.log'
    with log_path.open('ab', buffering=0) as output:
        process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'f8studio_server', *arguments, '--exit-on-stdin-close'],
            stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=os.name != 'nt',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        def open_studio() -> None:
            try:
                if not webbrowser.open(url):
                    logger.warning('Open Studio manually: %s', url)
            except (OSError, webbrowser.Error):
                logger.exception('Failed to open Studio browser')

        def show_console() -> None:
            try:
                open_console(log_path)
            except (OSError, webbrowser.Error):
                logger.exception('Failed to open Studio console: %s', log_path)

        def exit_studio() -> None:
            # Keep the UI callback nonblocking; the supervisor's finally owns cleanup.
            icon.stop()

        icon.menu = pystray.Menu(
            pystray.MenuItem('Open Studio', open_studio, default=True),
            pystray.MenuItem('Open console / logs', show_console),
            pystray.MenuItem('Exit', exit_studio),
        )

        def monitor() -> None:
            returncode = process.wait()
            if returncode:
                logger.error('Studio exited with code %s; log: %s', returncode, log_path)
                show_console()
            icon.stop()

        def setup(_tray_icon: object) -> None:
            icon.visible = True
            Thread(target=monitor, name='studio-tray-monitor', daemon=True).start()

        print(f'Studio tray is running. Console log: {log_path}', flush=True)
        previous_int = signal.getsignal(signal.SIGINT)
        previous_term = signal.signal(signal.SIGTERM, lambda signum, frame: icon.stop())
        if sys.platform.startswith('linux'):
            from gi.repository import GLib  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]  # GI exposes native modules without stubs.

            def install_interrupt_handler() -> bool:
                # GTK's pystray backend resets SIGINT during initialization.
                # Run on the main loop after that reset, before user interaction.
                signal.signal(signal.SIGINT, lambda signum, frame: icon.stop())
                return False

            GLib.idle_add(install_interrupt_handler)  # pyright: ignore[reportUnknownMemberType]
        try:
            icon.run(setup)  # pyright: ignore[reportUnknownMemberType]  # pystray leaves setup untyped
        except KeyboardInterrupt:
            logger.info('Stopping Studio tray')
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)
            stop_process(process)
            icon.stop()
