from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


PIXI_INSTALL_DOCS_URL = "https://pixi.prefix.dev/latest/installation/"
LAUNCHER_RUNTIME_FEATURE = "launcher-runtime"
STUDIO_RUNTIME_ENVIRONMENT = "studio-runtime"
STUDIO_DEFAULT_HOST = "127.0.0.1"
STUDIO_DEFAULT_PORT = 8210
STUDIO_STARTUP_TIMEOUT_S = 120.0
SPLASH_MIN_VISIBLE_S = 2.0
SPLASH_FADE_DURATION_S = 1.0
SPLASH_POLL_MS = 50
SPLASH_FADE_STEP_S = 0.04
SPLASH_LOGO_FILENAME = "logo_transparent.png"
SPLASH_ICON_FILENAME = "icon.png"
WINDOW_ICON_FILENAME = "icon.ico"
SPLASH_BOOT_MESSAGE = "Preparing F8Studio launcher resources..."
SPLASH_RUNTIME_INSTALL_MESSAGE = "Preparing runtime environment..."
SPLASH_PIXI_INSTALL_MESSAGE = "Installing Pixi..."
SPLASH_LAUNCH_MESSAGE = "Starting F8Studio..."
SPLASH_READY_MESSAGE = "F8Studio is ready. Enjoy."
SPLASH_SUBTITLE = "First launch may take a little longer."


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launcher for Feel8 Web Studio via Pixi.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit without launching.")
    parser.add_argument("--port", type=int, default=STUDIO_DEFAULT_PORT, help="Loopback HTTP port.")
    parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser.")
    return parser


def _show_error_dialog(title: str, message: str) -> None:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()


def _launcher_dir() -> Path:
    if "__compiled__" in globals():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _asset_search_roots() -> list[Path]:
    roots: list[Path] = []
    for candidate in (
        _launcher_dir(),
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent,
    ):
        try:
            resolved = candidate.resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved not in roots:
            roots.append(resolved)
    return roots


def _find_asset_path(filename: str) -> Path | None:
    for root in _asset_search_roots():
        for candidate in (root / "assets" / filename, root / filename):
            if candidate.is_file():
                return candidate
    return None


def _find_workspace_root(start_dir: Path) -> Path:
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "pixi.toml").is_file():
            return candidate
    raise FileNotFoundError(f"pixi.toml was not found from {start_dir}")


def _read_workspace_version(workspace_root: Path) -> str | None:
    pixi_toml_path = workspace_root / "pixi.toml"
    with pixi_toml_path.open("rb") as pixi_file:
        manifest = tomllib.load(pixi_file)

    workspace_table = manifest.get("workspace")
    if not isinstance(workspace_table, dict):
        return None

    version = workspace_table.get("version")
    if not isinstance(version, str):
        return None
    version_text = version.strip()
    if not version_text:
        return None
    return version_text


def _pixi_candidates() -> list[Path]:
    home = Path.home()
    candidates = [home / ".pixi" / "bin" / "pixi"]
    if os.name == "nt":
        candidates.append(home / ".pixi" / "bin" / "pixi.exe")
    return candidates


def _find_pixi_executable() -> str | None:
    path_hit = shutil.which("pixi")
    if path_hit:
        return path_hit
    for candidate in _pixi_candidates():
        if candidate.is_file():
            return str(candidate)
    return None


class _StatusWindow:
    def __init__(self, *, title: str, message: str, version_text: str | None = None) -> None:
        import tkinter as tk

        self._tk = tk
        self._closed = False
        self._icon_image = None
        self._logo_image = None
        self._shown_at = time.monotonic()
        self._root = tk.Tk()
        self._root.title(title)
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", lambda: None)
        self._root.attributes("-topmost", True)
        self._root.overrideredirect(True)
        self._root.configure(background="#020617")
        self._apply_window_icon()

        frame = tk.Frame(
            self._root,
            bg="#020617",
            padx=18,
            pady=18,
        )
        frame.pack(fill="both", expand=True)

        border_frame = tk.Frame(
            frame,
            bg="#1e293b",
            highlightthickness=0,
            bd=0,
        )
        border_frame.pack(fill="both", expand=True)

        card = tk.Frame(
            border_frame,
            bg="#0f172a",
            padx=28,
            pady=24,
        )
        card.pack(fill="both", expand=True, padx=1, pady=1)

        header = tk.Frame(card, bg="#0f172a")
        header.pack(fill="x", pady=(0, 14))

        brand_label = tk.Label(
            header,
            text="F8Studio Launcher",
            font=("Segoe UI", 9, "bold"),
            fg="#7dd3fc",
            bg="#0f172a",
        )
        brand_label.pack(side="left")

        if version_text is not None:
            version_badge = tk.Label(
                header,
                text=f"v{version_text}",
                font=("Segoe UI", 8, "bold"),
                fg="#e2e8f0",
                bg="#162033",
                padx=8,
                pady=3,
            )
            version_badge.pack(side="right")

        accent_bar = tk.Frame(card, bg="#38bdf8", height=2)
        accent_bar.pack(fill="x", pady=(0, 18))

        logo_path = _find_asset_path(SPLASH_LOGO_FILENAME) or _find_asset_path(SPLASH_ICON_FILENAME)
        if logo_path is not None:
            image = tk.PhotoImage(file=os.fspath(logo_path))
            max_width = 320
            if image.width() > max_width:
                scale = max(1, math.ceil(image.width() / max_width))
                image = image.subsample(scale, scale)
            self._logo_image = image
            logo_label = tk.Label(card, image=image, bg="#0f172a", bd=0)
            logo_label.pack(pady=(0, 14))

        self._label = tk.Label(
            card,
            text=message,
            justify="center",
            wraplength=380,
            font=("Segoe UI", 11, "bold"),
            fg="#f8fafc",
            bg="#0f172a",
            pady=2,
        )
        self._label.pack(fill="x")

        self._subtitle_label = tk.Label(
            card,
            text=SPLASH_SUBTITLE,
            justify="center",
            wraplength=380,
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#0f172a",
        )
        self._subtitle_label.pack(fill="x", pady=(10, 0))

        self._root.update_idletasks()
        width = self._root.winfo_width()
        height = self._root.winfo_height()
        screen_width = self._root.winfo_screenwidth()
        screen_height = self._root.winfo_screenheight()
        x_pos = max(0, (screen_width - width) // 2)
        y_pos = max(0, (screen_height - height) // 2)
        self._root.geometry(f"{width}x{height}+{x_pos}+{y_pos}")

    def _apply_window_icon(self) -> None:
        icon_ico_path = _find_asset_path(WINDOW_ICON_FILENAME)
        if icon_ico_path is not None and os.name == "nt":
            try:
                self._root.iconbitmap(default=os.fspath(icon_ico_path))
                return
            except self._tk.TclError:
                pass

        icon_png_path = _find_asset_path(SPLASH_ICON_FILENAME)
        if icon_png_path is not None:
            try:
                icon_image = self._tk.PhotoImage(file=os.fspath(icon_png_path))
                self._root.iconphoto(True, icon_image)
                self._icon_image = icon_image
            except self._tk.TclError:
                return

    def update(self) -> None:
        if self._closed:
            return
        self._root.update_idletasks()
        self._root.update()

    def wait(self, ms: int) -> None:
        if self._closed:
            return
        self._root.after(ms)

    def set_message(self, message: str) -> None:
        if self._closed:
            return
        self._label.configure(text=message)
        self.update()

    def elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self._shown_at)

    def fade_out(self, *, duration_s: float) -> None:
        if duration_s <= 0:
            self.close()
            return

        try:
            self._root.attributes("-alpha", 1.0)
        except self._tk.TclError:
            self.close()
            return

        steps = max(1, int(duration_s / SPLASH_FADE_STEP_S))
        for step in range(steps + 1):
            progress = step / steps
            alpha = pow(max(0.0, 1.0 - progress), 3.0)
            try:
                self._root.attributes("-alpha", alpha)
            except self._tk.TclError:
                break
            self.update()
            if step < steps:
                self.wait(int(SPLASH_FADE_STEP_S * 1000))
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._root.destroy()


def _create_status_window(
    *,
    title: str,
    message: str,
    version_text: str | None = None,
) -> _StatusWindow | None:
    try:
        import tkinter as tk
    except ImportError:
        return None

    try:
        return _StatusWindow(title=title, message=message, version_text=version_text)
    except tk.TclError:
        return None
    except RuntimeError:
        return None


def _close_status_window(status_window: _StatusWindow | None) -> None:
    if status_window is None:
        return
    status_window.close()


def _subprocess_creationflags() -> int:
    if os.name == "nt":
        return subprocess.CREATE_NO_WINDOW
    return 0


def _popen_kwargs(*, cwd: Path | None = None, env: dict[str, str] | None = None) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if cwd is not None:
        kwargs["cwd"] = cwd
    if env is not None:
        kwargs["env"] = env
    if os.name == "nt":
        kwargs["creationflags"] = _subprocess_creationflags()
    return kwargs


def _run_subprocess(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[object]:
    if os.name == "nt":
        return subprocess.run(command, creationflags=_subprocess_creationflags(), **kwargs)
    return subprocess.run(command, **kwargs)


def _run_subprocess_with_status(
    command: list[str],
    *,
    cwd: Path | None = None,
    check: bool,
    status_title: str,
    status_message: str,
    status_window: _StatusWindow | None = None,
) -> subprocess.CompletedProcess[object]:
    owned_status_window = status_window
    if owned_status_window is None:
        owned_status_window = _create_status_window(title=status_title, message=status_message)
    elif status_message:
        owned_status_window.set_message(status_message)

    if owned_status_window is None:
        return _run_subprocess(command, cwd=cwd, check=check)

    proc = subprocess.Popen(command, **_popen_kwargs(cwd=cwd))
    returncode: int | None = None

    try:
        while True:
            owned_status_window.update()
            returncode = proc.poll()
            if returncode is not None:
                break
            owned_status_window.wait(SPLASH_POLL_MS)
    finally:
        if status_window is None:
            owned_status_window.close()

    completed = subprocess.CompletedProcess(command, returncode)
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    return completed


def _start_subprocess(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[object]:
    return subprocess.Popen(command, **_popen_kwargs(cwd=cwd, env=env))


def _server_is_ready(health_url: str) -> bool:
    request = urllib.request.Request(health_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=0.5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _open_studio_browser(app_url: str) -> bool:
    try:
        return bool(webbrowser.open(app_url, new=1, autoraise=True))
    except webbrowser.Error:
        return False


def _inspect_launch_state(launch_proc: subprocess.Popen[object], *, health_url: str) -> tuple[str, int | None]:
    returncode = launch_proc.poll()
    if returncode is not None:
        return "exited", int(returncode)
    if _server_is_ready(health_url):
        return "ready", None
    return "waiting", None


def _complete_startup_splash(
    status_window: _StatusWindow | None,
    *,
    launch_proc: subprocess.Popen[object],
    min_visible_s: float,
    fade_duration_s: float,
    health_url: str,
    app_url: str,
    open_browser: bool,
    startup_timeout_s: float = STUDIO_STARTUP_TIMEOUT_S,
) -> int | None:
    started_at = time.monotonic()
    if status_window is None:
        while time.monotonic() - started_at < startup_timeout_s:
            state, returncode = _inspect_launch_state(launch_proc, health_url=health_url)
            if state == "exited":
                return int(returncode)
            if state == "ready":
                if open_browser:
                    _open_studio_browser(app_url)
                return None
            time.sleep(SPLASH_POLL_MS / 1000)
        return 5

    while status_window.elapsed_s() < min_visible_s:
        status_window.update()
        state, returncode = _inspect_launch_state(launch_proc, health_url=health_url)
        if state == "exited":
            status_window.close()
            return int(returncode)
        status_window.wait(SPLASH_POLL_MS)

    while time.monotonic() - started_at < startup_timeout_s:
        state, returncode = _inspect_launch_state(launch_proc, health_url=health_url)
        if state == "exited":
            status_window.close()
            return int(returncode)
        if state == "ready":
            break
        status_window.update()
        status_window.wait(SPLASH_POLL_MS)
    else:
        status_window.close()
        return 5

    if open_browser:
        _open_studio_browser(app_url)
    status_window.set_message(SPLASH_READY_MESSAGE)
    status_window.fade_out(duration_s=fade_duration_s)

    returncode = launch_proc.poll()
    if returncode is None:
        return None
    return int(returncode)


def _installed_pixi_environment_names(workspace_root: Path) -> set[str]:
    envs_dir = workspace_root / ".pixi" / "envs"
    if not envs_dir.is_dir():
        return set()
    installed_environment_names: set[str] = set()
    for candidate in envs_dir.iterdir():
        if candidate.is_dir():
            installed_environment_names.add(candidate.name)
    return installed_environment_names


def _discover_launcher_install_environments(workspace_root: Path) -> list[str]:
    pixi_toml_path = workspace_root / "pixi.toml"
    with pixi_toml_path.open("rb") as pixi_file:
        manifest = tomllib.load(pixi_file)

    environments_table = manifest.get("environments")
    if not isinstance(environments_table, dict):
        raise ValueError("pixi.toml does not define [environments]")

    launcher_environment_names: list[str] = []
    for environment_name, environment_spec in environments_table.items():
        if not isinstance(environment_name, str):
            raise ValueError("Environment names in [environments] must be strings")
        if not isinstance(environment_spec, dict):
            continue

        features = environment_spec.get("features")
        if not isinstance(features, list):
            continue
        if LAUNCHER_RUNTIME_FEATURE in features:
            launcher_environment_names.append(environment_name)

    if not launcher_environment_names:
        raise ValueError(
            f"No launcher runtime environments found. "
            f"Add feature '{LAUNCHER_RUNTIME_FEATURE}' to at least one [environments] entry."
        )
    return launcher_environment_names


def _install_workspace_environments(
    pixi_executable: str,
    workspace_root: Path,
    environment_names: list[str],
    *,
    status_window: _StatusWindow | None = None,
) -> bool:
    install_command = [pixi_executable, "install"]
    for environment_name in environment_names:
        install_command.extend(["-e", environment_name])

    completed = _run_subprocess_with_status(
        install_command,
        cwd=workspace_root,
        check=False,
        status_title="f8studio",
        status_message=SPLASH_RUNTIME_INSTALL_MESSAGE,
        status_window=status_window,
    )
    if completed.returncode != 0:
        return False
    installed_environment_names = _installed_pixi_environment_names(workspace_root)
    return all(environment_name in installed_environment_names for environment_name in environment_names)


def _install_pixi(*, status_window: _StatusWindow | None = None) -> bool:
    if os.name == "nt":
        install_cmd = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-c",
            "irm -useb https://pixi.sh/install.ps1 | iex",
        ]
    else:
        install_cmd = ["sh", "-c", "wget -qO- https://pixi.sh/install.sh | sh"]

    completed = _run_subprocess_with_status(
        install_cmd,
        check=False,
        status_title="f8studio",
        status_message=SPLASH_PIXI_INSTALL_MESSAGE,
        status_window=status_window,
    )
    if completed.returncode != 0:
        return False
    return _find_pixi_executable() is not None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    launcher_dir = _launcher_dir()

    try:
        workspace_root = _find_workspace_root(launcher_dir)
    except FileNotFoundError as exc:
        _show_error_dialog(
            "f8studio launcher",
            f"Failed to locate pixi.toml for launcher startup.\n{exc}",
        )
        return 4

    workspace_version: str | None = None
    try:
        workspace_version = _read_workspace_version(workspace_root)
    except (FileNotFoundError, tomllib.TOMLDecodeError, OSError):
        workspace_version = None

    splash_window: _StatusWindow | None = None
    if not args.dry_run:
        splash_window = _create_status_window(
            title="f8studio",
            message=SPLASH_BOOT_MESSAGE,
            version_text=workspace_version,
        )

    pixi_executable = _find_pixi_executable()
    if pixi_executable is None:
        if not _install_pixi(status_window=splash_window):
            _close_status_window(splash_window)
            _show_error_dialog(
                "f8studio launcher",
                "Pixi is required but installation failed.\n"
                f"Please install Pixi manually: {PIXI_INSTALL_DOCS_URL}",
            )
            return 2
        pixi_executable = _find_pixi_executable()
        if pixi_executable is None:
            _close_status_window(splash_window)
            _show_error_dialog(
                "f8studio launcher",
                "Pixi install completed but executable was not found.\n"
                f"Please install Pixi manually: {PIXI_INSTALL_DOCS_URL}",
            )
            return 2

    try:
        launcher_environment_names = _discover_launcher_install_environments(workspace_root)
    except (FileNotFoundError, tomllib.TOMLDecodeError, ValueError) as exc:
        _close_status_window(splash_window)
        _show_error_dialog(
            "f8studio launcher",
            f"Failed to read launcher runtime environments from pixi.toml.\n{exc}",
        )
        return 4

    if STUDIO_RUNTIME_ENVIRONMENT not in launcher_environment_names:
        _close_status_window(splash_window)
        _show_error_dialog(
            "f8studio launcher",
            "Failed to read launcher runtime environments from pixi.toml.\n"
            f"Required environment '{STUDIO_RUNTIME_ENVIRONMENT}' is missing.",
        )
        return 4

    installed_environment_names = _installed_pixi_environment_names(workspace_root)
    missing_environments = [
        environment_name
        for environment_name in launcher_environment_names
        if environment_name not in installed_environment_names
    ]
    install_command = [pixi_executable, "install"]
    for environment_name in missing_environments:
        install_command.extend(["-e", environment_name])
    if not 1 <= args.port <= 65535:
        _close_status_window(splash_window)
        _show_error_dialog("f8studio launcher", f"Invalid Studio port: {args.port}")
        return 4
    app_url = f"http://{STUDIO_DEFAULT_HOST}:{args.port}"
    health_url = f"{app_url}/api/health"
    launch_command = [
        pixi_executable,
        "run",
        "-e",
        STUDIO_RUNTIME_ENVIRONMENT,
        "studio_server",
        "--host",
        STUDIO_DEFAULT_HOST,
        "--port",
        str(args.port),
    ]
    should_install_environments = len(missing_environments) > 0
    if args.dry_run:
        print("workspace:", workspace_root)
        print("environments_installed:", not should_install_environments)
        print("launcher_environments:", ", ".join(launcher_environment_names))
        if should_install_environments:
            print("missing_environments:", ", ".join(missing_environments))
        if should_install_environments:
            print("install_command:", " ".join(install_command))
        print("launch_command:", " ".join(launch_command))
        return 0

    if should_install_environments and not _install_workspace_environments(
        pixi_executable,
        workspace_root,
        missing_environments,
        status_window=splash_window,
    ):
        _close_status_window(splash_window)
        _show_error_dialog(
            "f8studio launcher",
            "Pixi environments are missing and installation failed.\n"
            f"Command: {' '.join(install_command)}",
        )
        return 3

    if splash_window is not None:
        splash_window.set_message(SPLASH_LAUNCH_MESSAGE)

    launch_proc = _start_subprocess(launch_command, cwd=workspace_root, env=os.environ.copy())
    launch_returncode = _complete_startup_splash(
        splash_window,
        launch_proc=launch_proc,
        min_visible_s=SPLASH_MIN_VISIBLE_S,
        fade_duration_s=SPLASH_FADE_DURATION_S,
        health_url=health_url,
        app_url=app_url,
        open_browser=not args.no_browser,
    )
    if launch_returncode is not None and launch_returncode != 0:
        _show_error_dialog(
            "f8studio launcher",
            f"Failed to start Studio.\nCommand: {' '.join(launch_command)}\nExit code: {launch_returncode}",
        )
        return int(launch_returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
