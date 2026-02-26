from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PIXI_INSTALL_DOCS_URL = "https://pixi.prefix.dev/latest/installation/"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launcher for f8pystudio via Pixi.")
    parser.add_argument("--dry-run", action="store_true", help="Print command and exit without launching.")
    return parser


def _show_error_dialog(title: str, message: str) -> None:
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()


def _launcher_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    if "__compiled__" in globals():
        return Path(sys.argv[0]).resolve().parent
    return Path(__file__).resolve().parent.parent


def _find_workspace_root(start_dir: Path) -> Path:
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "pixi.toml").is_file():
            return candidate
    raise FileNotFoundError(f"pixi.toml was not found from {start_dir}")


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


def _install_pixi() -> bool:
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

    completed = subprocess.run(install_cmd, check=False)
    if completed.returncode != 0:
        return False
    return _find_pixi_executable() is not None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    workspace_root = _find_workspace_root(_launcher_dir())
    pixi_executable = _find_pixi_executable()
    if pixi_executable is None:
        if not _install_pixi():
            _show_error_dialog(
                "f8studio launcher",
                "Pixi is required but installation failed.\n"
                f"Please install Pixi manually: {PIXI_INSTALL_DOCS_URL}",
            )
            return 2
        pixi_executable = _find_pixi_executable()
        if pixi_executable is None:
            _show_error_dialog(
                "f8studio launcher",
                "Pixi install completed but executable was not found.\n"
                f"Please install Pixi manually: {PIXI_INSTALL_DOCS_URL}",
            )
            return 2

    command = [pixi_executable, "run", "f8pystudio"]
    if args.dry_run:
        print("workspace:", workspace_root)
        print("command:", " ".join(command))
        return 0

    completed = subprocess.run(command, cwd=workspace_root, check=False)
    if completed.returncode != 0:
        _show_error_dialog(
            "f8studio launcher",
            f"Failed to start Studio.\nCommand: {' '.join(command)}\nExit code: {completed.returncode}",
        )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
