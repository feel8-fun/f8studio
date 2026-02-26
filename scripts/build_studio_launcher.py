from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Final


DEFAULT_APP_NAME: Final[str] = "f8studio"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Studio launcher executable with Nuitka.")
    parser.add_argument("--name", default=DEFAULT_APP_NAME, help="Executable base name.")
    parser.add_argument("--icon-ico", default="assets/icon.ico", help="Source ICO icon path.")
    parser.add_argument("--dist", default="build/dist", help="Nuitka output directory.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if an up-to-date binary exists.")
    return parser


def _run_nuitka(*, repo_root: Path, app_name: str, icon_ico: Path | None, dist_dir: Path) -> int:
    entry_script = repo_root / "scripts" / "f8studio_launcher.py"

    cmd = [
        sys.executable,
        "-m",
        "nuitka",
        "--onefile",
        "--static-libpython=no",
        "--assume-yes-for-downloads",
        f"--output-dir={dist_dir}",
        f"--output-filename={app_name}",
        str(entry_script),
    ]
    if sys.platform.startswith("win"):
        cmd.extend(["--windows-console-mode=disable"])
        if icon_ico is not None:
            cmd.append(f"--windows-icon-from-ico={icon_ico}")

    nuitka_cache_dir = repo_root / "build" / ".nuitka-cache"
    nuitka_cache_dir.mkdir(parents=True, exist_ok=True)
    ccache_tmp_dir = repo_root / "build" / ".ccache-tmp"
    ccache_tmp_dir.mkdir(parents=True, exist_ok=True)
    command_env = os.environ.copy()
    command_env["NUITKA_CACHE_DIR"] = str(nuitka_cache_dir)
    command_env["CCACHE_TEMPDIR"] = str(ccache_tmp_dir)

    proc = subprocess.run(cmd, cwd=repo_root, check=False, env=command_env)
    return int(proc.returncode)


def _binary_path(*, dist_dir: Path, app_name: str) -> Path:
    if sys.platform.startswith("win"):
        return dist_dir / f"{app_name}.exe"
    return dist_dir / app_name


def _latest_input_mtime(*, repo_root: Path, icon_ico: Path | None) -> float:
    inputs = [
        repo_root / "scripts" / "f8studio_launcher.py",
        repo_root / "scripts" / "build_studio_launcher.py",
    ]
    if icon_ico is not None:
        inputs.append(icon_ico)
    return max(path.stat().st_mtime for path in inputs)


def _should_reuse_existing_binary(
    *, repo_root: Path, dist_dir: Path, app_name: str, icon_ico: Path | None, force: bool
) -> bool:
    if force:
        return False

    executable_path = _binary_path(dist_dir=dist_dir, app_name=app_name)
    if not executable_path.is_file():
        return False

    latest_input_mtime = _latest_input_mtime(repo_root=repo_root, icon_ico=icon_ico)
    return executable_path.stat().st_mtime >= latest_input_mtime


def _fix_linux_ncurses_stub() -> None:
    if not sys.platform.startswith("linux"):
        return

    env_prefix = Path(sys.executable).resolve().parent.parent
    lib_dir = env_prefix / "lib"
    stub_path = lib_dir / "libncursesw.so"
    real_soname_path = lib_dir / "libncursesw.so.6"
    if not stub_path.exists() or not real_soname_path.exists():
        return

    if stub_path.is_symlink():
        return

    if stub_path.stat().st_size > 256:
        return

    stub_text = stub_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not stub_text.startswith("INPUT("):
        return

    stub_path.unlink()
    stub_path.symlink_to(real_soname_path.name)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent

    dist_dir = (repo_root / args.dist).resolve()
    dist_dir.mkdir(parents=True, exist_ok=True)

    icon_ico: Path | None = None
    if sys.platform.startswith("win"):
        icon_ico = (repo_root / args.icon_ico).resolve()
        if not icon_ico.exists():
            raise FileNotFoundError(f"Icon ICO not found: {icon_ico}")

    app_name = str(args.name)
    if _should_reuse_existing_binary(
        repo_root=repo_root, dist_dir=dist_dir, app_name=app_name, icon_ico=icon_ico, force=bool(args.force)
    ):
        print(f"Reusing existing launcher: {_binary_path(dist_dir=dist_dir, app_name=app_name)}")
        return 0

    _fix_linux_ncurses_stub()

    return _run_nuitka(
        repo_root=repo_root,
        app_name=app_name,
        icon_ico=icon_ico,
        dist_dir=dist_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
