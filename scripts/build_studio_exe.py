from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Final


DEFAULT_APP_NAME: Final[str] = "f8studio"
ICON_SIZES: Final[list[tuple[int, int]]] = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (24, 24), (16, 16)]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Windows exe for f8pystudio with app icon.")
    parser.add_argument("--name", default=DEFAULT_APP_NAME, help="Executable base name.")
    parser.add_argument("--icon-png", default="assets/icon.png", help="Source PNG icon path.")
    parser.add_argument("--icon-ico", default="build/icon.ico", help="Generated ICO path.")
    parser.add_argument("--dist", default="build/dist", help="PyInstaller dist directory.")
    parser.add_argument("--work", default="build/pyinstaller", help="PyInstaller work directory.")
    parser.add_argument("--spec", default="build/pyinstaller", help="PyInstaller spec directory.")
    return parser


def _png_to_ico(icon_png: Path, icon_ico: Path) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required to convert PNG to ICO. Install with: pip install pillow") from exc

    with Image.open(icon_png) as image:
        rgba = image.convert("RGBA")
        icon_ico.parent.mkdir(parents=True, exist_ok=True)
        rgba.save(icon_ico, format="ICO", sizes=ICON_SIZES)


def _run_pyinstaller(*, repo_root: Path, app_name: str, icon_ico: Path, dist_dir: Path, work_dir: Path, spec_dir: Path) -> int:
    entry_script = repo_root / "packages" / "f8pystudio" / "f8pystudio" / "main.py"
    icon_png = repo_root / "assets" / "icon.png"
    add_data_arg = f"{icon_png}{';' if sys.platform.startswith('win') else ':'}assets"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        app_name,
        "--icon",
        str(icon_ico),
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        "--add-data",
        add_data_arg,
        str(entry_script),
    ]

    proc = subprocess.run(cmd, cwd=repo_root, check=False)
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent

    icon_png = (repo_root / args.icon_png).resolve()
    icon_ico = (repo_root / args.icon_ico).resolve()
    if not icon_png.exists():
        raise FileNotFoundError(f"Icon PNG not found: {icon_png}")

    _png_to_ico(icon_png=icon_png, icon_ico=icon_ico)

    dist_dir = (repo_root / args.dist).resolve()
    work_dir = (repo_root / args.work).resolve()
    spec_dir = (repo_root / args.spec).resolve()
    return _run_pyinstaller(
        repo_root=repo_root,
        app_name=str(args.name),
        icon_ico=icon_ico,
        dist_dir=dist_dir,
        work_dir=work_dir,
        spec_dir=spec_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
