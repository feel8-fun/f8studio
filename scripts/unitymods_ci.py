from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UNITYMODS_ROOT = REPO_ROOT / "external" / "f8unitymods"
UNITYMODS_MANIFEST = UNITYMODS_ROOT / "pixi.toml"


def _require_submodule() -> None:
    required = (
        UNITYMODS_ROOT / ".git",
        UNITYMODS_MANIFEST,
        UNITYMODS_ROOT / "f8unitymods_setup" / "__init__.py",
        UNITYMODS_ROOT / "src" / "F8SkeletonStreamer" / "F8SkeletonStreamer.csproj",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "f8unitymods submodule is not initialized; run "
            "`git submodule update --init --recursive`. Missing: " + ", ".join(missing)
        )


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(UNITYMODS_ROOT), *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _run_unitymods_task(task: str) -> None:
    if sys.platform != "win32":
        raise RuntimeError(f"f8unitymods task '{task}' requires Windows")
    subprocess.run(
        ["pixi", "run", "--frozen", "-e", "default", "--manifest-path", str(UNITYMODS_MANIFEST), task],
        cwd=UNITYMODS_ROOT,
        check=True,
    )


def validate(*, allow_dirty: bool) -> dict[str, object]:
    _require_submodule()
    commit = _git_output("rev-parse", "HEAD")
    status_lines = [line for line in _git_output("status", "--short").splitlines() if line]
    if status_lines and not allow_dirty:
        raise RuntimeError("f8unitymods submodule has uncommitted changes: " + "; ".join(status_lines))
    return {
        "status": "ok",
        "path": str(UNITYMODS_ROOT),
        "commit": commit,
        "dirty": bool(status_lines),
        "changes": status_lines,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bundle(*, output_dir: Path, build_assets: bool, allow_dirty: bool) -> dict[str, object]:
    validation = validate(allow_dirty=allow_dirty)
    if build_assets:
        _run_unitymods_task("setup-wheel")
        _run_unitymods_task("release-package")

    source_root = UNITYMODS_ROOT / "dist"
    if not source_root.is_dir():
        raise FileNotFoundError(f"f8unitymods dist directory was not found: {source_root}")

    selected: list[Path] = []
    for pattern in ("*.whl", "*.zip", "*manifest*.json"):
        selected.extend(path for path in source_root.rglob(pattern) if path.is_file())
    unique_files = sorted(set(selected), key=lambda path: path.as_posix().lower())
    if not unique_files:
        raise FileNotFoundError(f"No f8unitymods release assets were found under {source_root}")

    duplicate_names = sorted(
        name
        for name in {path.name for path in unique_files}
        if sum(path.name == name for path in unique_files) > 1
    )
    if duplicate_names:
        raise ValueError(
            "f8unitymods release assets must have unique file names; duplicates: "
            + ", ".join(duplicate_names)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    assets: list[dict[str, object]] = []
    for source in unique_files:
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        assets.append(
            {
                "name": destination.name,
                "size": destination.stat().st_size,
                "sha256": _sha256(destination),
                "source": source.relative_to(UNITYMODS_ROOT).as_posix(),
            }
        )

    manifest = {
        "schemaVersion": "f8unitymodsBundle/1",
        "commit": validation["commit"],
        "dirty": validation["dirty"],
        "assets": assets,
    }
    manifest_path = output_dir / "f8unitymods-bundle.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "ok", "output": str(output_dir), **manifest}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate, build, and bundle the pinned f8unitymods submodule.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--allow-dirty", action="store_true")

    subparsers.add_parser("build")
    subparsers.add_parser("test")
    subparsers.add_parser("package")

    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("--output", required=True)
    bundle_parser.add_argument("--skip-build", action="store_true")
    bundle_parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate":
        result = validate(allow_dirty=bool(args.allow_dirty))
    elif args.command == "build":
        _require_submodule()
        _run_unitymods_task("build")
        result = validate(allow_dirty=True)
    elif args.command == "test":
        _require_submodule()
        _run_unitymods_task("test")
        result = validate(allow_dirty=True)
    elif args.command == "package":
        _require_submodule()
        _run_unitymods_task("release-package")
        _run_unitymods_task("setup-wheel")
        result = validate(allow_dirty=True)
    elif args.command == "bundle":
        result = bundle(
            output_dir=Path(args.output).expanduser().resolve(),
            build_assets=not bool(args.skip_build),
            allow_dirty=bool(args.allow_dirty),
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
