"""Remove reproducible development output without touching sources or model data."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRECTORIES = frozenset({
    'build', 'dist', 'node_modules', '__pycache__', '.pytest_cache', '.ruff_cache',
    '.import_linter_cache', '.cache', '.ccache-tmp', '.tmp', 'CMakeFiles',
    'test-results', 'playwright-report',
})
GENERATED_FILES = frozenset({
    'CMakeCache.txt', 'CMakeUserPresets.json', 'cmake_install.cmake',
    'install_manifest.txt', 'CTestTestfile.cmake',
})
INPUT_DIRECTORIES = frozenset({'.git', '.sdk', '.platform', '.media-dependency'})


def candidates(root: Path, *, environments: bool) -> list[Path]:
    directories = set(GENERATED_DIRECTORIES)
    if environments:
        directories.add('.pixi')
    selected: list[Path] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            if path.is_symlink() or path.name in INPUT_DIRECTORIES:
                continue
            if path.is_dir():
                if path.name in directories or path.name.endswith('.egg-info'):
                    selected.append(path)
                elif path.name not in {'.pixi', '.codex', '.vscode', 'resources'}:
                    pending.append(path)
            elif path.name in GENERATED_FILES or path.suffix == '.tsbuildinfo':
                selected.append(path)
    for relative in ('runtime', 'site', 'config/release-tools', 'cloud/.wrangler/tmp'):
        path = root / relative
        if path.is_dir():
            selected = [item for item in selected if not item.is_relative_to(path)]
            selected.append(path)
    embedded_web = root / 'extensions/f8webstudio/f8studio_server/f8studio_server/web_dist'
    if embedded_web.is_dir():
        selected = [item for item in selected if not item.is_relative_to(embedded_web)]
        selected.extend(path for path in embedded_web.iterdir() if path.name != '.gitkeep')
    return sorted(selected)


def clean(root: Path, *, environments: bool, dry_run: bool) -> None:
    paths = candidates(root, environments=environments)
    if not dry_run and any(Path(sys.prefix).resolve().is_relative_to(path.resolve()) for path in paths):
        raise ValueError('Run environment cleanup with host Python, outside the environments being removed')
    # Validate every candidate before deleting any. Query from its parent so Git
    # also checks tracked files inside source submodules.
    for path in paths:
        tracked = subprocess.run(['git', '-C', str(path.parent), 'ls-files', '--', path.name],
                                 check=True, capture_output=True, text=True).stdout
        if tracked.strip():
            raise ValueError(f'Refusing to remove tracked content: {path}')
    for path in paths:
        print(('Would remove ' if dry_run else 'Remove ') + str(path.relative_to(root)), flush=True)
        if not dry_run:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environments', action='store_true', help='Also remove local Pixi environments; use host Python')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    clean(ROOT, environments=args.environments, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
