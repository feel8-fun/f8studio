"""Prepare explicitly declared library inputs for workspace development."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def prepare(root: Path = ROOT) -> None:
    inputs = (
        (root / 'launcher/.sdk', root / 'sdk'),
        (root / 'extensions/f8mediagateway/.sdk', root / 'sdk'),
        (root / 'extensions/f8webstudio/.sdk', root / 'sdk'),
        (root / 'extensions/f8webstudio/.platform', root / 'launcher'),
    )
    for destination, source in inputs:
        if destination.exists():
            continue
        if not source.is_dir():
            raise FileNotFoundError(f'Application dependency checkout missing: {source}')
        # Use the same real directory layout as independent CI checkouts.
        # Pixi canonicalizes symlink targets into external paths in lock files.
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
            '.git', '.pixi', '.sdk', '.platform', '.media-dependency', 'build', 'dist',
            'node_modules', '__pycache__', '*.egg-info', '.pytest_cache', '.ruff_cache'))
        print(f'Prepared {destination.relative_to(root)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare',))
    parser.parse_args()
    prepare()
