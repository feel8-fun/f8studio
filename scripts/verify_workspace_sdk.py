"""Run inside an extension environment to verify its imported SDK source."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import f8media_protocol
import f8pysdk

STALE_SDK_EXIT_CODE = 2


def package_files(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob('*')) if path.is_file() and '__pycache__' not in path.parts
            and path.suffix not in {'.pyc', '.pyo'}}


def verify(source: Path) -> bool:
    assert f8pysdk.__file__ is not None
    assert f8media_protocol.__file__ is not None
    packages = (
        ('f8pysdk', Path(f8pysdk.__file__).parent),
        ('f8media_protocol', Path(f8media_protocol.__file__).parent),
    )
    verified = True
    for name, installed in packages:
        expected = package_files(source / name)
        if not expected:
            raise ValueError(f'SDK source is missing or empty: {source / name}')
        actual = package_files(installed)
        different = sorted(key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key))
        if different:
            verified = False
            remaining = f' (+{len(different) - 8} more)' if len(different) > 8 else ''
            print(f'{name} needs updating: {installed}; changed files: {", ".join(different[:8])}{remaining}',
                  file=sys.stderr, flush=True)
    if verified:
        print(f'SDK verified: {f8pysdk.__file__}', flush=True)
    return verified


if __name__ == '__main__':
    raise SystemExit(0 if verify(Path(sys.argv[1])) else STALE_SDK_EXIT_CODE)
