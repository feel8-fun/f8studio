"""Run inside an extension environment to verify its imported SDK source."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import f8media_protocol
import f8pysdk


def package_files(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob('*')) if path.is_file() and '__pycache__' not in path.parts
            and path.suffix not in {'.pyc', '.pyo'}}


def verify(source: Path) -> None:
    assert f8pysdk.__file__ is not None
    assert f8media_protocol.__file__ is not None
    packages = (
        ('f8pysdk', Path(f8pysdk.__file__).parent),
        ('f8media_protocol', Path(f8media_protocol.__file__).parent),
    )
    for name, installed in packages:
        expected = package_files(source / name)
        actual = package_files(installed)
        different = sorted(key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key))
        if not expected or different:
            raise ValueError(f'Stale installed {name}: {installed}; differs from {source / name}: {different[:8]}')
    print(f'SDK verified: {f8pysdk.__file__}', flush=True)


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
