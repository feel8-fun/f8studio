"""Synchronize explicitly declared SDK inputs for workspace development."""
from __future__ import annotations

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
IGNORED = frozenset({'.git', '.pixi', '.sdk', '.platform', '.media-dependency', 'build', 'dist',
                     'node_modules', '__pycache__', '.pytest_cache', '.ruff_cache'})


@dataclass(frozen=True)
class PythonWorkspace:
    path: str
    environments: tuple[str, ...]


# Development ownership only; published extensions retain their own SDK pins/locks.
PYTHON_WORKSPACES = (
    PythonWorkspace('platform', ('platform-runtime', 'platform-desktop')),
    PythonWorkspace('extensions/f8mediagateway', ('media',)),
    PythonWorkspace('extensions/f8webstudio', ('webstudio',)),
    PythonWorkspace('extensions/f8pyengine', ('pyengine',)),
    PythonWorkspace('extensions/f8pydl', ('dl',)),
    PythonWorkspace('extensions/f8proclauncher', ('proclauncher',)),
    PythonWorkspace('extensions/f8pymppose', ('mediapipe',)),
    PythonWorkspace('extensions/f8pyaudiofeat', ('audiofeat',)),
    PythonWorkspace('extensions/f8diagnostics', ('diagnostics',)),
)


def source_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for directory, children, filenames in os.walk(root):
        children[:] = sorted(name for name in children if name not in IGNORED and not name.endswith('.egg-info'))
        for name in sorted(filenames):
            path = Path(directory) / name
            if name not in IGNORED and path.suffix not in {'.pyc', '.pyo'}:
                result[path.relative_to(root).as_posix()] = path
    return result


def file_hash(path: Path) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def tree_hash(root: Path) -> str:
    hashes = {name: file_hash(path) for name, path in source_files(root).items()}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def read_hashes(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    raw: object = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError(f'Invalid SDK sync record: {path}')
    values = cast(dict[object, object], raw)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in values.items()):
        raise ValueError(f'Invalid SDK sync record: {path}')
    for key in values:
        assert isinstance(key, str)
        relative = Path(key)
        if relative.is_absolute() or '..' in relative.parts or any(part in IGNORED for part in relative.parts):
            raise ValueError(f'Invalid SDK sync path in {path}: {key}')
    return cast(dict[str, str], values)


def git_output(directory: Path, *arguments: str) -> str:
    return subprocess.check_output(['git', '-C', str(directory), *arguments], text=True).strip()


def sync_record(root: Path, workspace: PythonWorkspace) -> Path:
    return root / 'build/workspace/sdk-inputs' / (workspace.path.replace('/', '_') + '.json')


@contextmanager
def workspace_lock(root: Path) -> Generator[None]:
    path = root / 'build/workspace/preparation.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'\0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another workspace preparation is running. Retry after it completes.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_checkout(destination: Path, previous: dict[str, str]) -> tuple[list[str], list[str]]:
    if git_output(destination, 'diff', '--cached', '--name-only'):
        raise ValueError(f'Local staged SDK changes in {destination}. Preserve them before workspace preparation.')
    tracked = git_output(destination, 'diff', '--name-only', '--no-renames', '-z', 'HEAD').split('\0')
    untracked = git_output(destination, 'ls-files', '--others', '--exclude-standard', '-z').split('\0')
    tracked = [name for name in tracked if name]
    untracked = [name for name in untracked if name]
    for name in tracked + untracked:
        path = destination / name
        actual = file_hash(path) if path.is_file() else ''
        if name not in previous or actual != previous[name]:
            raise ValueError(f'Local SDK changes in {destination}: {name}. Preserve these changes before workspace preparation.')
    return tracked, untracked


def prepare(root: Path = ROOT, *, workspaces: tuple[PythonWorkspace, ...] = PYTHON_WORKSPACES) -> None:
    with workspace_lock(root):
        sync_inputs(root, workspaces=workspaces)


def sync_inputs(root: Path, *, workspaces: tuple[PythonWorkspace, ...]) -> None:
    """Synchronize source inputs while the caller holds workspace_lock."""
    source = root / 'sdk'
    if not source.is_dir():
        raise FileNotFoundError(f'SDK checkout missing: {source}')
    files = source_files(source)
    plans: list[tuple[PythonWorkspace, dict[str, str], list[str], list[str]]] = []
    # Check all user work before changing any SDK input.
    for workspace in workspaces:
        destination = root / workspace.path / '.sdk'
        previous = read_hashes(sync_record(root, workspace))
        tracked: list[str] = []
        untracked: list[str] = []
        if destination.resolve() != source.resolve() and (destination / '.git').exists():
            tracked, untracked = validate_checkout(destination, previous)
            revision = git_output(source, 'rev-parse', 'HEAD')
            current = git_output(destination, 'rev-parse', 'HEAD')
            if current != revision:
                subprocess.run(['git', '-C', str(destination), 'fetch', str(source.resolve()), revision], check=True)
                ancestor = subprocess.run(['git', '-C', str(destination), 'merge-base', '--is-ancestor', current, revision], check=False)
                if ancestor.returncode == 1:
                    raise ValueError(f'SDK checkout has independent commits: {destination}. Preserve them before updating to {revision}.')
                ancestor.check_returncode()
            if not previous:
                previous = {name: file_hash(path) for name, path in source_files(destination).items()}
        elif destination.resolve() != source.resolve() and previous:
            for name, expected in previous.items():
                target = destination / name
                actual = file_hash(target) if target.is_file() else ''
                if actual != expected:
                    raise ValueError(f'Local SDK changes in {target}; preserve them before workspace preparation')
            for name in files.keys() - previous.keys():
                target = destination / name
                if target.is_file() and file_hash(target) != file_hash(files[name]):
                    raise ValueError(f'Local SDK changes in {target}; preserve them before workspace preparation')
        plans.append((workspace, previous, tracked, untracked))
    for workspace, previous, tracked, untracked in plans:
        destination = root / workspace.path / '.sdk'
        if destination.resolve() == source.resolve():
            continue
        if (destination / '.git').exists():
            revision = git_output(source, 'rev-parse', 'HEAD')
            current = git_output(destination, 'rev-parse', 'HEAD')
            if current != revision:
                # Only undo edits whose contents still match our last managed snapshot.
                if tracked:
                    subprocess.run(['git', '-C', str(destination), 'restore', '--worktree', '--', *tracked], check=True)
                for name in untracked:
                    (destination / name).unlink()
                subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', revision], check=True)
        for name in previous.keys() - files.keys():
            target = destination / name
            if target.is_file():
                if file_hash(target) != previous[name]:
                    raise ValueError(f'Local SDK changes in {target}; refusing to remove it')
                target.unlink()
        hashes: dict[str, str] = {}
        for name, path in files.items():
            digest = file_hash(path)
            target = destination / name
            if not target.is_file() or file_hash(target) != digest:
                if name in previous and target.is_file() and file_hash(target) != previous[name]:
                    raise ValueError(f'Local SDK changes in {target}; refusing to overwrite it')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            hashes[name] = digest
        # Include managed deletions so a dirty Git checkout can be checked next time.
        for name in previous.keys() - files.keys():
            hashes[name] = ''
        record = sync_record(root, workspace)
        record.parent.mkdir(parents=True, exist_ok=True)
        temporary = record.with_suffix('.tmp')
        temporary.write_text(json.dumps(hashes, sort_keys=True), encoding='utf-8')
        temporary.replace(record)
        print(f'Prepared {destination.relative_to(root)}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare',))
    parser.parse_args()
    prepare()
