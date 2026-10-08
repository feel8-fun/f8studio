from pathlib import Path
import subprocess

import pytest

from scripts.workspace_inputs import PythonWorkspace, prepare, workspace_lock

WORKSPACE = PythonWorkspace('extensions/fixture', ('runtime',))


def _git(directory: Path, *arguments: str) -> str:
    return subprocess.check_output(['git', '-C', str(directory), *arguments], text=True).strip()


def _commit(source: Path) -> str:
    _git(source, 'add', '.')
    _git(source, '-c', 'user.name=Test', '-c', 'user.email=test@example.com', 'commit', '-m', 'SDK')
    return _git(source, 'rev-parse', 'HEAD')


def _checkout(root: Path) -> tuple[Path, Path]:
    source = root / 'sdk'
    source.mkdir()
    _git(source, 'init')
    (source / 'sdk.txt').write_text('old SDK')
    _commit(source)
    destination = root / WORKSPACE.path / '.sdk'
    subprocess.run(['git', 'clone', str(source), str(destination)], check=True, capture_output=True)
    return source, destination


def test_prepare_updates_clean_git_checkout_from_local_workspace(tmp_path: Path) -> None:
    source, destination = _checkout(tmp_path)
    (source / 'sdk.txt').write_text('persistent and publishable')
    revision = _commit(source)
    prepare(tmp_path, workspaces=(WORKSPACE,))
    assert _git(destination, 'rev-parse', 'HEAD') == revision
    assert not _git(destination, 'status', '--porcelain')
    assert (destination / 'sdk.txt').read_text() == 'persistent and publishable'


def test_prepare_preserves_unknown_local_changes_before_any_sync(tmp_path: Path) -> None:
    source, destination = _checkout(tmp_path)
    (source / 'sdk.txt').write_text('new SDK')
    _commit(source)
    (destination / 'sdk.txt').write_text('local changes')
    first = PythonWorkspace('platform', ('platform-runtime',))
    with pytest.raises(ValueError, match='Local SDK changes'):
        prepare(tmp_path, workspaces=(first, WORKSPACE))
    assert not (tmp_path / 'platform/.sdk').exists()
    assert (destination / 'sdk.txt').read_text() == 'local changes'


def test_prepare_mirrors_uncommitted_sdk_edits_and_deletions(tmp_path: Path) -> None:
    source, destination = _checkout(tmp_path)
    # An initial working-tree deletion must also reach a clean clone.
    (source / 'sdk.txt').unlink()
    (source / 'new.txt').write_text('new uncommitted SDK source')
    prepare(tmp_path, workspaces=(WORKSPACE,))
    assert not (destination / 'sdk.txt').exists()
    assert (destination / 'new.txt').read_text() == 'new uncommitted SDK source'
    # A later edit to the root SDK replaces only the tool's previous contents.
    (source / 'new.txt').write_text('next SDK source')
    prepare(tmp_path, workspaces=(WORKSPACE,))
    assert (destination / 'new.txt').read_text() == 'next SDK source'
    # Committing those changes must advance the clone without discarding user work.
    revision = _commit(source)
    prepare(tmp_path, workspaces=(WORKSPACE,))
    assert _git(destination, 'rev-parse', 'HEAD') == revision
    assert not _git(destination, 'status', '--porcelain')
    (destination / 'new.txt').write_text('extension developer edits')
    with pytest.raises(ValueError, match='Local SDK changes'):
        prepare(tmp_path, workspaces=(WORKSPACE,))
    assert (destination / 'new.txt').read_text() == 'extension developer edits'


def test_prepare_does_not_discard_independent_sdk_commits(tmp_path: Path) -> None:
    _, destination = _checkout(tmp_path)
    (destination / 'local.txt').write_text('local committed work')
    revision = _commit(destination)
    with pytest.raises(ValueError, match='independent commits'):
        prepare(tmp_path, workspaces=(WORKSPACE,))
    assert _git(destination, 'rev-parse', 'HEAD') == revision
    assert (destination / 'local.txt').read_text() == 'local committed work'


def test_prepare_refreshes_copies_removes_deleted_files_and_keeps_unchanged_timestamps(tmp_path: Path) -> None:
    source = tmp_path / 'sdk/python/f8pysdk/specs.py'
    source.parent.mkdir(parents=True)
    source.write_text('old SDK')
    obsolete = source.with_name('obsolete.py')
    obsolete.write_text('removed SDK module')
    prepare(tmp_path, workspaces=(WORKSPACE,))
    destination = tmp_path / WORKSPACE.path / '.sdk/python/f8pysdk/specs.py'
    unchanged_mtime = destination.stat().st_mtime_ns
    prepare(tmp_path, workspaces=(WORKSPACE,))
    assert destination.stat().st_mtime_ns == unchanged_mtime
    source.write_text('persistent and publishable')
    obsolete.unlink()
    prepare(tmp_path, workspaces=(WORKSPACE,))
    assert destination.read_text() == source.read_text()
    assert not destination.with_name('obsolete.py').exists()
    destination.write_text('local SDK work')
    with pytest.raises(ValueError, match='Local SDK changes'):
        prepare(tmp_path, workspaces=(WORKSPACE,))


def test_preparation_lock_rejects_concurrent_writer_and_releases_after_failure(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='test failure'):
        with workspace_lock(tmp_path):
            with pytest.raises(RuntimeError, match='Another workspace preparation'):
                with workspace_lock(tmp_path):
                    pytest.fail('Concurrent preparation must not acquire the lock')
            raise ValueError('test failure')
    with workspace_lock(tmp_path):
        assert (tmp_path / 'build/workspace/preparation.lock').is_file()
