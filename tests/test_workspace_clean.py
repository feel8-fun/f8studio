from pathlib import Path
import subprocess

import pytest

from scripts.workspace_clean import candidates, clean


def repository(root: Path) -> None:
    subprocess.run(['git', 'init', '-q', str(root)], check=True)


def test_cleanup_preserves_sources_inputs_models_and_environments(tmp_path: Path) -> None:
    repository(tmp_path)
    for name in ('build/old', '.pixi/envs/build-check', '.sdk/build', 'resources/models', 'src'):
        path = tmp_path / name
        path.mkdir(parents=True)
        (path / 'keep').write_text('content')
    clean(tmp_path, environments=False, dry_run=False)
    assert not (tmp_path / 'build').exists()
    for name in ('.pixi/envs/build-check', '.sdk/build', 'resources/models', 'src'):
        assert (tmp_path / name / 'keep').read_text() == 'content'


def test_cleanup_checks_all_candidates_before_deleting(tmp_path: Path) -> None:
    repository(tmp_path)
    (tmp_path / 'build').mkdir()
    tracked = tmp_path / 'dist/source.txt'
    tracked.parent.mkdir()
    tracked.write_text('source')
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'dist/source.txt'], check=True)
    with pytest.raises(ValueError, match='tracked content'):
        clean(tmp_path, environments=False, dry_run=False)
    assert (tmp_path / 'build').is_dir()
    assert tracked.read_text() == 'source'


def test_cleanup_checks_tracked_files_in_nested_repositories(tmp_path: Path) -> None:
    repository(tmp_path)
    extension = tmp_path / 'extensions/example'
    extension.mkdir(parents=True)
    repository(extension)
    tracked = extension / 'build/fixture.txt'
    tracked.parent.mkdir()
    tracked.write_text('fixture')
    subprocess.run(['git', '-C', str(extension), 'add', 'build/fixture.txt'], check=True)
    with pytest.raises(ValueError, match='tracked content'):
        clean(tmp_path, environments=False, dry_run=False)
    assert tracked.is_file()


def test_environment_removal_is_explicit_and_dry_run_preserves_output(tmp_path: Path) -> None:
    repository(tmp_path)
    (tmp_path / '.pixi').mkdir()
    assert candidates(tmp_path, environments=False) == []
    assert candidates(tmp_path, environments=True) == [tmp_path / '.pixi']
    clean(tmp_path, environments=True, dry_run=True)
    assert (tmp_path / '.pixi').is_dir()
