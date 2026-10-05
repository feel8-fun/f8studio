from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.quality.check_cpp_contracts import PRODUCTS, service_executables, validate_executable_paths
from scripts.extension_workspace import workspace_index


@pytest.mark.parametrize('platform,suffix', [('win32', '.exe'), ('linux', '')])
def test_deployed_products_resolve_from_platform_manifests(platform: str, suffix: str) -> None:
    with patch('sys.platform', platform):
        executables = service_executables(workspace_index())
    assert [path.name for path in executables] == [name + suffix for name in PRODUCTS]
    assert all(path.is_absolute() and 'runtime/bundles' in path.as_posix() for path in executables)
    assert all(('win' if platform == 'win32' else 'linux') == path.parent.name for path in executables)


def test_missing_executables_report_all_paths_before_execution(tmp_path: Path) -> None:
    present = tmp_path / 'present.exe'
    present.touch()
    missing = [tmp_path / 'first.exe', tmp_path / 'second.exe']
    with pytest.raises(FileNotFoundError) as caught:
        validate_executable_paths([present, *missing])
    assert all(str(path) in str(caught.value) for path in missing)
    validate_executable_paths([present])
