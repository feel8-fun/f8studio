from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.verify_dist import verify_distribution


@pytest.mark.parametrize('skip_gpu', [True, False])
def test_gpu_install_is_explicitly_optional(tmp_path: Path, skip_gpu: bool) -> None:
    (tmp_path / 'pixi.toml').write_text('''
[environments.studio-runtime]
features = ["studio"]
[environments.onnx]
features = ["onnx"]
[feature.studio]
[feature.onnx.pypi-dependencies]
f8pydl = { path = "wheels/f8pydl.whl" }
''')
    (tmp_path / 'wheels').mkdir()
    (tmp_path / 'wheels/f8pydl.whl').touch()
    with patch('scripts.verify_dist.subprocess.run') as run:
        verify_distribution(tmp_path, skip_gpu_install=skip_gpu)
    commands = [call.args[0] for call in run.call_args_list]
    assert any(command[1:3] == ['lock', '--check'] for command in commands)
    assert any('onnx' in command for command in commands) is not skip_gpu
    assert any('studio_launch' in command for command in commands) is skip_gpu


def test_skipping_gpu_still_requires_its_wheels(tmp_path: Path) -> None:
    (tmp_path / 'pixi.toml').write_text('''
[environments.onnx]
features = ["onnx"]
[feature.onnx.pypi-dependencies]
f8pydl = { path = "wheels/missing.whl" }
''')
    with patch('scripts.verify_dist.subprocess.run'), pytest.raises(ValueError, match='Invalid release dependency'):
        verify_distribution(tmp_path, skip_gpu_install=True)
