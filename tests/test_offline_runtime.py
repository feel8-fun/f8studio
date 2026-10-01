import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts.offline_runtime import rewrite_base_services, tool
from scripts.verify_dist import verify_distribution


@pytest.mark.parametrize('windows', [True, False])
def test_base_services_use_bundled_python_and_optional_services_are_not_active(tmp_path: Path, windows: bool) -> None:
    (tmp_path / 'pixi.toml').write_text('''
[environments.studio-runtime]
features = ["engine"]
[feature.engine.tasks]
engine = "python -m f8pyengine.main"
''')
    services = []
    for name, env in [('engine', 'studio-runtime'), ('detector', 'onnx'), ('pose', 'mediapipe')]:
        path = tmp_path / 'config/services' / name / 'service.yml'
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump({'launch': {'command': 'pixi', 'args': ['run', '-e', env, name]}}))
        services.append({'serviceClass': name, 'manifests': {'any': f'services/{name}/service.yml'}})
    (tmp_path / 'config/service-index.json').write_text(json.dumps({'services': services}))
    (tmp_path / 'config/extensions.json').write_text(json.dumps({
        'schemaVersion': 'f8extensionCatalog/1', 'preinstalled': ['engine', 'detector', 'pose'],
        'extensions': [{'extensionId': name, 'serviceClasses': [name]} for name in ('engine', 'detector', 'pose')],
    }))
    rewrite_base_services(tmp_path, windows=windows)
    entry = yaml.safe_load((tmp_path / 'config/services/engine/service.yml').read_text())['launch']
    assert entry['command'] == ('./env/python.exe' if windows else './env/bin/python')
    assert entry['args'] == ['-I', '-m', 'f8pyengine.main']
    assert entry['workdir'] == '${F8_PACKAGE_ROOT}'
    assert len(json.loads((tmp_path / 'config/service-index.json').read_text())['services']) == 3
    catalog = json.loads((tmp_path / 'config/extensions.json').read_text())
    assert catalog['preinstalled'] == ['engine']
    assert catalog['extensions'][0]['runtime'] == {'kind': 'bundled'}
    assert catalog['extensions'][1]['runtime'] == {'kind': 'pixi', 'environment': 'onnx'}
    assert catalog['extensions'][2]['runtime'] == {'kind': 'pixi', 'environment': 'mediapipe'}
    rewrite_base_services(tmp_path, windows=windows, preset='core')
    assert json.loads((tmp_path / 'config/extensions.json').read_text())['preinstalled'] == []


def test_cached_tool_with_wrong_digest_is_rejected(tmp_path: Path) -> None:
    from scripts.offline_runtime import VERSION
    with patch('scripts.offline_runtime.os.name', 'posix'):
        asset = tmp_path / VERSION / 'pixi-pack-x86_64-unknown-linux-musl'
        asset.parent.mkdir()
        asset.write_bytes(b'corrupted')
        with pytest.raises(ValueError, match='Checksum mismatch'):
            tool('pixi-pack', tmp_path)


def test_verifier_does_not_install_dependencies_and_checks_second_launch(tmp_path: Path) -> None:
    marker = tmp_path / '.runtime-location'
    marker.write_text(str(tmp_path))
    with patch('scripts.verify_dist.subprocess.run') as run:
        verify_distribution(tmp_path)
    commands = [call.args[0] for call in run.call_args_list]
    assert len(commands) == 3
    assert commands[0] == commands[1]
    assert all('pixi' not in command and 'install' not in command for command in commands)
    assert all('PYTHONPATH' not in call.kwargs['env'] for call in run.call_args_list)
