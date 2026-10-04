"""Executed by the unpacked Python, never the checkout's interpreter."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from importlib import metadata


def main() -> None:
    root = Path(sys.argv[1]).resolve()
    prefix = Path(sys.prefix).resolve()
    assert prefix == root / 'env', prefix
    required = ['f8pysdk', 'f8studio-server', 'f8media-gateway']
    for name in required:
        distribution = metadata.distribution(name)
        assert Path(distribution.locate_file('')).resolve().is_relative_to(prefix), name
        direct = json.loads(distribution.read_text('direct_url.json') or '{}')
        assert not direct.get('dir_info', {}).get('editable'), name
    from fastapi.testclient import TestClient
    from f8media_gateway.service import InProcessMediaGateway
    from f8studio_server.app import create_app, default_web_dist
    from f8studio_server.extensions import ExtensionManager
    from f8pysdk.service_runtime_tools.inventory.index import read_service_index, indexed_entry

    subprocess.run(["basedpyright", "--version"], check=True, timeout=30)
    assert default_web_dist().is_relative_to(prefix)
    index_path = root / 'config/service-index.json'
    count = 0
    with tempfile.TemporaryDirectory() as data:
        manager = ExtensionManager(Path(data), base_index=index_path)
        for installed_index in manager.active_indexes():
            index = read_service_index(installed_index)
            for item in index.services:
                entry = indexed_entry(installed_index, index, item)
                if entry is None:
                    continue
                if Path(entry.launch.command).name in {'pixi', 'pixi.exe'}:
                    raise ValueError(f'Offline service still depends on Pixi: {item.serviceClass}')
                command = [entry.launch.command, *(entry.launch.args or []), '--describe']
                result = subprocess.run(command, cwd=entry.launch.workdir,
                                        env={**os.environ, **(entry.launch.env or {})},
                                        capture_output=True, text=True, check=True, timeout=30)
                assert json.loads(result.stdout)['service']['serviceClass'] == item.serviceClass
                count += 1
        # Exercise an upgrade from the legacy provider schema, not only empty data.
        (Path(data) / 'agent-providers.json').write_text(json.dumps({
            'connection_legacy': {'displayName': 'Legacy', 'model': 'test', 'protocol': 'openai_chat',
                                  'endpoint': 'http://127.0.0.1:1/v1', 'supportsImage': True},
        }))
        app = create_app(data_dir=Path(data), media_gateway=InProcessMediaGateway())
        with TestClient(app) as client:
            assert client.get('/api/health').status_code == 200
            page = client.get('/')
            assert page.status_code == 200 and '<div id="root"></div>' in page.text
            assert len(client.get('/api/extensions').json()) == len(manager.statuses())
            presets = client.get('/api/environments/presets').json()
            assert any(preset['environment'] == 'studio-runtime' and preset['ready'] for preset in presets)
            assert len(client.get('/api/catalog').json()['services']) == count + 1
    print(f'Offline runtime passed: {count} enabled services, installed wheels, health and Web')


if __name__ == '__main__':
    main()
