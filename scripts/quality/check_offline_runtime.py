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
    for name in ('f8pysdk', 'f8pyengine', 'f8studio-server', 'f8media-gateway'):
        distribution = metadata.distribution(name)
        assert Path(distribution.locate_file('')).resolve().is_relative_to(prefix), name
        direct = json.loads(distribution.read_text('direct_url.json') or '{}')
        assert not direct.get('dir_info', {}).get('editable'), name
    from fastapi.testclient import TestClient
    from f8media_gateway.service import InProcessMediaGateway
    from f8studio_server.app import create_app, default_web_dist
    from f8pysdk.service_runtime_tools.inventory.index import read_service_index, indexed_entry

    subprocess.run(["basedpyright", "--version"], check=True, timeout=30)
    assert default_web_dist().is_relative_to(prefix)
    index_path = root / 'config/service-index.json'
    index = read_service_index(index_path)
    for item in index.services:
        entry = indexed_entry(index_path, index, item)
        if entry is None:
            continue
        if entry.launch.command in {'pixi', 'pixi.exe'}:
            raise ValueError(f'Offline service still depends on Pixi: {item.serviceClass}')
        command = [entry.launch.command, *(entry.launch.args or []), '--describe']
        result = subprocess.run(command, cwd=entry.launch.workdir,
                                env={**os.environ, **(entry.launch.env or {})},
                                capture_output=True, text=True, check=True, timeout=30)
        assert json.loads(result.stdout)['service']['serviceClass'] == item.serviceClass
    with tempfile.TemporaryDirectory() as data:
        # Exercise an upgrade from the legacy provider schema, not only empty data.
        (Path(data) / 'agent-providers.json').write_text(json.dumps({
            'connection_legacy': {'displayName': 'Legacy', 'model': 'test', 'protocol': 'openai_chat',
                                  'endpoint': 'http://127.0.0.1:1/v1', 'supportsImage': True},
        }))
        app = create_app(data_dir=Path(data), service_roots=(), media_gateway=InProcessMediaGateway())
        with TestClient(app) as client:
            assert client.get('/api/health').status_code == 200
            page = client.get('/')
            assert page.status_code == 200 and '<div id="root"></div>' in page.text
    print(f'Offline runtime passed: {len(index.services)} services, installed wheels, health and Web')


if __name__ == '__main__':
    main()
