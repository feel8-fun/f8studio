from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import socket

import pytest

from f8studio_server.errors import ConflictError
from f8studio_server.environments import EnvironmentManager
from f8studio_server.extensions import ExtensionManager
from f8studio_server.extension_tools import ExtensionTools, ToolRunRequest


def test_diagnostics_install_execution_and_disable_use_generic_extension_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / 'package'
    (source / 'config').mkdir(parents=True)
    (source / 'env').mkdir()
    manifest = json.loads((Path(__file__).resolve().parents[3] / 'extensions/f8diagnostics/extension.json').read_text())
    # Isolate runtime provisioning; preserve the actual commands, fields, and tool IDs.
    manifest['extensions'][0]['runtime'] = {'kind': 'bundled'}
    (source / 'config/extensions.json').write_text(json.dumps(manifest))
    (source / 'pixi.toml').write_text('[workspace]\nname="diagnostics-test"\n[environments]\nstudio-runtime={features=[]}\n')
    (source / 'pixi.lock').write_text('fixture lock')
    monkeypatch.setattr(EnvironmentManager, '_python', staticmethod(lambda _root: Path(sys.executable)))
    manager = ExtensionManager(tmp_path / 'data', base_index=source / 'config/service-index.json')
    tools = ExtensionTools(manager, tmp_path / 'jobs')
    assert tools.list() == ()

    async def exercise() -> None:
        await manager.install('diagnostics', lambda: None)
        assert manager._task is not None
        await manager._task
        assert manager.status('diagnostics').state == 'installed'
        assert {tool.tool_id for tool in tools.list()} == {'skeleton-verify', 'skeleton-simulate'}
        job = tools.submit('diagnostics', 'skeleton-simulate', ToolRunRequest(
            arguments={'port': 0}, confirm=True))
        await tools._tasks[job.job_id]
        result = tools.get(job.job_id)
        assert result.status == 'failed'
        assert result.result is not None and 'port' in result.result.message
        assert 'Traceback' in result.log
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            receiver.bind(('127.0.0.1', 0))
            receiver.settimeout(1)
            simulation = tools.submit('diagnostics', 'skeleton-simulate', ToolRunRequest(
                arguments={'port': receiver.getsockname()[1], 'durationSeconds': 0.1, 'fps': 10}, confirm=True))
            await tools._tasks[simulation.job_id]
            assert tools.get(simulation.job_id).status == 'succeeded'
            payload = json.loads(receiver.recvfrom(65535)[0])
            assert payload['modelName'] == 'F8 diagnostic skeleton'
            assert len(payload['bones']) == 2
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        verification = tools.submit('diagnostics', 'skeleton-verify', ToolRunRequest(
            arguments={'port': port, 'timeoutMs': 2000, 'minimumFrames': 2}))
        continuous = tools.submit('diagnostics', 'skeleton-simulate', ToolRunRequest(arguments={'port': port}, confirm=True))
        try:
            await asyncio.wait_for(tools._tasks[verification.job_id], timeout=5)
            assert tools.get(verification.job_id).status == 'succeeded'
            assert tools.get(continuous.job_id).status == 'running'
            assert tools.running('diagnostics')
            with pytest.raises(ConflictError, match='running'):
                tools.submit('diagnostics', 'skeleton-simulate', ToolRunRequest(arguments={'port': port}, confirm=True))
            with pytest.raises(ConflictError, match='running'):
                await manager.set_enabled('diagnostics', False, lambda: None, lambda _name: False)
        finally:
            stopped = await tools.cancel(continuous.job_id)
        assert stopped.status == 'cancelled'
        assert not tools.running('diagnostics')
        await manager.set_enabled('diagnostics', False, lambda: None, lambda _name: False)
        assert tools.list() == ()
        await manager.uninstall('diagnostics', lambda: None, lambda _name: False)
        assert tools.get(job.job_id).status == 'failed'
        await tools.close()
    asyncio.run(exercise())
