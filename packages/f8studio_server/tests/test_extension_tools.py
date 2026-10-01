from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import pytest

from f8studio_server.agents.skills import AgentSkillLibrary
from f8studio_server.environments import EnvironmentManager
from f8studio_server.errors import ConflictError, InvalidRequestError
from f8studio_server.extension_tools import ExtensionTools, ToolRunRequest
from f8studio_server.extensions import ExtensionManager


def fixture(root: Path, monkeypatch: pytest.MonkeyPatch, *, script: str | None = None) -> tuple[ExtensionManager, ExtensionTools]:
    source = root / 'package'
    (source / 'config').mkdir(parents=True)
    (source / 'env').mkdir()
    (source / 'skills').mkdir()
    (source / 'skills/SKILL.md').write_text('Use the example tool and inspect its result.')
    (source / 'profile.json').write_text('{"profile":"example"}')
    (source / 'tool.py').write_text(script or (
        'import json,sys\nrequest=json.load(sys.stdin)\nprint("diagnostic", file=sys.stderr)\n'
        'print(json.dumps({"schemaVersion":"f8toolResult/1","success":True,"message":"Done","data":request["arguments"]}))\n'
    ))
    (source / 'pixi.toml').write_text('[workspace]\nname="test"\n[environments]\nstudio-runtime={features=[]}\n')
    (source / 'pixi.lock').write_text('fixture lock')
    (source / 'config/extensions.json').write_text(json.dumps({
        'schemaVersion': 'f8extensionCatalog/1', 'extensions': [{
            'extensionId': 'example', 'name': 'Example', 'version': '1.0.0', 'description': 'Tool-only package',
            'runtime': {'kind': 'bundled'},
            'tools': [{'toolId': 'inspect', 'name': 'Inspect', 'description': 'Inspect a target', 'command': 'python',
                'args': ['${F8_PACKAGE_ROOT}/tool.py'], 'fields': [{'name': 'target', 'label': 'Target', 'required': True}]}],
            'skills': [{'skillId': 'workflow', 'path': '${F8_PACKAGE_ROOT}/skills/SKILL.md'}],
            'resources': [{'resourceId': 'profile', 'path': '${F8_PACKAGE_ROOT}/profile.json'}],
        }],
    }))
    monkeypatch.setattr(EnvironmentManager, '_python', staticmethod(lambda _root: Path(sys.executable)))
    manager = ExtensionManager(root / 'data', base_index=source / 'config/service-index.json')
    tools = ExtensionTools(manager, root / 'jobs')
    return manager, tools


async def install(manager: ExtensionManager) -> None:
    await manager.install('example', lambda: None)
    assert manager._task is not None
    await manager._task
    assert manager.status('example').state == 'installed'


def test_tool_only_lifecycle_skills_resources_and_persistent_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, tools = fixture(tmp_path, monkeypatch)
    skills = AgentSkillLibrary(user_root=tmp_path / 'user-skills', extension_files=manager.active_skill_files)
    assert tools.list() == ()
    assert 'example:workflow' not in skills.list()

    async def exercise() -> None:
        await install(manager)
        assert manager.status('example').service_classes == ()
        assert [tool.tool_id for tool in tools.list()] == ['inspect']
        assert skills.read('example:workflow').startswith('Use the example')
        assert tools.read_resource('example', 'profile').content == '{"profile":"example"}'
        with pytest.raises(InvalidRequestError, match='confirmation'):
            tools.submit('example', 'inspect', ToolRunRequest(arguments={'target': '/games/test'}))
        with pytest.raises(InvalidRequestError, match='required'):
            tools.submit('example', 'inspect', ToolRunRequest(arguments={}, confirm=True))
        job = tools.submit('example', 'inspect', ToolRunRequest(arguments={'target': '/games/test'}, confirm=True))
        with pytest.raises(ConflictError, match='running'):
            await manager.set_enabled('example', False, lambda: None, lambda _name: False)
        await tools._tasks[job.job_id]
        result = tools.get(job.job_id)
        assert result.status == 'succeeded'
        assert result.result is not None and result.result.data == {'target': '/games/test'}
        assert 'diagnostic' in result.log
        reloaded = ExtensionTools(manager, tmp_path / 'jobs')
        assert reloaded.get(job.job_id) == result
        await reloaded.close()
        await manager.set_enabled('example', False, lambda: None, lambda _name: False)
        assert tools.list() == ()
        assert 'example:workflow' not in skills.list()
        await manager.uninstall('example', lambda: None, lambda _name: False)
        assert tools.get(job.job_id).status == 'succeeded'
        await tools.close()
    asyncio.run(exercise())


@pytest.mark.parametrize('script', [
    'raise RuntimeError("intentional failure")',
    'print("not json")',
    'import sys;sys.stdout.write("x" * (2 * 1024 * 1024))',
])
def test_tool_failures_have_actionable_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    manager, tools = fixture(tmp_path, monkeypatch, script=script)
    async def exercise() -> None:
        await install(manager)
        job = tools.submit('example', 'inspect', ToolRunRequest(arguments={'target': 'test'}, confirm=True))
        await tools._tasks[job.job_id]
        failed = tools.get(job.job_id)
        assert failed.status == 'failed' and failed.error
        assert 'Traceback' in failed.log
        await tools.close()
    asyncio.run(exercise())


def test_cancellation_terminates_process_and_preserves_history(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, tools = fixture(tmp_path, monkeypatch, script='import time;time.sleep(60)')
    async def exercise() -> None:
        await install(manager)
        job = tools.submit('example', 'inspect', ToolRunRequest(arguments={'target': 'test'}, confirm=True))
        await asyncio.sleep(0.1)
        cancelled = await tools.cancel(job.job_id)
        assert cancelled.status == 'cancelled'
        assert not tools.running('example')
        await tools.close()
    asyncio.run(exercise())


def test_declared_capability_cannot_escape_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _manager, _tools = fixture(tmp_path, monkeypatch)
    catalog = tmp_path / 'package/config/extensions.json'
    raw = json.loads(catalog.read_text())
    raw['extensions'][0]['skills'][0]['path'] = '${F8_PACKAGE_ROOT}/../../secret'
    catalog.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='escapes'):
        ExtensionManager(tmp_path / 'other-data', base_index=tmp_path / 'package/config/service-index.json')


def test_imported_skill_package_needs_no_service_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib
    import io
    import zipfile
    from f8studio_server.extension_models import ExtensionImportRequest
    manager, tools = fixture(tmp_path, monkeypatch)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('config/extensions.json', json.dumps({
            'schemaVersion': 'f8extensionCatalog/1', 'extensions': [{
                'extensionId': 'knowledge', 'name': 'Knowledge', 'version': '1.0', 'description': 'Reference package',
                'skills': [{'skillId': 'howto', 'path': '${F8_PACKAGE_ROOT}/skills/SKILL.md'}],
            }],
        }))
        archive.writestr('skills/SKILL.md', 'Use verified profiles before installing.')
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    cache = manager._root / 'packages'
    cache.mkdir(parents=True)
    (cache / f'{digest}.zip').write_bytes(payload)
    async def exercise() -> None:
        await manager.import_package(ExtensionImportRequest(url='https://example.test/knowledge.zip', sha256=digest))
        await manager.install('knowledge', lambda: None)
        assert manager._task is not None
        await manager._task
        assert manager.status('knowledge').state == 'installed'
        library = AgentSkillLibrary(user_root=tmp_path / 'skills', extension_files=manager.active_skill_files)
        assert library.read('knowledge:howto').startswith('Use verified profiles')
        assert manager.status('knowledge').service_classes == ()
        await manager.uninstall('knowledge', lambda: None, lambda _name: False)
        assert 'knowledge:howto' not in library.list()
        await tools.close()
    asyncio.run(exercise())
