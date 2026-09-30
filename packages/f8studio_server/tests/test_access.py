from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.responses import HTMLResponse

from f8studio_server.access import StudioAccess, StudioAccessMiddleware, client_access_token


def access_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(StudioAccessMiddleware, access=StudioAccess('test-secret', frozenset({'http://localhost:8210'})))

    @app.get('/')
    def index() -> HTMLResponse:
        return HTMLResponse('studio')

    @app.get('/api/secret')
    def secret() -> dict[str, bool]:
        return {'ok': True}

    @app.websocket('/api/events')
    async def events(socket: WebSocket) -> None:
        await socket.accept()
        await socket.send_json({'ok': True})
        await socket.close()

    return app


def test_access_requires_token_and_exact_origin() -> None:
    with TestClient(access_app(), base_url='http://localhost:8210') as client:
        assert client.get('/api/secret').status_code == 401
        assert client.get('/api/secret', headers={'Authorization': 'Bearer test-secret'}).status_code == 200
        for origin in ('http://localhost:8211', 'https://localhost:8210', 'http://evil.test:8210'):
            assert client.get('/api/secret', headers={'Authorization': 'Bearer test-secret', 'Origin': origin}).status_code == 403
        assert client.post('/api/auth/session', headers={'Authorization': 'Bearer test-secret',
                                                       'Origin': 'http://localhost:8210'}).status_code == 200
        assert client.get('/api/secret').status_code == 200
        with client.websocket_connect('ws://localhost:8210/api/events', headers={'Origin': 'http://localhost:8210'}) as socket:
            assert socket.receive_json() == {'ok': True}


def test_loopback_bootstrap_and_remote_login_do_not_expose_token() -> None:
    with TestClient(access_app(), base_url='http://localhost:8210', client=('127.0.0.1', 40000)) as local:
        response = local.get('/')
        assert response.text == 'studio'
        assert 'HttpOnly' in response.headers['set-cookie']
        assert local.get('/api/secret').status_code == 200
    with TestClient(access_app(), base_url='http://localhost:8210', client=('192.0.2.1', 40000)) as remote:
        response = remote.get('/')
        assert 'test-secret' not in response.text
        assert 'set-cookie' not in response.headers
        assert remote.get('/api/secret').status_code == 401
    with TestClient(access_app(), base_url='http://localhost:8210', client=('127.0.0.1', 40000)) as cross_site:
        response = cross_site.get('/', headers={'Sec-Fetch-Site': 'cross-site'})
        assert 'set-cookie' not in response.headers


def test_access_file_is_private_and_client_credentials_are_origin_scoped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv('F8STUDIO_DATA_DIR', str(tmp_path))
    monkeypatch.delenv('F8STUDIO_ACCESS_TOKEN', raising=False)
    access = StudioAccess.create(tmp_path, origins=('http://localhost:8210',))
    assert (tmp_path / 'access.json').stat().st_mode & 0o777 == 0o600
    assert client_access_token('http://localhost:8210/api') == access.token
    assert client_access_token('http://localhost:8211') is None
    assert client_access_token('https://example.com') is None
