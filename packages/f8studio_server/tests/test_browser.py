from threading import Event, Thread
from unittest import mock

import pytest
import uvicorn

from f8studio_server.browser import open_browser_when_ready, run_server


def test_browser_opens_once_only_after_server_startup() -> None:
    server = uvicorn.Server(uvicorn.Config("unused:app"))
    stopped = Event()
    opened = Event()
    with mock.patch("f8studio_server.browser.webbrowser.open", side_effect=lambda url: opened.set() or True) as browser:
        thread = Thread(target=open_browser_when_ready, args=(server, "http://127.0.0.1:8210", stopped))
        thread.start()
        try:
            assert not opened.wait(0.15)
            server.started = True
            assert opened.wait(2)
        finally:
            stopped.set()
            thread.join(timeout=2)
        browser.assert_called_once_with("http://127.0.0.1:8210")
        assert not thread.is_alive()


def test_failed_startup_does_not_open_browser_or_leave_waiter() -> None:
    server = uvicorn.Server(uvicorn.Config("unused:app"))
    with (
        mock.patch.object(server, "run", side_effect=RuntimeError("bind failed")),
        mock.patch("f8studio_server.browser.webbrowser.open") as browser,
    ):
        with pytest.raises(RuntimeError, match="bind failed"):
            run_server(server, browser_url="http://127.0.0.1:8210")
        browser.assert_not_called()


def test_headless_startup_does_not_create_browser_thread() -> None:
    server = uvicorn.Server(uvicorn.Config("unused:app"))
    with mock.patch.object(server, "run") as run, mock.patch("f8studio_server.browser.Thread") as thread:
        run_server(server)
        run.assert_called_once_with()
        thread.assert_not_called()


def test_browser_failure_is_logged_without_stopping_server(caplog: pytest.LogCaptureFixture) -> None:
    server = uvicorn.Server(uvicorn.Config("unused:app"))
    server.started = True
    with mock.patch("f8studio_server.browser.webbrowser.open", side_effect=OSError("no desktop")):
        open_browser_when_ready(server, "http://127.0.0.1:8210", Event())
    assert "Open http://127.0.0.1:8210 manually" in caplog.text
    assert "OSError: no desktop" in caplog.text


def test_browser_is_opened_after_real_http_socket_is_ready() -> None:
    import socket
    from threading import Timer
    from urllib.request import urlopen

    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/ready")
    def ready() -> dict[str, bool]:
        return {"ready": True}

    # Reserve the port until Uvicorn takes ownership, avoiding a free-port race.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    responses: list[bytes] = []

    def open_browser(url: str) -> bool:
        try:
            with urlopen(url + "/ready", timeout=2) as response:
                responses.append(response.read())
        finally:
            server.should_exit = True
        return True

    def stop_server() -> None:
        server.should_exit = True

    # Fail boundedly if readiness notification regresses.
    deadline = Timer(5, stop_server)
    original_run = server.run
    deadline.start()
    try:
        with (
            mock.patch.object(server, "run", side_effect=lambda: original_run(sockets=[listener])),
            mock.patch("f8studio_server.browser.webbrowser.open", side_effect=open_browser),
        ):
            run_server(server, browser_url=f"http://127.0.0.1:{port}")
    finally:
        deadline.cancel()
        listener.close()
    assert responses == [b'{"ready":true}']
