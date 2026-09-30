"""Open the browser after Uvicorn has completed startup and bound its sockets."""
from __future__ import annotations

import logging
from threading import Event, Thread
import webbrowser

from uvicorn import Server

logger = logging.getLogger(__name__)


def open_browser_when_ready(server: Server, url: str, stopped: Event) -> None:
    while not stopped.wait(0.1):
        if not server.started:
            continue
        try:
            if not webbrowser.open(url):
                logger.warning("Could not open a browser. Open %s manually.", url)
        except (webbrowser.Error, OSError):
            logger.exception("Could not open a browser. Open %s manually.", url)
        return


def run_server(server: Server, *, browser_url: str | None = None) -> None:
    if browser_url is None:
        server.run()
        return
    stopped = Event()
    browser_thread = Thread(
        target=open_browser_when_ready,
        args=(server, browser_url, stopped),
        name="studio-browser",
        daemon=True,
    )
    browser_thread.start()
    try:
        server.run()
    finally:
        stopped.set()
        browser_thread.join(timeout=1.0)
