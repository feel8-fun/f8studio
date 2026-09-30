from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading

import uvicorn

from .app import create_app


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Feel8 Media Gateway.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8211, type=int)
    parser.add_argument("--report-bound-port", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--exit-on-stdin-close",
        action="store_true",
        help="Exit when the managing process closes the inherited stdin pipe.",
    )
    return parser.parse_args()


def _watch_stdin() -> None:
    try:
        sys.stdin.buffer.read()
    except OSError:
        logger.exception("Media Gateway parent-watch pipe failed")
    logger.info("Media Gateway parent-watch pipe closed; shutting down")
    os.kill(os.getpid(), signal.SIGTERM)


def main() -> None:
    args = _parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Media Gateway only supports loopback hosts")
    if args.exit_on_stdin_close:
        threading.Thread(target=_watch_stdin, name="media-gateway-parent-watch", daemon=True).start()
    config = uvicorn.Config(create_app(), host=args.host, port=args.port, log_level="info")
    if args.report_bound_port:
        # Keep the socket bound throughout handoff; the parent never probes or reserves a port.
        with config.bind_socket() as listener:
            print(listener.getsockname()[1], flush=True)
            uvicorn.Server(config).run(sockets=[listener])
    else:
        uvicorn.Server(config).run()


if __name__ == "__main__":
    main()
