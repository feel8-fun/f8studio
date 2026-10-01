from __future__ import annotations

import argparse
import logging
import os
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


def _watch_stdin(server: uvicorn.Server) -> None:
    try:
        # Raw descriptor reads never hold BufferedReader locks during interpreter shutdown.
        while os.read(sys.stdin.fileno(), 4096):
            continue
    except OSError:
        logger.exception("Media Gateway parent-watch pipe failed")
    logger.info("Media Gateway parent-watch pipe closed; shutting down")
    server.should_exit = True


def main() -> None:
    args = _parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Media Gateway only supports loopback hosts")
    config = uvicorn.Config(create_app(), host=args.host, port=args.port, log_level="info",
                            timeout_graceful_shutdown=5)
    server = uvicorn.Server(config)
    if args.exit_on_stdin_close:
        threading.Thread(target=_watch_stdin, args=(server,), name="media-gateway-parent-watch", daemon=True).start()
    try:
        if args.report_bound_port:
            with config.bind_socket() as listener:
                print(listener.getsockname()[1], flush=True)
                server.run(sockets=[listener])
        else:
            server.run()
    except KeyboardInterrupt:
        logger.info("Media Gateway stopped by user")


if __name__ == "__main__":
    main()
