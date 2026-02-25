from __future__ import annotations

import argparse
import importlib
import json
import logging
import signal
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
PYSTUDIO_SERVICE_CLASS = "f8.pystudio"


def _load_injector(ref: str) -> Callable[[Any], str | None]:
    text = str(ref or "").strip()
    if ":" not in text:
        raise ValueError(f"injector must be 'module:function', got: {text!r}")
    module_name, function_name = text.split(":", 1)
    module_name = module_name.strip()
    function_name = function_name.strip()
    if not module_name or not function_name:
        raise ValueError(f"invalid injector reference: {text!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, function_name)
    if not callable(fn):
        raise TypeError(f"injector is not callable: {text!r}")
    return fn


async def _deploy_runtime_graph(*, nats_url: str, service_id: str, graph: Any) -> None:
    from f8pysdk.nats_naming import kv_bucket_for_service, new_id, svc_endpoint_subject
    from f8pysdk.nats_transport import NatsTransport, NatsTransportConfig
    from f8pysdk.service_ready import wait_service_ready

    bucket = kv_bucket_for_service(service_id)
    transport = NatsTransport(NatsTransportConfig(url=str(nats_url).strip(), kv_bucket=bucket))
    await transport.connect()
    try:
        await wait_service_ready(transport, timeout_s=6.0)
        graph_payload = graph.model_dump(mode="json", by_alias=True)
        request_payload = {
            "reqId": new_id(),
            "args": {"graph": graph_payload},
            "meta": {"source": "headless"},
        }
        request_bytes = json.dumps(request_payload, ensure_ascii=False, default=str).encode("utf-8")
        response_bytes = await transport.request(
            svc_endpoint_subject(service_id, "set_rungraph"),
            request_bytes,
            timeout=2.0,
            raise_on_error=True,
        )
        if not response_bytes:
            raise RuntimeError(f"set_rungraph failed for serviceId={service_id}: empty response")
        response_payload = json.loads(response_bytes.decode("utf-8"))
        if not isinstance(response_payload, dict):
            raise RuntimeError(f"set_rungraph failed for serviceId={service_id}: invalid response")
        if response_payload.get("ok") is True:
            return
        if response_payload.get("ok") is False:
            error = response_payload.get("error")
            if isinstance(error, dict):
                message = str(error.get("message") or "")
            else:
                message = ""
            raise RuntimeError(message or f"set_rungraph rejected for serviceId={service_id}")
        raise RuntimeError(f"set_rungraph failed for serviceId={service_id}: malformed envelope")
    finally:
        await transport.close()


def _wait_for_terminate() -> None:
    stop_event = threading.Event()

    def _handler(signum: int, _frame: Any) -> None:
        logger.info("received signal %s, shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    stop_event.wait()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Headless runner for f8studio session JSON.")
    parser.add_argument("--session", required=True, help="Path to studio session JSON file.")
    parser.add_argument("--nats-url", default="nats://127.0.0.1:4222", help="NATS server URL.")
    parser.add_argument(
        "--discovery-root",
        action="append",
        default=[],
        help="Service discovery root directory. May be provided multiple times.",
    )
    parser.add_argument(
        "--builtin-injector",
        action="append",
        default=[],
        help="Built-in injector callable reference in form module:function. May be provided multiple times.",
    )
    parser.add_argument(
        "--discovery-live",
        action="store_true",
        help="Disable static describe fast-paths and always run describe subprocesses.",
    )
    parser.add_argument(
        "--no-auto-start",
        action="store_true",
        help="Do not auto-start service processes; only deploy to already-running services.",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable local NATS bootstrap (auto-start/download).",
    )
    return parser


def _ensure_bootstrap_or_raise(*, nats_url: str, bootstrap: bool) -> None:
    if not bool(bootstrap):
        return
    from f8pysdk.nats_server_bootstrap import ensure_nats_server

    ok = ensure_nats_server(str(nats_url), log_cb=lambda line: logger.info("%s", line))
    if not ok:
        raise RuntimeError(f"NATS bootstrap failed for {nats_url}")


def _run_headless(args: argparse.Namespace) -> int:
    from f8pysdk.service_runtime_tools import (
        ServiceCatalog,
        ServiceProcessConfig,
        ServiceProcessManager,
        compile_runtime_graphs_from_session_layout,
        default_discovery_roots,
        load_discovery_into_catalog,
        load_session_layout,
    )

    catalog = ServiceCatalog.instance()
    catalog.clear()

    injectors: list[Callable[[Any], str | None]] = []
    for injector_ref in list(args.builtin_injector or []):
        injectors.append(_load_injector(str(injector_ref)))

    roots = [Path(p).expanduser().resolve() for p in list(args.discovery_root or [])]
    if not roots:
        roots = default_discovery_roots()
    found = load_discovery_into_catalog(roots=roots, catalog=catalog, builtin_injectors=tuple(injectors))
    logger.info("discovery loaded services=%s", len(found))

    layout = load_session_layout(str(args.session))
    compiled = compile_runtime_graphs_from_session_layout(layout=layout, catalog=catalog, pystudio_service_class=PYSTUDIO_SERVICE_CLASS)
    for warning in compiled.warnings:
        logger.info("%s", warning)

    process_manager = ServiceProcessManager(catalog=catalog)
    started_service_ids: list[str] = []
    try:
        if not bool(args.no_auto_start):
            for service in list(compiled.global_graph.services or []):
                service_class = str(service.serviceClass or "")
                if service_class == PYSTUDIO_SERVICE_CLASS:
                    continue
                service_id = str(service.serviceId or "")
                process_manager.start(
                    ServiceProcessConfig(
                        service_class=service_class,
                        service_id=service_id,
                        nats_url=str(args.nats_url),
                    )
                )
                started_service_ids.append(service_id)

        import asyncio

        async def _deploy_all() -> None:
            for service_id, graph in compiled.per_service.items():
                await _deploy_runtime_graph(nats_url=str(args.nats_url), service_id=str(service_id), graph=graph)

        asyncio.run(_deploy_all())
        logger.info("deployment complete, waiting for termination signal")
        _wait_for_terminate()
        return 0
    except ValueError:
        logger.exception("headless validation failed")
        return 2
    except RuntimeError:
        logger.exception("headless runtime failed")
        return 3
    except Exception:
        logger.exception("headless runner failed")
        return 3
    finally:
        for service_id in reversed(started_service_ids):
            try:
                process_manager.stop(service_id)
            except Exception:
                logger.exception("failed to stop service process: %s", service_id)


def main(argv: list[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    args = _build_arg_parser().parse_args(argv)
    if args.discovery_live:
        import os

        os.environ["F8_DISCOVERY_DISABLE_STATIC_DESCRIBE"] = "1"

    try:
        _ensure_bootstrap_or_raise(nats_url=str(args.nats_url), bootstrap=(not bool(args.no_bootstrap)))
        return _run_headless(args)
    except ValueError:
        logger.exception("headless validation failed")
        return 2
    except RuntimeError:
        logger.exception("headless runtime failed")
        return 3
    except Exception:
        logger.exception("headless runner failed")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
