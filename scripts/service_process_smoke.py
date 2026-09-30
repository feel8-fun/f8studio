from __future__ import annotations

import argparse
import json
import queue
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from f8pysdk.codec import decode_obj, encode_obj
from f8pysdk.f8_naming import new_id, svc_endpoint_key
from f8pysdk.service_runtime_tools.deploy import ServiceProcessConfig, ServiceProcessManager
from f8pysdk.service_runtime_tools.inventory.catalog import ServiceCatalog
from f8pysdk.service_runtime_tools.inventory.discovery import load_discovery_into_catalog
from f8pysdk.specs import F8EmptyArgs, F8StatusRequest
from f8pysdk.time_utils import now_ms
from f8pysdk.zenoh_config import apply_zenoh_shared_memory_config, apply_zenoh_timestamping_config
from f8pysdk.zenoh_shutdown import close_zenoh_session_best_effort


@dataclass(frozen=True)
class SmokeTarget:
    service_id: str
    service_class: str


@dataclass(frozen=True)
class MonitorEvent:
    source: str
    key: str
    kind: str
    payload: bytes


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    key: str
    detail: str
    payload: bytes = b""


@dataclass
class SmokeState:
    service_id: str
    min_ready_ts_ms: int
    live_key: str = ""
    ready_payload: dict[str, Any] | None = None
    status_payload: dict[str, Any] | None = None
    last_probe_detail: str = ""
    process_exited_early: bool = False
    output_tail: list[str] = field(default_factory=list)

    @property
    def live_ok(self) -> bool:
        return bool(self.live_key)

    @property
    def ready_ok(self) -> bool:
        return self.ready_payload is not None

    @property
    def status_ok(self) -> bool:
        return self.status_payload is not None

    @property
    def ok(self) -> bool:
        return self.live_ok and self.ready_ok and self.status_ok and not self.process_exited_early


DEFAULT_TARGETS: tuple[SmokeTarget, ...] = (
    SmokeTarget(service_id="smokeengine", service_class="f8.pyengine"),
    SmokeTarget(service_id="smokedetector", service_class="f8.dl.detector"),
    SmokeTarget(service_id="smokeopticalflow", service_class="f8.dl.optflow"),
    SmokeTarget(service_id="smokedetsorter", service_class="f8.dl.detsorter"),
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Start real service.yml-backed service processes and verify Zenoh liveliness, "
            "retained ready status, and status endpoint replies."
        ),
    )
    parser.add_argument(
        "--service",
        action="append",
        default=[],
        metavar="SERVICE_ID=SERVICE_CLASS",
        help=(
            "Service smoke target. Repeatable. "
            "Default: smokeengine=f8.pyengine, smokedetector=f8.dl.detector, "
            "smokeopticalflow=f8.dl.optflow, smokedetsorter=f8.dl.detsorter."
        ),
    )
    parser.add_argument(
        "--discovery-root",
        action="append",
        default=[],
        help="Explicit legacy discovery root. Repeatable. Default: installed service index",
    )
    parser.add_argument("--timeout-s", type=float, default=12.0, help="Per-service smoke timeout.")
    parser.add_argument("--probe-interval-s", type=float, default=0.5)
    parser.add_argument("--probe-timeout-s", type=float, default=0.8)
    parser.add_argument("--max-output-lines", type=int, default=120)
    parser.add_argument(
        "--supervision-mode",
        choices=("studio_owned", "detached"),
        default="studio_owned",
        help="Use studio_owned to match Studio's supervised launch path.",
    )
    parser.add_argument("--zenoh-config", default="", help="Zenoh config file path.")
    parser.add_argument("--zenoh-connect", action="append", default=[], help="Zenoh endpoint to connect to. Repeatable.")
    parser.add_argument("--zenoh-listen", action="append", default=[], help="Zenoh endpoint to listen on. Repeatable.")
    parser.add_argument("--zenoh-shm-pool-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--no-shm", action="store_true", help="Do not force F8's shared-memory Zenoh config knobs.")
    parser.add_argument("--keep-running", action="store_true", help="Leave successfully started services running.")
    return parser.parse_args(argv)


def _parse_service_arg(raw: str) -> SmokeTarget:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("--service must be non-empty")
    if "=" not in text:
        raise ValueError(f"--service must use SERVICE_ID=SERVICE_CLASS, got {text!r}")
    service_id, service_class = text.split("=", 1)
    service_id = service_id.strip()
    service_class = service_class.strip()
    if not service_id:
        raise ValueError(f"--service has empty service id: {text!r}")
    if not service_class:
        raise ValueError(f"--service has empty service class: {text!r}")
    return SmokeTarget(service_id=service_id, service_class=service_class)


def _targets_from_args(args: argparse.Namespace) -> tuple[SmokeTarget, ...]:
    raw_targets = list(args.service or [])
    if not raw_targets:
        return DEFAULT_TARGETS
    return tuple(_parse_service_arg(item) for item in raw_targets)


def _discovery_roots_from_args(args: argparse.Namespace) -> list[Path] | None:
    raw_roots = list(args.discovery_root or [])
    if not raw_roots:
        return None
    roots: list[Path] = []
    for raw_root in raw_roots:
        root = Path(str(raw_root)).expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        else:
            root = root.resolve()
        roots.append(root)
    return roots


def _load_catalog(roots: list[Path] | None) -> ServiceCatalog:
    catalog = ServiceCatalog()
    found = load_discovery_into_catalog(roots=roots, catalog=catalog, builtin_injectors=())
    print(
        f"[smoke] discovery roots={', '.join(str(root) for root in roots) if roots is not None else 'installed index'} services={len(found)}",
        flush=True,
    )
    return catalog


def _build_zenoh_config(args: argparse.Namespace, zenoh_module: Any) -> Any:
    config_path = str(args.zenoh_config or "").strip()
    if config_path:
        config = zenoh_module.Config.from_file(config_path)
    else:
        config = zenoh_module.Config()

    connect = tuple(str(item).strip() for item in args.zenoh_connect if str(item).strip())
    listen = tuple(str(item).strip() for item in args.zenoh_listen if str(item).strip())
    if connect:
        config.insert_json5("connect/endpoints", json.dumps(list(connect)))
    if listen:
        config.insert_json5("listen/endpoints", json.dumps(list(listen)))
    if not bool(args.no_shm):
        apply_zenoh_shared_memory_config(
            config,
            zenoh_module=zenoh_module,
            shm_pool_bytes=max(0, int(args.zenoh_shm_pool_bytes)),
            log_context="service-process-smoke",
        )
    apply_zenoh_timestamping_config(
        config,
        zenoh_module=zenoh_module,
        log_context="service-process-smoke",
    )
    return config


def _sample_kind(zenoh_module: Any, sample: Any) -> str:
    if sample.kind == zenoh_module.SampleKind.PUT:
        return "PUT"
    if sample.kind == zenoh_module.SampleKind.DELETE:
        return "DELETE"
    return str(sample.kind)


def _declare_liveliness_subscriber(
    session: Any,
    zenoh_module: Any,
    service_id: str,
    events: queue.Queue[MonitorEvent],
) -> Any:
    key_expr = f"f8/live/svc/{service_id}/instances/**"

    def _on_sample(sample: Any) -> None:
        events.put(
            MonitorEvent(
                source="live",
                key=str(sample.key_expr),
                kind=_sample_kind(zenoh_module, sample),
                payload=bytes(sample.payload),
            )
        )

    return session.liveliness().declare_subscriber(key_expr, _on_sample, history=True)


def _declare_ready_subscriber(
    session: Any,
    zenoh_module: Any,
    service_id: str,
    events: queue.Queue[MonitorEvent],
) -> Any:
    key_expr = f"f8/svc/{service_id}/status/ready"

    def _on_sample(sample: Any) -> None:
        events.put(
            MonitorEvent(
                source="ready",
                key=str(sample.key_expr),
                kind=_sample_kind(zenoh_module, sample),
                payload=bytes(sample.payload),
            )
        )

    return zenoh_module.ext.declare_advanced_subscriber(
        session,
        key_expr,
        _on_sample,
        history=zenoh_module.ext.HistoryConfig(detect_late_publishers=True, max_samples=1),
    )


def _undeclare(handle: Any, *, label: str) -> None:
    try:
        handle.undeclare()
    except Exception as exc:
        print(f"[smoke] undeclare failed label={label}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def _query_status(session: Any, zenoh_module: Any, service_id: str, timeout_s: float) -> ProbeResult:
    key = svc_endpoint_key(service_id, "status")
    replies: queue.Queue[Any] = queue.Queue()

    def _on_reply(reply: Any) -> None:
        replies.put(reply)

    payload = encode_obj(
        F8StatusRequest(
            reqId=new_id(),
            args=F8EmptyArgs(),
            meta={"actor": "service_process_smoke", "cmd": "status"},
        )
    )
    try:
        session.get(
            key,
            _on_reply,
            payload=payload,
            encoding=zenoh_module.Encoding.APPLICATION_OCTET_STREAM,
            target=zenoh_module.QueryTarget.BEST_MATCHING,
            consolidation=zenoh_module.QueryConsolidation.AUTO,
            timeout=int(max(1.0, float(timeout_s) * 1000.0)),
            congestion_control=zenoh_module.CongestionControl.BLOCK,
            priority=zenoh_module.Priority.INTERACTIVE_HIGH,
            express=True,
        )
    except Exception as exc:
        return ProbeResult(ok=False, key=key, detail=f"{type(exc).__name__}: {exc}")

    deadline_s = time.monotonic() + max(0.02, float(timeout_s)) + 0.05
    error_detail = ""
    while time.monotonic() < deadline_s:
        remaining_s = max(0.001, deadline_s - time.monotonic())
        try:
            reply = replies.get(timeout=min(0.02, remaining_s))
        except queue.Empty:
            continue
        sample = reply.ok
        if sample is not None:
            return ProbeResult(ok=True, key=key, detail="", payload=bytes(sample.payload))
        err = reply.err
        if err is not None:
            try:
                error_detail = bytes(err.payload).decode("utf-8", errors="replace")
            except (TypeError, ValueError, UnicodeError) as exc:
                error_detail = f"error payload decode failed: {type(exc).__name__}: {exc}"
    if error_detail:
        return ProbeResult(ok=False, key=key, detail=error_detail)
    return ProbeResult(ok=False, key=key, detail=f"no reply within {float(timeout_s):g}s")


def _decode_payload(raw: bytes) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        return decode_obj(raw)
    except ValueError:
        return None


def _event_preview(event: MonitorEvent) -> str:
    payload = _decode_payload(event.payload)
    if payload is not None:
        try:
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            return repr(payload)
    if not event.payload:
        return ""
    try:
        return event.payload.decode("utf-8", errors="replace")
    except (TypeError, ValueError, UnicodeError):
        return event.payload.hex()


def _record_event(state: SmokeState, event: MonitorEvent, *, start_s: float) -> None:
    elapsed_s = time.monotonic() - start_s
    preview = _event_preview(event)
    suffix = f" {preview}" if preview else ""
    print(
        f"[{elapsed_s:7.3f}] {event.source:<5} {event.kind:<6} serviceId={state.service_id} key={event.key}{suffix}",
        flush=True,
    )
    if event.source == "live" and event.kind == "PUT":
        state.live_key = event.key
        return
    if event.source != "ready" or event.kind != "PUT":
        return
    payload = _decode_payload(event.payload)
    if payload is None or payload.get("ready") is not True:
        return
    try:
        ts_ms = int(payload.get("ts") or 0)
    except (TypeError, ValueError):
        ts_ms = 0
    if ts_ms < int(state.min_ready_ts_ms):
        print(
            f"[{elapsed_s:7.3f}] ready stale serviceId={state.service_id} ts={ts_ms} minTs={state.min_ready_ts_ms}",
            flush=True,
        )
        return
    state.ready_payload = payload


def _drain_events(state: SmokeState, events: queue.Queue[MonitorEvent], *, start_s: float) -> None:
    while True:
        try:
            event = events.get_nowait()
        except queue.Empty:
            return
        _record_event(state, event, start_s=start_s)


def _make_output_callback(state: SmokeState, *, start_s: float, max_lines: int) -> Any:
    max_tail = max(1, int(max_lines))

    def _on_output(service_id: str, line: str) -> None:
        elapsed_s = time.monotonic() - start_s
        raw_lines = str(line).splitlines()
        if not raw_lines:
            raw_lines = [str(line).rstrip()]
        for raw_line in raw_lines:
            text = str(raw_line).rstrip()
            state.output_tail.append(text)
            if len(state.output_tail) > max_tail:
                del state.output_tail[0 : len(state.output_tail) - max_tail]
            print(f"[{elapsed_s:7.3f}] output serviceId={service_id} {text}", flush=True)

    return _on_output


def _record_probe_result(state: SmokeState, result: ProbeResult, *, start_s: float) -> None:
    elapsed_s = time.monotonic() - start_s
    if not result.ok:
        state.last_probe_detail = result.detail
        print(
            f"[{elapsed_s:7.3f}] probe MISS   serviceId={state.service_id} key={result.key} {result.detail}",
            flush=True,
        )
        return

    payload = _decode_payload(result.payload)
    if payload is None:
        state.last_probe_detail = "reply payload is not msgpack object"
        print(
            f"[{elapsed_s:7.3f}] probe MISS   serviceId={state.service_id} key={result.key} invalid-payload",
            flush=True,
        )
        return
    state.status_payload = payload
    preview = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    print(f"[{elapsed_s:7.3f}] probe OK     serviceId={state.service_id} key={result.key} {preview}", flush=True)


def _print_summary(target: SmokeTarget, state: SmokeState) -> None:
    ready_text = "yes" if state.ready_ok else "no"
    live_text = "yes" if state.live_ok else "no"
    status_text = "yes" if state.status_ok else "no"
    exited_text = "yes" if state.process_exited_early else "no"
    result_text = "PASS" if state.ok else "FAIL"
    print(
        (
            f"[smoke] {result_text} serviceId={target.service_id} serviceClass={target.service_class} "
            f"live={live_text} ready={ready_text} status={status_text} exitedEarly={exited_text}"
        ),
        flush=True,
    )
    if not state.ok and state.last_probe_detail:
        print(f"[smoke] last probe detail serviceId={target.service_id}: {state.last_probe_detail}", flush=True)
    if not state.ok and state.output_tail:
        print(f"[smoke] output tail serviceId={target.service_id}:", flush=True)
        for line in state.output_tail[-20:]:
            print(f"[smoke]   {line}", flush=True)


def _run_target(
    target: SmokeTarget,
    *,
    args: argparse.Namespace,
    manager: ServiceProcessManager,
    zenoh_module: Any,
) -> bool:
    service_id = target.service_id
    service_class = target.service_class
    events: queue.Queue[MonitorEvent] = queue.Queue()
    session = None
    handles: list[tuple[str, Any]] = []
    start_s = time.monotonic()
    state = SmokeState(service_id=service_id, min_ready_ts_ms=int(now_ms()))

    print(
        f"[smoke] START serviceId={service_id} serviceClass={service_class} supervision={args.supervision_mode}",
        flush=True,
    )
    try:
        config = _build_zenoh_config(args, zenoh_module)
        session = zenoh_module.open(config)
        handles.append(("ready", _declare_ready_subscriber(session, zenoh_module, service_id, events)))
        handles.append(("live", _declare_liveliness_subscriber(session, zenoh_module, service_id, events)))

        manager.start(
            ServiceProcessConfig(
                service_class=service_class,
                service_id=service_id,
                supervision_mode=str(args.supervision_mode),
                bus_backend="zenoh",
                zenoh_config_path=str(args.zenoh_config or "").strip() or None,
                zenoh_connect=tuple(str(item).strip() for item in args.zenoh_connect if str(item).strip()),
                zenoh_listen=tuple(str(item).strip() for item in args.zenoh_listen if str(item).strip()),
                zenoh_shm_pool_bytes=max(0, int(args.zenoh_shm_pool_bytes)),
            ),
            on_output=_make_output_callback(
                state,
                start_s=start_s,
                max_lines=int(args.max_output_lines),
            ),
        )

        deadline_s = time.monotonic() + max(0.1, float(args.timeout_s))
        next_probe_s = start_s
        while time.monotonic() < deadline_s:
            _drain_events(state, events, start_s=start_s)
            if state.ok:
                break
            if time.monotonic() >= next_probe_s:
                result = _query_status(session, zenoh_module, service_id, float(args.probe_timeout_s))
                _record_probe_result(state, result, start_s=start_s)
                next_probe_s = time.monotonic() + max(0.05, float(args.probe_interval_s))
            if not manager.is_running(service_id):
                state.process_exited_early = True
                time.sleep(0.25)
                _drain_events(state, events, start_s=start_s)
                break
            time.sleep(0.02)
        _drain_events(state, events, start_s=start_s)
        return state.ok
    except Exception:
        print(f"[smoke] ERROR serviceId={service_id} serviceClass={service_class}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return False
    finally:
        if not bool(args.keep_running):
            stopped = manager.stop(service_id)
            print(f"[smoke] stop serviceId={service_id} stopped={stopped}", flush=True)
            time.sleep(0.15)
            _drain_events(state, events, start_s=start_s)
        _print_summary(target, state)
        for label, handle in reversed(handles):
            _undeclare(handle, label=label)
        if session is not None:
            close_zenoh_session_best_effort(
                session,
                context=f"service-process-smoke:{service_id}",
                native_close=False,
            )


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    try:
        import zenoh  # type: ignore[import-not-found]
    except ImportError as exc:
        print(f"[smoke] import zenoh failed: {exc}", file=sys.stderr)
        return 2

    try:
        targets = _targets_from_args(args)
        catalog = _load_catalog(_discovery_roots_from_args(args))
        missing = [target.service_class for target in targets if catalog.service_entry_path(target.service_class) is None]
        if missing:
            print(f"[smoke] missing service discovery entries: {', '.join(missing)}", file=sys.stderr, flush=True)
            return 2
        manager = ServiceProcessManager(catalog)
        ok_by_target: list[bool] = []
        for target in targets:
            ok_by_target.append(_run_target(target, args=args, manager=manager, zenoh_module=zenoh))
        return 0 if all(ok_by_target) else 1
    except ValueError as exc:
        print(f"[smoke] invalid arguments: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
