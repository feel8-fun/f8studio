from __future__ import annotations

import concurrent.futures
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from f8pysdk.specs import F8ServiceEntry

from .catalog import ServiceCatalog
from .describe import (
    clear_discovery_errors,
    describe_entry_timed,
    discovery_log_timings_enabled,
    discovery_parallelism,
    discovery_slow_ms_default,
    last_discovery_error_lines,
    last_discovery_timing_lines,
    read_static_describe_payload,
    set_discovery_timing_lines,
)
from .entry import default_discovery_roots, find_service_dirs, load_service_entry
from .policy import (
    DISABLED_SERVICE_CLASSES_ENV,
    merge_disabled_service_classes,
)


logger = logging.getLogger(__name__)
_DISCOVERY_BUILTIN_INJECTOR_ERRORS = (Exception,)
_DISCOVERY_CATALOG_REGISTRATION_ERRORS = (KeyError, TypeError, ValueError)
# Worker exceptions are isolated per service so one bad describe cannot abort discovery.
_DISCOVERY_DESCRIBE_FUTURE_ERRORS = (Exception,)
_DISCOVERY_PATH_REWRITE_ERRORS = (AttributeError, OSError, RuntimeError, TypeError, ValueError)


def load_discovery_into_catalog(
    *,
    roots: list[Path] | None = None,
    overwrite: bool = True,
    catalog: ServiceCatalog | None = None,
    builtin_injectors: Sequence[Callable[[ServiceCatalog], str | None]] = (),
    disabled_service_classes: Sequence[str] | None = None,
) -> list[str]:
    _ = overwrite
    clear_discovery_errors()

    resolved_roots = roots if roots is not None else default_discovery_roots()
    target_catalog = catalog or ServiceCatalog.instance()
    disabled_service_class_set = set(
        merge_disabled_service_classes(explicit_service_classes=disabled_service_classes)
    )

    found: list[str] = []
    entries: list[tuple[Path, F8ServiceEntry]] = []
    for service_dir in find_service_dirs(resolved_roots):
        try:
            entry = load_service_entry(service_dir)
        except ValueError as exc:
            logger.warning("Skipping service in %s: %s", service_dir, exc)
            continue
        entry_service_class = str(entry.serviceClass or "").strip()
        if entry_service_class and entry_service_class in disabled_service_class_set:
            logger.info("Skipping disabled service %s in %s", entry_service_class, service_dir)
            continue
        entries.append((service_dir, entry))

    payload_by_dir: dict[Path, dict[str, Any] | None] = {}
    timing_by_dir: dict[Path, tuple[float, str]] = {}
    subprocess_entries: list[tuple[Path, F8ServiceEntry]] = []
    for service_dir, entry in entries:
        static_payload, static_source = read_static_describe_payload(service_dir, entry)
        if static_payload is None or static_source is None:
            subprocess_entries.append((service_dir, entry))
            continue
        payload, dt_ms, source = describe_entry_timed(service_dir, entry, initial_payload=static_payload, source=static_source)
        payload_by_dir[service_dir] = payload
        timing_by_dir[service_dir] = (dt_ms, source)

    jobs = discovery_parallelism(len(subprocess_entries))
    if os.name == "nt":
        has_pixi_describe = False
        for _service_dir, entry in subprocess_entries:
            try:
                command_name = str(entry.launch.command or "").strip().lower()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Failed to inspect service launch command for pixi discovery guard", exc_info=exc)
                continue
            if command_name in ("pixi", "pixi.exe", "pixi.bat", "pixi.cmd") or Path(command_name).name in (
                "pixi",
                "pixi.exe",
                "pixi.bat",
                "pixi.cmd",
            ):
                has_pixi_describe = True
                break
        if has_pixi_describe and jobs > 1:
            jobs = 1

    if jobs <= 1 or len(subprocess_entries) <= 1:
        for service_dir, entry in subprocess_entries:
            payload, dt_ms, source = describe_entry_timed(service_dir, entry)
            payload_by_dir[service_dir] = payload
            timing_by_dir[service_dir] = (dt_ms, source)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures: dict[
                concurrent.futures.Future[tuple[dict[str, Any] | None, float, str]],
                tuple[Path, F8ServiceEntry],
            ] = {}
            for service_dir, entry in subprocess_entries:
                futures[executor.submit(describe_entry_timed, service_dir, entry)] = (service_dir, entry)
            for future in concurrent.futures.as_completed(futures):
                service_dir, _entry = futures[future]
                try:
                    payload, dt_ms, source = future.result()
                    payload_by_dir[service_dir] = payload
                    timing_by_dir[service_dir] = (dt_ms, source)
                except _DISCOVERY_DESCRIBE_FUTURE_ERRORS as exc:
                    logger.warning("Describe failed for %s: %s", service_dir, exc)
                    payload_by_dir[service_dir] = None
                    timing_by_dir[service_dir] = (0.0, "none")

    if discovery_log_timings_enabled() and timing_by_dir:
        slow_ms = discovery_slow_ms_default()
        rows: list[tuple[float, str, Path]] = []
        for service_dir, entry in entries:
            dt_ms, source = timing_by_dir.get(service_dir, (0.0, "none"))
            label = str(entry.serviceClass or "").strip() or str(service_dir.name)
            rows.append((dt_ms, f"{label} ({source})", service_dir))
        rows.sort(key=lambda item: item[0], reverse=True)

        total_ms = sum(item[0] for item in rows)
        lines: list[str] = [f"service discovery describe timings: services={len(rows)} jobs={jobs} total={total_ms:.1f}ms"]
        for dt_ms, label, service_dir in rows:
            if slow_ms > 0.0 and dt_ms < slow_ms:
                continue
            lines.append(f"{dt_ms:7.1f}ms  {label}  [{service_dir}]")
        errors = last_discovery_error_lines()
        if errors:
            lines.append("")
            lines.append("discovery errors:")
            for error_line in errors:
                lines.append(f"ERROR {error_line}")
        set_discovery_timing_lines(lines)
    else:
        set_discovery_timing_lines([])

    for service_dir, entry in entries:
        payload = payload_by_dir.get(service_dir)
        if payload is None:
            continue
        service_payload_raw = payload.get("service")
        if isinstance(service_payload_raw, dict):
            described_service_class = str(service_payload_raw.get("serviceClass") or "").strip()
            if described_service_class and described_service_class in disabled_service_class_set:
                logger.info("Skipping disabled service %s from %s", described_service_class, service_dir)
                continue
        try:
            service_spec = target_catalog.register_service(
                payload["service"],
                service_entry_path=service_dir,
            )
        except _DISCOVERY_CATALOG_REGISTRATION_ERRORS as exc:
            logger.warning("Failed to register service from %s: %s", service_dir, exc)
            continue
        found.append(str(service_spec.serviceClass))
        try:
            target_catalog.register_operators(payload.get("operators") or [])
        except _DISCOVERY_CATALOG_REGISTRATION_ERRORS as exc:
            logger.warning("Failed to register operators from %s: %s", service_dir, exc)

    for injector in list(builtin_injectors or ()):
        try:
            service_class = injector(target_catalog)
        except _DISCOVERY_BUILTIN_INJECTOR_ERRORS:
            logger.exception("Built-in injector failed: %r", injector)
            continue
        if service_class is not None and service_class not in found:
            found.append(str(service_class))

    return found


def load_discovery_into_registries(
    *,
    roots: list[Path] | None = None,
    overwrite: bool = True,
    catalog: ServiceCatalog | None = None,
    builtin_injectors: Sequence[Callable[[ServiceCatalog], str | None]] = (),
    disabled_service_classes: Sequence[str] | None = None,
) -> list[str]:
    return load_discovery_into_catalog(
        roots=roots,
        overwrite=overwrite,
        catalog=catalog,
        builtin_injectors=builtin_injectors,
        disabled_service_classes=disabled_service_classes,
    )


__all__ = ["DISABLED_SERVICE_CLASSES_ENV", "load_discovery_into_catalog", "load_discovery_into_registries"]
