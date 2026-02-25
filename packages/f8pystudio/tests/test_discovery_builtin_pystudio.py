from __future__ import annotations

from f8pystudio.constants import SERVICE_CLASS as STUDIO_SERVICE_CLASS
from f8pystudio.service_catalog.discovery import load_discovery_into_registries
from f8pystudio.service_catalog.service_catalog import ServiceCatalog


def _reset_service_catalog() -> ServiceCatalog:
    catalog = ServiceCatalog.instance()
    catalog.services.clear()
    catalog.operators.clear()
    catalog._service_entry_paths.clear()
    return catalog


def test_discovery_injects_builtin_pystudio_without_service_yml() -> None:
    catalog = _reset_service_catalog()

    found = load_discovery_into_registries(roots=[])

    assert STUDIO_SERVICE_CLASS in found
    assert catalog.services.has(STUDIO_SERVICE_CLASS)
    assert all(op.serviceClass == STUDIO_SERVICE_CLASS for op in catalog.operators.all())
    assert catalog.service_entry_path(STUDIO_SERVICE_CLASS) is None


def test_discovery_builtin_pystudio_injection_is_idempotent() -> None:
    catalog = _reset_service_catalog()

    first_found = load_discovery_into_registries(roots=[])
    first_operators = [op for op in catalog.operators.all() if op.serviceClass == STUDIO_SERVICE_CLASS]

    second_found = load_discovery_into_registries(roots=[])
    second_operators = [op for op in catalog.operators.all() if op.serviceClass == STUDIO_SERVICE_CLASS]

    assert STUDIO_SERVICE_CLASS in first_found
    assert STUDIO_SERVICE_CLASS in second_found
    assert len(first_operators) > 0
    assert len(second_operators) == len(first_operators)
