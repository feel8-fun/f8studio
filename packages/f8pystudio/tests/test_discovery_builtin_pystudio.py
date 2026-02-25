from __future__ import annotations

from f8pystudio.constants import SERVICE_CLASS as STUDIO_SERVICE_CLASS
from f8pystudio.pystudio_node_registry import register_pystudio_specs
from f8pysdk.service_runtime_tools.catalog import ServiceCatalog
from f8pysdk.service_runtime_tools.discovery import load_discovery_into_catalog


def _inject_builtin_pystudio_specs(catalog: ServiceCatalog) -> str | None:
    registry = register_pystudio_specs()
    service_spec = registry.service_spec(STUDIO_SERVICE_CLASS)
    if service_spec is None:
        return None
    catalog.register_service(service_spec)
    for operator_spec in registry.operator_specs(STUDIO_SERVICE_CLASS):
        catalog.register_operator(operator_spec)
    return str(service_spec.serviceClass)


def _reset_service_catalog() -> ServiceCatalog:
    catalog = ServiceCatalog.instance()
    catalog.clear()
    return catalog


def test_discovery_injects_builtin_pystudio_without_service_yml() -> None:
    catalog = _reset_service_catalog()

    found = load_discovery_into_catalog(
        roots=[],
        catalog=catalog,
        builtin_injectors=(_inject_builtin_pystudio_specs,),
    )

    assert STUDIO_SERVICE_CLASS in found
    assert catalog.services.has(STUDIO_SERVICE_CLASS)
    assert all(op.serviceClass == STUDIO_SERVICE_CLASS for op in catalog.operators.all())
    assert catalog.service_entry_path(STUDIO_SERVICE_CLASS) is None


def test_discovery_builtin_pystudio_injection_is_idempotent() -> None:
    catalog = _reset_service_catalog()

    first_found = load_discovery_into_catalog(
        roots=[],
        catalog=catalog,
        builtin_injectors=(_inject_builtin_pystudio_specs,),
    )
    first_operators = [op for op in catalog.operators.all() if op.serviceClass == STUDIO_SERVICE_CLASS]

    second_found = load_discovery_into_catalog(
        roots=[],
        catalog=catalog,
        builtin_injectors=(_inject_builtin_pystudio_specs,),
    )
    second_operators = [op for op in catalog.operators.all() if op.serviceClass == STUDIO_SERVICE_CLASS]

    assert STUDIO_SERVICE_CLASS in first_found
    assert STUDIO_SERVICE_CLASS in second_found
    assert len(first_operators) > 0
    assert len(second_operators) == len(first_operators)
