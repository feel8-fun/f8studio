from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ..generated import F8OperatorSpec
from .service_runtime_node import ServiceRuntimeNode


Factory = Callable[[str, F8OperatorSpec, dict[str, Any]], ServiceRuntimeNode]


class RegistryError(Exception):
    """Base class for registry failures."""


class ServiceNotRegistered(RegistryError):
    """Raised when a serviceClass has no runtime registry."""


class OperatorAlreadyRegistered(RegistryError):
    """Raised when an operatorClass is already registered for the same serviceClass."""


class ServiceOperatorRuntimeRegistry:
    """
    Per-service operator runtime registries (push-based).

    Developers register their operator runtime implementations here, typically
    from an importable module (see `load_modules`).
    """

    @staticmethod
    def instance() -> "ServiceOperatorRuntimeRegistry":
        global _GLOBAL_SERVICE_OPERATOR_RUNTIME_REGISTRY
        try:
            return _GLOBAL_SERVICE_OPERATOR_RUNTIME_REGISTRY
        except NameError:
            _GLOBAL_SERVICE_OPERATOR_RUNTIME_REGISTRY = ServiceOperatorRuntimeRegistry()
            return _GLOBAL_SERVICE_OPERATOR_RUNTIME_REGISTRY

    def __init__(self) -> None:
        self._by_service: dict[str, dict[str, Factory]] = {}

    def services(self) -> list[str]:
        return sorted(self._by_service.keys())

    def ensure_service(self, service_class: str) -> dict[str, Factory]:
        service_class = str(service_class or "").strip()
        if not service_class:
            raise ValueError("service_class must be non-empty")
        reg = self._by_service.get(service_class)
        if reg is None:
            reg = {}
            self._by_service[service_class] = reg
        return reg

    def register(
        self,
        service_class: str,
        operator_class: str,
        factory: Factory,
        *,
        overwrite: bool = False,
    ) -> None:
        service_class = str(service_class or "").strip()
        operator_class = str(operator_class or "").strip()
        if not service_class:
            raise ValueError("service_class must be non-empty")
        if not operator_class:
            raise ValueError("operator_class must be non-empty")

        reg = self.ensure_service(service_class)
        if operator_class in reg and not overwrite:
            raise OperatorAlreadyRegistered(f"{operator_class} already registered for {service_class}")

        reg[operator_class] = factory

    def create(
        self,
        *,
        node_id: str,
        spec: F8OperatorSpec,
        initial_state: dict[str, Any] | None = None,
    ) -> ServiceRuntimeNode:
        service_class = str(getattr(spec, "serviceClass", "") or "").strip()
        if not service_class:
            raise ValueError("spec.serviceClass must be non-empty")
        if service_class not in self._by_service:
            raise ServiceNotRegistered(service_class)
        reg = self._by_service[service_class]
        factory = reg.get(str(spec.operatorClass))
        if factory is None:
            return ServiceRuntimeNode(
                node_id=str(node_id),
                data_in_ports=[p.name for p in (spec.dataInPorts or [])],
                data_out_ports=[p.name for p in (spec.dataOutPorts or [])],
                state_fields=[s.name for s in (spec.states or [])],
            )
        return factory(str(node_id), spec, dict(initial_state or {}))

    def load_modules(self, modules: list[str]) -> None:
        """
        Import modules that register runtime implementations.

        A module can register by calling:
        - `ServiceOperatorRuntimeRegistry.instance().register(...)`
        """
        for m in modules:
            name = str(m or "").strip()
            if not name:
                continue
            importlib.import_module(name)
