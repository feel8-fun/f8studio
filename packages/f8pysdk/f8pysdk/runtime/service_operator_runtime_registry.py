from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ..generated import F8RuntimeNode
from .service_runtime_node import ServiceRuntimeNode


OperatorFactory = Callable[[str, F8RuntimeNode, dict[str, Any]], ServiceRuntimeNode]
ServiceFactory = Callable[[str, F8RuntimeNode, dict[str, Any]], ServiceRuntimeNode]


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
        # Singleton instance accessor.
        if not hasattr(ServiceOperatorRuntimeRegistry, "_instance"):
            ServiceOperatorRuntimeRegistry._instance = ServiceOperatorRuntimeRegistry()
        return ServiceOperatorRuntimeRegistry._instance
    

    def __init__(self) -> None:
        self._by_service_operator: dict[str, dict[str, OperatorFactory]] = {}
        self._by_service_service: dict[str, ServiceFactory] = {}

    def services(self) -> list[str]:
        keys = set(self._by_service_operator.keys())
        keys.update(self._by_service_service.keys())
        return sorted(keys)

    def ensure_service(self, service_class: str) -> dict[str, OperatorFactory]:
        service_class = str(service_class or "").strip()
        if not service_class:
            raise ValueError("service_class must be non-empty")
        reg = self._by_service_operator.get(service_class)
        if reg is None:
            reg = {}
            self._by_service_operator[service_class] = reg
        return reg

    def register(
        self,
        service_class: str,
        operator_class: str,
        factory: OperatorFactory,
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

    def register_service(
        self,
        service_class: str,
        factory: ServiceFactory,
        *,
        overwrite: bool = False,
    ) -> None:
        service_class = str(service_class or "").strip()
        if not service_class:
            raise ValueError("service_class must be non-empty")
        if service_class in self._by_service_service and not overwrite:
            raise OperatorAlreadyRegistered(f"service runtime already registered for {service_class}")
        self._by_service_service[service_class] = factory

    def create(
        self,
        *,
        node_id: str,
        node: F8RuntimeNode,
        initial_state: dict[str, Any] | None = None,
    ) -> ServiceRuntimeNode:
        service_class = node.serviceClass
        if not service_class:
            raise ValueError("node.serviceClass must be non-empty")

        operator_class = node.operatorClass
        if operator_class is None:
            factory = self._by_service_service.get(service_class)
            if factory is None:
                if service_class not in self._by_service_operator and service_class not in self._by_service_service:
                    raise ServiceNotRegistered(service_class)
                return ServiceRuntimeNode(node_id=str(node_id))
            return factory(str(node_id), node, dict(initial_state or {}))

        reg = self._by_service_operator.get(service_class)
        if reg is None:
            if service_class not in self._by_service_service:
                raise ServiceNotRegistered(service_class)
            return ServiceRuntimeNode(node_id=str(node_id))

        factory = reg.get(str(operator_class))
        if factory is None:
            return ServiceRuntimeNode(node_id=str(node_id))
        return factory(str(node_id), node, dict(initial_state or {}))

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
