from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from ..generated import F8RuntimeNode
from .service_runtime_node import OperatorRuntimeNode, RuntimeNode, ServiceNodeRuntimeNode


OperatorFactory = Callable[[str, F8RuntimeNode, dict[str, Any]], OperatorRuntimeNode]
ServiceFactory = Callable[[str, F8RuntimeNode, dict[str, Any]], ServiceNodeRuntimeNode]


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
        # Optional: service/operator specs for `--describe` style discovery.
        # Kept generic to avoid importing pydantic models on import of this module.
        self._service_specs: dict[str, Any] = {}
        self._operator_specs: dict[str, dict[str, Any]] = {}

    def services(self) -> list[str]:
        keys = set(self._by_service_operator.keys())
        keys.update(self._by_service_service.keys())
        keys.update(self._service_specs.keys())
        keys.update(self._operator_specs.keys())
        return sorted(keys)

    # ---- specs (optional) ----------------------------------------------
    def register_service_spec(self, spec: Any, *, overwrite: bool = False) -> None:
        """
        Register a `F8ServiceSpec` for discovery / `--describe`.
        """
        service_class = str(getattr(spec, "serviceClass", "") or "").strip()
        if not service_class:
            raise ValueError("spec.serviceClass must be non-empty")
        if service_class in self._service_specs and not overwrite:
            raise OperatorAlreadyRegistered(f"service spec already registered for {service_class}")
        self._service_specs[service_class] = spec

    def register_operator_spec(self, spec: Any, *, overwrite: bool = False) -> None:
        """
        Register a `F8OperatorSpec` for discovery / `--describe`.
        """
        service_class = str(getattr(spec, "serviceClass", "") or "").strip()
        operator_class = str(getattr(spec, "operatorClass", "") or "").strip()
        if not service_class:
            raise ValueError("spec.serviceClass must be non-empty")
        if not operator_class:
            raise ValueError("spec.operatorClass must be non-empty")
        reg = self._operator_specs.get(service_class)
        if reg is None:
            reg = {}
            self._operator_specs[service_class] = reg
        if operator_class in reg and not overwrite:
            raise OperatorAlreadyRegistered(f"operator spec already registered for {service_class}/{operator_class}")
        reg[operator_class] = spec

    def service_spec(self, service_class: str) -> Any | None:
        return self._service_specs.get(str(service_class or "").strip())

    def operator_specs(self, service_class: str) -> list[Any]:
        reg = self._operator_specs.get(str(service_class or "").strip()) or {}
        return list(reg.values())

    def describe(self, service_class: str) -> Any:
        """
        Build a `F8ServiceDescribe` payload for the given serviceClass.
        """
        service_class = str(service_class or "").strip()
        if not service_class:
            raise ValueError("service_class must be non-empty")
        service = self._service_specs.get(service_class)
        if service is None:
            raise ServiceNotRegistered(service_class)
        operators = list((self._operator_specs.get(service_class) or {}).values())
        # Lazy import to keep runtime module light by default.
        from ..generated import F8ServiceDescribe  # type: ignore[import-not-found]

        return F8ServiceDescribe(service=service, operators=operators)

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
    ) -> RuntimeNode:
        service_class = node.serviceClass
        if not service_class:
            raise ValueError("node.serviceClass must be non-empty")

        operator_class = node.operatorClass
        if operator_class is None:
            factory = self._by_service_service.get(service_class)
            if factory is None:
                if service_class not in self._by_service_operator and service_class not in self._by_service_service:
                    raise ServiceNotRegistered(service_class)
                return ServiceNodeRuntimeNode(node_id=str(node_id))
            return factory(str(node_id), node, dict(initial_state or {}))

        reg = self._by_service_operator.get(service_class)
        if reg is None:
            if service_class not in self._by_service_service:
                raise ServiceNotRegistered(service_class)
            return OperatorRuntimeNode(node_id=str(node_id))

        factory = reg.get(str(operator_class))
        if factory is None:
            return OperatorRuntimeNode(node_id=str(node_id))
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
