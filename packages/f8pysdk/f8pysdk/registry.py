from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, Protocol, cast

import msgspec

from ._specs.builtin_fields import (
    upsert_builtin_state_fields_for_operator_spec,
    upsert_builtin_state_fields_for_service_spec,
)
from .generated import F8OperatorSpec, F8RuntimeNode, F8ServiceDescribe, F8ServiceSpec
from .codec import copy_model
from .nodes import OperatorNode, RuntimeNode, ServiceNode


OperatorFactory = Callable[[str, F8RuntimeNode, dict[str, Any]], OperatorNode]
ServiceFactory = Callable[[str, F8RuntimeNode, dict[str, Any]], RuntimeNode]


class RegistryError(Exception):
    """Base class for registry failures."""


class ServiceNotRegistered(RegistryError):
    """Raised when a serviceClass has no runtime registry."""


class ServiceFactoryNotRegistered(RegistryError):
    """Raised when a serviceClass is known but has no service runtime factory."""

    def __init__(self, service_class: str) -> None:
        self.service_class = str(service_class or "").strip()
        super().__init__(f"service runtime factory not registered for {self.service_class}")


class OperatorFactoryNotRegistered(RegistryError):
    """Raised when an operatorClass has no registered runtime factory."""

    def __init__(self, service_class: str, operator_class: str) -> None:
        self.service_class = str(service_class or "").strip()
        self.operator_class = str(operator_class or "").strip()
        super().__init__(
            f"operator runtime factory not registered for {self.service_class}/{self.operator_class}"
        )


class OperatorAlreadyRegistered(RegistryError):
    """Raised when an operatorClass is already registered for the same serviceClass."""


class RuntimeNodeRegistry:
    """
    Per-service runtime node registry.

    This registry stores factories for creating `RuntimeNode`-derived instances
    (service node + operator nodes) for a single `serviceClass`.

    Note: This is a process-local registry used by `ServiceHost` to create nodes
    when applying a rungraph. It is not a network registry.
    """

    _instance: "RuntimeNodeRegistry | None" = None

    @staticmethod
    def instance() -> "RuntimeNodeRegistry":
        if RuntimeNodeRegistry._instance is None:
            RuntimeNodeRegistry._instance = RuntimeNodeRegistry()
        return RuntimeNodeRegistry._instance

    def __init__(self) -> None:
        self._by_service_operator: dict[str, dict[str, OperatorFactory]] = {}
        self._by_service_service: dict[str, ServiceFactory] = {}
        self._service_specs: dict[str, F8ServiceSpec] = {}
        self._operator_specs: dict[str, dict[str, F8OperatorSpec]] = {}

    def services(self) -> list[str]:
        keys = set(self._by_service_operator.keys())
        keys.update(self._by_service_service.keys())
        keys.update(self._service_specs.keys())
        keys.update(self._operator_specs.keys())
        return sorted(keys)

    def _service_known(self, service_class: str) -> bool:
        normalized_service_class = str(service_class or "").strip()
        return (
            normalized_service_class in self._by_service_operator
            or normalized_service_class in self._by_service_service
            or normalized_service_class in self._service_specs
            or normalized_service_class in self._operator_specs
        )

    def register_service_spec(self, spec: F8ServiceSpec, *, overwrite: bool = False) -> None:
        service_class = str(spec.serviceClass or "").strip()
        if not service_class:
            raise ValueError("spec.serviceClass must be non-empty")
        if service_class in self._service_specs and not overwrite:
            raise OperatorAlreadyRegistered(f"service spec already registered for {service_class}")
        self._service_specs[service_class] = spec

    def register_operator_spec(self, spec: F8OperatorSpec, *, overwrite: bool = False) -> None:
        service_class = str(spec.serviceClass or "").strip()
        operator_class = str(spec.operatorClass or "").strip()
        if not service_class:
            raise ValueError("spec.serviceClass must be non-empty")
        if not operator_class:
            raise ValueError("spec.operatorClass must be non-empty")
        registry = self._operator_specs.get(service_class)
        if registry is None:
            registry = {}
            self._operator_specs[service_class] = registry
        if operator_class in registry and not overwrite:
            raise OperatorAlreadyRegistered(f"operator spec already registered for {service_class}/{operator_class}")
        registry[operator_class] = spec

    def service_spec(self, service_class: str) -> F8ServiceSpec | None:
        return self._service_specs.get(str(service_class or "").strip())

    def operator_specs(self, service_class: str) -> list[F8OperatorSpec]:
        registry = self._operator_specs.get(str(service_class or "").strip()) or {}
        return list(registry.values())

    def describe(self, service_class: str) -> F8ServiceDescribe:
        normalized_service_class = str(service_class or "").strip()
        if not normalized_service_class:
            raise ValueError("service_class must be non-empty")
        service_spec = self._service_specs.get(normalized_service_class)
        if service_spec is None:
            raise ServiceNotRegistered(normalized_service_class)
        copied_service_spec = copy_model(service_spec, deep=True)
        copied_operator_specs = [
            copy_model(operator_spec, deep=True)
            for operator_spec in list((self._operator_specs.get(normalized_service_class) or {}).values())
        ]
        self._inject_builtin_state_fields(copied_service_spec, copied_operator_specs)
        return F8ServiceDescribe(service=copied_service_spec, operators=copied_operator_specs)

    @staticmethod
    def _inject_builtin_state_fields(service_spec: F8ServiceSpec, operator_specs: list[F8OperatorSpec]) -> None:
        upsert_builtin_state_fields_for_service_spec(service_spec)
        for operator_spec in list(operator_specs or []):
            upsert_builtin_state_fields_for_operator_spec(operator_spec)

    def ensure_service(self, service_class: str) -> dict[str, OperatorFactory]:
        normalized_service_class = str(service_class or "").strip()
        if not normalized_service_class:
            raise ValueError("service_class must be non-empty")
        registry = self._by_service_operator.get(normalized_service_class)
        if registry is None:
            registry = {}
            self._by_service_operator[normalized_service_class] = registry
        return registry

    def register_operator_factory(
        self,
        service_class: str,
        operator_class: str,
        factory: OperatorFactory,
        *,
        overwrite: bool = False,
    ) -> None:
        normalized_service_class = str(service_class or "").strip()
        normalized_operator_class = str(operator_class or "").strip()
        if not normalized_service_class:
            raise ValueError("service_class must be non-empty")
        if not normalized_operator_class:
            raise ValueError("operator_class must be non-empty")

        registry = self.ensure_service(normalized_service_class)
        if normalized_operator_class in registry and not overwrite:
            raise OperatorAlreadyRegistered(
                f"{normalized_operator_class} already registered for {normalized_service_class}"
            )

        registry[normalized_operator_class] = factory

    def register_service_factory(
        self,
        service_class: str,
        factory: ServiceFactory,
        *,
        overwrite: bool = False,
    ) -> None:
        normalized_service_class = str(service_class or "").strip()
        if not normalized_service_class:
            raise ValueError("service_class must be non-empty")
        if normalized_service_class in self._by_service_service and not overwrite:
            raise OperatorAlreadyRegistered(f"service runtime already registered for {normalized_service_class}")
        self._by_service_service[normalized_service_class] = factory

    def create_service_node(
        self,
        *,
        service_class: str,
        node_id: str,
        initial_state: dict[str, Any] | None = None,
        node: F8RuntimeNode | None = None,
    ) -> RuntimeNode:
        normalized_service_class = str(service_class or "").strip()
        normalized_node_id = str(node_id or "").strip()
        if not normalized_service_class:
            raise ValueError("service_class must be non-empty")
        if not normalized_node_id:
            raise ValueError("node_id must be non-empty")
        if node is None:
            node = F8RuntimeNode(
                nodeId=normalized_node_id,
                serviceId=normalized_node_id,
                serviceClass=normalized_service_class,
                operatorClass=msgspec.UNSET,
            )
        factory = self._by_service_service.get(normalized_service_class)
        if factory is None:
            if not self._service_known(normalized_service_class):
                raise ServiceNotRegistered(normalized_service_class)
            return ServiceNode(node_id=normalized_node_id)
        return factory(normalized_node_id, node, dict(initial_state or {}))

    def create_operator_node(
        self,
        *,
        node_id: str,
        node: F8RuntimeNode,
        initial_state: dict[str, Any] | None = None,
    ) -> OperatorNode:
        service_class = node.serviceClass
        if not service_class:
            raise ValueError("node.serviceClass must be non-empty")

        operator_class = node.operatorClass
        if operator_class is None or isinstance(operator_class, msgspec.UnsetType):
            raise ValueError("node.operatorClass must be set for operator runtime creation")

        registry = self._by_service_operator.get(service_class)
        if registry is None:
            if not self._service_known(str(service_class)):
                raise ServiceNotRegistered(service_class)
            raise OperatorFactoryNotRegistered(str(service_class), str(operator_class))

        factory = registry.get(str(operator_class))
        if factory is None:
            raise OperatorFactoryNotRegistered(str(service_class), str(operator_class))
        created = factory(str(node_id), node, dict(initial_state or {}))
        if not isinstance(created, OperatorNode):
            raise TypeError(
                f"operator factory returned {type(created).__name__}, expected OperatorNode for "
                f"{service_class}/{operator_class}"
            )
        return created

    def create_runtime_node(
        self,
        *,
        node_id: str,
        node: F8RuntimeNode,
        initial_state: dict[str, Any] | None = None,
    ) -> RuntimeNode:
        operator_class = node.operatorClass
        if operator_class is None or isinstance(operator_class, msgspec.UnsetType):
            return self.create_service_node(
                service_class=str(node.serviceClass or ""),
                node_id=str(node_id),
                initial_state=dict(initial_state or {}),
                node=node,
            )
        return self.create_operator_node(
            node_id=str(node_id),
            node=node,
            initial_state=dict(initial_state or {}),
        )



class _ServiceConstructor(Protocol):
    def __call__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode: ...


class _OperatorConstructor(Protocol):
    def __call__(self, *, node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode: ...


ServiceFactoryLike: TypeAlias = ServiceFactory | type[RuntimeNode]
OperatorFactoryLike: TypeAlias = OperatorFactory | type[OperatorNode]


def _coerce_service_factory(factory: ServiceFactoryLike) -> ServiceFactory:
    if isinstance(factory, type):
        if not issubclass(factory, RuntimeNode):
            raise TypeError(f"service factory type must inherit RuntimeNode, got {factory.__name__}")

        def _build(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> RuntimeNode:
            created = cast(_ServiceConstructor, factory)(
                node_id=str(node_id),
                node=node,
                initial_state=dict(initial_state or {}),
            )
            if not isinstance(created, RuntimeNode):
                raise TypeError(f"service factory returned {type(created).__name__}, expected RuntimeNode")
            return created

        return _build
    return factory


def _coerce_operator_factory(factory: OperatorFactoryLike) -> OperatorFactory:
    if isinstance(factory, type):
        if not issubclass(factory, OperatorNode):
            raise TypeError(f"operator factory type must inherit OperatorNode, got {factory.__name__}")

        def _build(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> OperatorNode:
            created = cast(_OperatorConstructor, factory)(
                node_id=str(node_id),
                node=node,
                initial_state=dict(initial_state or {}),
            )
            if not isinstance(created, OperatorNode):
                raise TypeError(f"operator factory returned {type(created).__name__}, expected OperatorNode")
            return created

        return _build
    return factory


class Registry:
    """
    User-facing registry owner that combines spec registration and runtime factories.

    `RuntimeNodeRegistry` remains available as the low-level owner. `Registry`
    provides a smaller ergonomic surface for service authors while still exposing
    the underlying runtime registry explicitly via `runtime_registry`.
    """

    def __init__(self, runtime_registry: RuntimeNodeRegistry | None = None) -> None:
        self._runtime_registry = runtime_registry if runtime_registry is not None else RuntimeNodeRegistry()

    @classmethod
    def wrap(cls, runtime_registry: RuntimeNodeRegistry) -> "Registry":
        return cls(runtime_registry=runtime_registry)

    @property
    def runtime_registry(self) -> RuntimeNodeRegistry:
        return self._runtime_registry

    def services(self) -> list[str]:
        return self._runtime_registry.services()

    def service_spec(self, service_class: str) -> F8ServiceSpec | None:
        return self._runtime_registry.service_spec(service_class)

    def operator_specs(self, service_class: str) -> list[F8OperatorSpec]:
        return self._runtime_registry.operator_specs(service_class)

    def describe(self, service_class: str) -> F8ServiceDescribe:
        return self._runtime_registry.describe(service_class)

    def register_service_spec(self, spec: F8ServiceSpec, *, overwrite: bool = False) -> "Registry":
        self._runtime_registry.register_service_spec(spec, overwrite=overwrite)
        return self

    def register_operator_spec(self, spec: F8OperatorSpec, *, overwrite: bool = False) -> "Registry":
        self._runtime_registry.register_operator_spec(spec, overwrite=overwrite)
        return self

    def register_service_factory(
        self,
        service_class: str,
        factory: ServiceFactoryLike,
        *,
        overwrite: bool = False,
    ) -> "Registry":
        self._runtime_registry.register_service_factory(
            service_class,
            _coerce_service_factory(factory),
            overwrite=overwrite,
        )
        return self

    def register_operator_factory(
        self,
        service_class: str,
        operator_class: str,
        factory: OperatorFactoryLike,
        *,
        overwrite: bool = False,
    ) -> "Registry":
        self._runtime_registry.register_operator_factory(
            service_class,
            operator_class,
            _coerce_operator_factory(factory),
            overwrite=overwrite,
        )
        return self

    def register_service(
        self,
        spec: F8ServiceSpec,
        factory: ServiceFactoryLike,
        *,
        overwrite: bool = False,
    ) -> "Registry":
        self._runtime_registry.register_service_spec(spec, overwrite=overwrite)
        self._runtime_registry.register_service_factory(
            str(spec.serviceClass or ""),
            _coerce_service_factory(factory),
            overwrite=overwrite,
        )
        return self

    def register_operator(
        self,
        spec: F8OperatorSpec,
        factory: OperatorFactoryLike,
        *,
        overwrite: bool = False,
    ) -> "Registry":
        self._runtime_registry.register_operator_spec(spec, overwrite=overwrite)
        self._runtime_registry.register_operator_factory(
            str(spec.serviceClass or ""),
            str(spec.operatorClass or ""),
            _coerce_operator_factory(factory),
            overwrite=overwrite,
        )
        return self

    def create_service_node(
        self,
        *,
        service_class: str,
        node_id: str,
        initial_state: dict[str, Any] | None = None,
        node: F8RuntimeNode | None = None,
    ) -> RuntimeNode:
        return self._runtime_registry.create_service_node(
            service_class=service_class,
            node_id=node_id,
            initial_state=initial_state,
            node=node,
        )

    def create_operator_node(
        self,
        *,
        node_id: str,
        node: F8RuntimeNode,
        initial_state: dict[str, Any] | None = None,
    ) -> OperatorNode:
        return self._runtime_registry.create_operator_node(
            node_id=node_id,
            node=node,
            initial_state=initial_state,
        )

    def create_runtime_node(
        self,
        *,
        node_id: str,
        node: F8RuntimeNode,
        initial_state: dict[str, Any] | None = None,
    ) -> RuntimeNode:
        return self._runtime_registry.create_runtime_node(
            node_id=node_id,
            node=node,
            initial_state=initial_state,
        )


def create_runtime_node_registry() -> RuntimeNodeRegistry:
    """
    Create a fresh process-local runtime registry.
    """
    return RuntimeNodeRegistry()


def shared_runtime_node_registry() -> RuntimeNodeRegistry:
    """
    Return the explicit process-global runtime registry singleton.
    """
    return RuntimeNodeRegistry.instance()


def create_registry() -> Registry:
    """
    Create a fresh user-facing registry wrapper over a fresh runtime registry.
    """
    return Registry()


def shared_registry() -> Registry:
    """
    Return a user-facing registry wrapper over the explicit shared runtime registry.
    """
    return Registry.wrap(shared_runtime_node_registry())


__all__ = [
    "create_registry",
    "create_runtime_node_registry",
    "OperatorFactoryNotRegistered",
    "OperatorAlreadyRegistered",
    "OperatorFactory",
    "Registry",
    "RegistryError",
    "OperatorFactoryLike",
    "RuntimeNodeRegistry",
    "ServiceFactory",
    "ServiceFactoryLike",
    "ServiceFactoryNotRegistered",
    "ServiceNotRegistered",
    "shared_registry",
    "shared_runtime_node_registry",
]
