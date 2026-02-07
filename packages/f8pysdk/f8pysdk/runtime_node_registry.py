from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, ClassVar

from .generated import F8OperatorSpec, F8RuntimeNode, F8ServiceDescribe, F8ServiceSpec, F8StateAccess, F8StateSpec
from .runtime_node import RuntimeNode, ServiceNode
from .schema_helpers import string_schema


OperatorFactory = Callable[[str, F8RuntimeNode, dict[str, Any]], RuntimeNode]
ServiceFactory = Callable[[str, F8RuntimeNode, dict[str, Any]], RuntimeNode]


class RegistryError(Exception):
    """Base class for registry failures."""


class ServiceNotRegistered(RegistryError):
    """Raised when a serviceClass has no runtime registry."""


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

    _instance: ClassVar["RuntimeNodeRegistry | None"] = None

    @staticmethod
    def instance() -> "RuntimeNodeRegistry":
        # Singleton instance accessor.
        if RuntimeNodeRegistry._instance is None:
            RuntimeNodeRegistry._instance = RuntimeNodeRegistry()
        return RuntimeNodeRegistry._instance
    

    def __init__(self) -> None:
        self._by_service_operator: dict[str, dict[str, OperatorFactory]] = {}
        self._by_service_service: dict[str, ServiceFactory] = {}
        # Optional: service/operator specs for `--describe` style discovery.
        self._service_specs: dict[str, F8ServiceSpec] = {}
        self._operator_specs: dict[str, dict[str, F8OperatorSpec]] = {}

    def services(self) -> list[str]:
        keys = set(self._by_service_operator.keys())
        keys.update(self._by_service_service.keys())
        keys.update(self._service_specs.keys())
        keys.update(self._operator_specs.keys())
        return sorted(keys)

    # ---- specs (optional) ----------------------------------------------
    def register_service_spec(self, spec: F8ServiceSpec, *, overwrite: bool = False) -> None:
        """
        Register a `F8ServiceSpec` for discovery / `--describe`.
        """
        service_class = str(spec.serviceClass or "").strip()
        if not service_class:
            raise ValueError("spec.serviceClass must be non-empty")
        if service_class in self._service_specs and not overwrite:
            raise OperatorAlreadyRegistered(f"service spec already registered for {service_class}")
        self._service_specs[service_class] = spec

    def register_operator_spec(self, spec: F8OperatorSpec, *, overwrite: bool = False) -> None:
        """
        Register a `F8OperatorSpec` for discovery / `--describe`.
        """
        service_class = str(spec.serviceClass or "").strip()
        operator_class = str(spec.operatorClass or "").strip()
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

    def service_spec(self, service_class: str) -> F8ServiceSpec | None:
        return self._service_specs.get(str(service_class or "").strip())

    def operator_specs(self, service_class: str) -> list[F8OperatorSpec]:
        reg = self._operator_specs.get(str(service_class or "").strip()) or {}
        return list(reg.values())

    def describe(self, service_class: str) -> F8ServiceDescribe:
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
        self._inject_builtin_identity_state_fields(service, operators)
        return F8ServiceDescribe(service=service, operators=operators)

    @staticmethod
    def _inject_builtin_identity_state_fields(service_spec: F8ServiceSpec, operator_specs: list[F8OperatorSpec]) -> None:
        """
        Inject readonly identity fields into specs so graphs can route them like normal state.

        - `svcId`: current service instance id
        - `operatorId`: runtime node id (operator id)
        """
        builtin_all = [
            F8StateSpec(
                name="svcId",
                label="Service Id",
                description="Readonly: current service instance id (svcId).",
                valueSchema=string_schema(),
                access=F8StateAccess.ro,
                showOnNode=False,
            ),
        ]
        builtin_operator_only = [
            F8StateSpec(
                name="operatorId",
                label="Operator Id",
                description="Readonly: current operator/node id (operatorId).",
                valueSchema=string_schema(),
                access=F8StateAccess.ro,
                showOnNode=False,
            ),
        ]

        def _apply(spec: F8ServiceSpec | F8OperatorSpec, extra: list[F8StateSpec]) -> None:
            fields = list(spec.stateFields or [])
            existing = {str(sf.name) for sf in fields}
            added = False
            for sf in [*builtin_all, *extra]:
                if sf.name in existing:
                    continue
                fields.append(sf)
                added = True
            if added:
                spec.stateFields = fields

        _apply(service_spec, [])
        for op in list(operator_specs or []):
            _apply(op, builtin_operator_only)

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

    def create_service_node(
        self,
        *,
        service_class: str,
        node_id: str,
        initial_state: dict[str, Any] | None = None,
        node: F8RuntimeNode | None = None,
    ) -> RuntimeNode:
        """
        Create the service/container node for a service process.

        If `node` is omitted, a minimal placeholder `F8RuntimeNode` is constructed.
        """
        service_class = str(service_class or "").strip()
        node_id = str(node_id or "").strip()
        if not service_class:
            raise ValueError("service_class must be non-empty")
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if node is None:
            # Service/container nodes use `nodeId == serviceId`.
            node = F8RuntimeNode(nodeId=str(node_id), serviceId=str(node_id), serviceClass=str(service_class), operatorClass=None)
        factory = self._by_service_service.get(service_class)
        if factory is None:
            if service_class not in self._by_service_operator and service_class not in self._by_service_service:
                raise ServiceNotRegistered(service_class)
            return ServiceNode(node_id=str(node_id))
        return factory(str(node_id), node, dict(initial_state or {}))

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
            return self.create_service_node(
                service_class=str(service_class),
                node_id=str(node_id),
                initial_state=dict(initial_state or {}),
                node=node,
            )

        reg = self._by_service_operator.get(service_class)
        if reg is None:
            if service_class not in self._by_service_service:
                raise ServiceNotRegistered(service_class)
            return RuntimeNode(node_id=str(node_id))

        factory = reg.get(str(operator_class))
        if factory is None:
            return RuntimeNode(node_id=str(node_id))
        return factory(str(node_id), node, dict(initial_state or {}))

    def load_modules(self, modules: list[str]) -> None:
        """
        Import modules that register runtime implementations.

        A module can register by calling:
        - `RuntimeNodeRegistry.instance().register(...)`
        """
        for m in modules:
            name = str(m or "").strip()
            if not name:
                continue
            importlib.import_module(name)
