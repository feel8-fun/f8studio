from __future__ import annotations

from collections.abc import Iterable

import msgspec

from f8pysdk.command import command_input_state_field, command_output_state_field, hidden_command_state_specs
from f8pysdk.specs import (
    F8DataPortSpec,
    F8ExecPortSpec,
    F8JsonValue,
    F8OperatorSpec,
    F8ServiceSpec,
    F8StateAccess,
    F8StateSpec,
)

from .models import GraphNode, GraphPort, OperatorNode, PortDirection, PortKind, ServiceNode


def _clone_service_spec(spec: F8ServiceSpec) -> F8ServiceSpec:
    return msgspec.json.decode(msgspec.json.encode(spec), type=F8ServiceSpec)


def _clone_operator_spec(spec: F8OperatorSpec) -> F8OperatorSpec:
    return msgspec.json.decode(msgspec.json.encode(spec), type=F8OperatorSpec)


def _data_ports(ports: list[F8DataPortSpec] | msgspec.UnsetType) -> list[F8DataPortSpec]:
    return [] if isinstance(ports, msgspec.UnsetType) else list(ports)


def _state_fields(fields: list[F8StateSpec] | msgspec.UnsetType) -> list[F8StateSpec]:
    return [] if isinstance(fields, msgspec.UnsetType) else list(fields)


def _exec_ports(ports: list[F8ExecPortSpec] | msgspec.UnsetType) -> list[F8ExecPortSpec]:
    return [] if isinstance(ports, msgspec.UnsetType) else list(ports)


def _port_id(kind: PortKind, direction: PortDirection, name: str) -> str:
    return f"{kind.value}:{direction.value}:{name}"


def _data_port(spec: F8DataPortSpec, direction: PortDirection) -> GraphPort:
    name = str(spec.name).strip()
    return GraphPort(
        port_id=_port_id(PortKind.data, direction, name),
        name=name,
        runtime_name=name,
        kind=PortKind.data,
        direction=direction,
        data_spec=spec,
    )


def _state_ports(spec: F8StateSpec) -> list[GraphPort]:
    name = str(spec.name).strip()
    ports: list[GraphPort] = []
    if spec.access != F8StateAccess.ro:
        ports.append(
            GraphPort(
                port_id=_port_id(PortKind.state, PortDirection.input, name),
                name=name,
                runtime_name=name,
                kind=PortKind.state,
                direction=PortDirection.input,
                state_spec=spec,
            )
        )
    if spec.access != F8StateAccess.wo:
        ports.append(
            GraphPort(
                port_id=_port_id(PortKind.state, PortDirection.output, name),
                name=name,
                runtime_name=name,
                kind=PortKind.state,
                direction=PortDirection.output,
                state_spec=spec,
            )
        )
    return ports


def ports_for_spec(spec: F8ServiceSpec | F8OperatorSpec, port_ids: dict[str, str] | None = None) -> tuple[GraphPort, ...]:
    ports: list[GraphPort] = []
    if isinstance(spec, F8OperatorSpec):
        for exec_spec in _exec_ports(spec.execInPorts):
            name = exec_spec.name
            ports.append(
                GraphPort(
                    port_id=_port_id(PortKind.exec, PortDirection.input, name),
                    name=name,
                    runtime_name=name,
                    kind=PortKind.exec,
                    direction=PortDirection.input,
                )
            )
        for exec_spec in _exec_ports(spec.execOutPorts):
            name = exec_spec.name
            ports.append(
                GraphPort(
                    port_id=_port_id(PortKind.exec, PortDirection.output, name),
                    name=name,
                    runtime_name=name,
                    kind=PortKind.exec,
                    direction=PortDirection.output,
                )
            )
    for data_spec in _data_ports(spec.dataInPorts):
        ports.append(_data_port(data_spec, PortDirection.input))
    for data_spec in _data_ports(spec.dataOutPorts):
        ports.append(_data_port(data_spec, PortDirection.output))
    for state_spec in _state_fields(spec.stateFields):
        ports.extend(_state_ports(state_spec))

    commands = [] if isinstance(spec.commands, msgspec.UnsetType) else list(spec.commands)
    for command in commands:
        command_states = hidden_command_state_specs([command])
        if len(command_states) != 2:
            raise ValueError(f"command must produce exactly two hidden state fields: {command.name}")
        input_state, output_state = command_states
        name = str(command.name).strip()
        ports.append(
            GraphPort(
                port_id=_port_id(PortKind.command, PortDirection.input, name),
                name=name,
                runtime_name=command_input_state_field(name),
                kind=PortKind.command,
                direction=PortDirection.input,
                state_spec=input_state,
            )
        )
        ports.append(
            GraphPort(
                port_id=_port_id(PortKind.command, PortDirection.output, name),
                name=name,
                runtime_name=command_output_state_field(name),
                kind=PortKind.command,
                direction=PortDirection.output,
                state_spec=output_state,
            )
        )
    if port_ids:
        return tuple(msgspec.structs.replace(port, port_id=port_ids.get(port.port_id, port.port_id)) for port in ports)
    return tuple(ports)


def _preserved_port_ids(
    node: GraphNode,
    spec: F8ServiceSpec | F8OperatorSpec,
    renames: dict[str, str],
) -> dict[str, str]:
    old_ids = {port.port_id for port in node.ports}
    unknown = set(renames) - old_ids
    if unknown:
        raise ValueError(f"renamed port not found: {', '.join(sorted(unknown))}")
    available = {port.port_id for port in ports_for_spec(spec)}
    preserved: dict[str, str] = {}
    for port in node.ports:
        name = renames.get(port.port_id, port.name)
        key = _port_id(port.kind, port.direction, name)
        if port.port_id in renames and key not in available:
            raise ValueError(f"renamed port target not found: {key}")
        if key not in available or key == port.port_id:
            continue
        if key in preserved:
            raise ValueError(f"multiple ports renamed to {key}")
        preserved[key] = port.port_id
    return preserved


def _state_values_after_rename(
    node: GraphNode,
    spec: F8ServiceSpec | F8OperatorSpec,
    renames: dict[str, str],
) -> dict[str, F8JsonValue]:
    writable = {
        str(field.name)
        for field in _state_fields(spec.stateFields)
        if field.access != F8StateAccess.ro
    }
    targets: dict[str, set[str]] = {}
    for port in node.ports:
        if port.kind == PortKind.state and port.port_id in renames:
            targets.setdefault(port.name, set()).add(renames[port.port_id])
    for old_name, names in targets.items():
        if len(names) != 1 or any(
            port.kind == PortKind.state and port.name == old_name and port.port_id not in renames
            for port in node.ports
        ):
            raise ValueError(f"state rename must include every endpoint: {old_name}")
    renamed = {old_name: next(iter(names)) for old_name, names in targets.items()}
    values: dict[str, F8JsonValue] = {}
    for old_name, value in node.state_values.items():
        name = renamed.get(old_name, old_name)
        if name in writable:
            if name in values:
                raise ValueError(f"state rename produces duplicate value: {name}")
            values[name] = value
    return values


def replace_node_spec(
    node: GraphNode,
    spec: F8ServiceSpec | F8OperatorSpec,
    *,
    port_renames: dict[str, str] | None = None,
) -> GraphNode:
    if isinstance(node, ServiceNode):
        if not isinstance(spec, F8ServiceSpec):
            raise TypeError("service node requires an F8ServiceSpec")
        if str(spec.serviceClass) != node.service_class:
            raise ValueError("serviceClass cannot change during spec replacement")
        copied_spec = _clone_service_spec(spec)
        port_ids = _preserved_port_ids(node, copied_spec, port_renames or {})
        return ServiceNode(
            node_id=node.node_id,
            name=node.name,
            service_id=node.service_id,
            service_class=node.service_class,
            spec=copied_spec,
            ports=ports_for_spec(copied_spec, port_ids),
            port_ids=port_ids,
            state_values=_state_values_after_rename(node, copied_spec, port_renames or {}),
            enabled=node.enabled,
        )
    if not isinstance(spec, F8OperatorSpec):
        raise TypeError("operator node requires an F8OperatorSpec")
    if str(spec.serviceClass) != node.service_class or str(spec.operatorClass) != node.operator_class:
        raise ValueError("operator identity cannot change during spec replacement")
    copied_spec = _clone_operator_spec(spec)
    port_ids = _preserved_port_ids(node, copied_spec, port_renames or {})
    return OperatorNode(
        node_id=node.node_id,
        name=node.name,
        service_id=node.service_id,
        service_class=node.service_class,
        operator_class=node.operator_class,
        spec=copied_spec,
        ports=ports_for_spec(copied_spec, port_ids),
        port_ids=port_ids,
        state_values=_state_values_after_rename(node, copied_spec, port_renames or {}),
        enabled=node.enabled,
    )


class NodeCatalog:
    def __init__(
        self,
        *,
        services: Iterable[F8ServiceSpec] = (),
        operators: Iterable[F8OperatorSpec] = (),
    ) -> None:
        self._services: dict[str, F8ServiceSpec] = {}
        self._operators: dict[tuple[str, str], F8OperatorSpec] = {}
        for spec in services:
            self.register_service(spec)
        for spec in operators:
            self.register_operator(spec)

    def register_service(self, spec: F8ServiceSpec) -> None:
        key = str(spec.serviceClass).strip()
        if not key:
            raise ValueError("serviceClass must be non-empty")
        if key in self._services:
            raise ValueError(f"duplicate service spec: {key}")
        self._services[key] = _clone_service_spec(spec)

    def register_operator(self, spec: F8OperatorSpec) -> None:
        key = (str(spec.serviceClass).strip(), str(spec.operatorClass).strip())
        if not key[0] or not key[1]:
            raise ValueError("operator serviceClass and operatorClass must be non-empty")
        if key in self._operators:
            raise ValueError(f"duplicate operator spec: {key[0]}/{key[1]}")
        self._operators[key] = _clone_operator_spec(spec)

    def create_service_node(
        self,
        *,
        node_id: str,
        service_class: str,
        name: str | None = None,
        state_values: dict[str, F8JsonValue] | None = None,
    ) -> ServiceNode:
        spec = self._services.get(service_class)
        if spec is None:
            raise KeyError(f"unknown service spec: {service_class}")
        copied_spec = _clone_service_spec(spec)
        return ServiceNode(
            node_id=node_id,
            name=name or str(copied_spec.label),
            service_id=node_id,
            service_class=str(copied_spec.serviceClass),
            spec=copied_spec,
            ports=ports_for_spec(copied_spec),
            state_values=dict(state_values or {}),
        )

    def create_operator_node(
        self,
        *,
        node_id: str,
        service_id: str,
        service_class: str,
        operator_class: str,
        name: str | None = None,
        state_values: dict[str, F8JsonValue] | None = None,
    ) -> OperatorNode:
        spec = self._operators.get((service_class, operator_class))
        if spec is None:
            raise KeyError(f"unknown operator spec: {service_class}/{operator_class}")
        copied_spec = _clone_operator_spec(spec)
        return OperatorNode(
            node_id=node_id,
            name=name or str(copied_spec.label),
            service_id=service_id,
            service_class=str(copied_spec.serviceClass),
            operator_class=str(copied_spec.operatorClass),
            spec=copied_spec,
            ports=ports_for_spec(copied_spec),
            state_values=dict(state_values or {}),
        )

    def create_node_from_spec(
        self,
        *,
        node_id: str,
        service_id: str,
        spec: F8ServiceSpec | F8OperatorSpec,
        name: str | None = None,
    ) -> GraphNode:
        if isinstance(spec, F8ServiceSpec):
            self.register_service(spec)
            return self.create_service_node(node_id=node_id, service_class=str(spec.serviceClass), name=name)
        self.register_operator(spec)
        return self.create_operator_node(
            node_id=node_id,
            service_id=service_id,
            service_class=str(spec.serviceClass),
            operator_class=str(spec.operatorClass),
            name=name,
        )
