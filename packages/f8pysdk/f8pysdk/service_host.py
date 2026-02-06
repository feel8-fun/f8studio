from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .generated import F8EdgeKindEnum, F8JsonValue, F8RuntimeGraph, F8RuntimeNode, F8StateAccess
from .runtime_node_registry import RuntimeNodeRegistry
from .service_bus import ServiceBus
from .service_bus import StateWriteOrigin


def _unwrap_json_value(v: Any) -> Any:
    if v is None:
        return None
    try:
        if isinstance(v, F8JsonValue):
            return v.root
    except Exception:
        pass
    try:
        root = getattr(v, "root", None)
        if root is not None:
            return root
    except Exception:
        pass
    return v


def _rungraph_ts(graph: F8RuntimeGraph) -> int:
    meta = getattr(graph, "meta", None)
    if isinstance(meta, dict):
        try:
            return int(meta.get("ts") or 0)
        except Exception:
            return 0
    try:
        return int(getattr(meta, "ts", 0) or 0)
    except Exception:
        return 0


@dataclass(frozen=True)
class ServiceHostConfig:
    """
    Generic push-based service host.

    - One process hosts exactly one service instance (service_id).
    - `service_class` selects which runtime registry is used to build nodes.
    """

    service_class: str


class ServiceHost:
    """
    Push-based service host that binds a `ServiceBus` to per-node runtime implementations.

    - Rungraph drives creation/removal of local runtime nodes.
    - Runtime buffers data edges; exec/data evaluation is driven by the engine layer.
    """

    def __init__(
        self,
        bus: ServiceBus,
        *,
        config: ServiceHostConfig,
        registry: RuntimeNodeRegistry | None = None,
    ) -> None:
        self._bus = bus
        self._config = config
        self._registry = registry or RuntimeNodeRegistry.instance()

        self._service_node: Any | None = None
        self._operator_nodes: dict[str, Any] = {}
        self._bus.register_rungraph_hook(self)

    async def start(self) -> None:
        """
        Ensure the service node exists before any rungraph arrives.

        Rungraph-provided service nodes are treated as state snapshots only.
        """
        if self._service_node is not None:
            return
        service_class = str(self._config.service_class or "").strip()
        if not service_class:
            raise ValueError("ServiceHostConfig.service_class must be non-empty")
        node_id = str(self._bus.service_id).strip()
        try:
            node = self._registry.create_service_node(service_class=service_class, node_id=node_id, initial_state={})
        except Exception:
            node = None
        if node is None:
            return
        self._service_node = node
        try:
            self._bus.register_node(node)
        except Exception:
            self._service_node = None
            return

    async def apply_rungraph(self, graph: F8RuntimeGraph) -> None:
        """
        Register/unregister local runtime nodes based on the latest rungraph snapshot.
        """
        if self._service_node is None:
            try:
                await self.start()
            except Exception:
                return
        service_class = str(self._config.service_class or "").strip()

        want_operator_nodes: list[F8RuntimeNode] = []
        service_snapshot: F8RuntimeNode | None = None
        for n in graph.nodes:
            try:
                if service_class and str(n.serviceClass) != service_class:
                    continue
                if n.operatorClass is None:
                    # Service/container node snapshot (state only).
                    if n.nodeId == str(self._bus.service_id):
                        service_snapshot = n
                    continue
                want_operator_nodes.append(n)
            except Exception:
                continue

        want_ids = {str(n.nodeId) for n in want_operator_nodes}

        for node_id in list(self._operator_nodes.keys()):
            if node_id in want_ids:
                continue
            try:
                self._bus.unregister_node(node_id)
            except Exception:
                pass
            self._operator_nodes.pop(node_id, None)

        for n in want_operator_nodes:
            node_id = str(n.nodeId)
            if node_id in self._operator_nodes:
                # If the node definition changed (ports/state), re-create the runtime node so
                # the local instance matches the rungraph snapshot. Otherwise, edited ports
                # may show in UI but not work at runtime.
                try:
                    existing = self._operator_nodes.get(node_id)
                    if existing is not None and self._needs_recreate(existing, n):
                        try:
                            self._bus.unregister_node(node_id)
                        except Exception:
                            pass
                        self._operator_nodes.pop(node_id, None)
                    else:
                        continue
                except Exception:
                    continue
            initial_state = self._node_initial_state(n)
            try:
                node = self._registry.create(node_id=node_id, node=n, initial_state=initial_state)
            except Exception:
                node = None
            if node is None:
                continue
            self._operator_nodes[node_id] = node
            try:
                self._bus.register_node(node)
            except Exception:
                self._operator_nodes.pop(node_id, None)
                continue

        rungraph_ts = _rungraph_ts(graph)

        # Cross-service state edges: downstream fields are driven by upstream KV,
        # so they must not be overwritten by rungraph reconciliation defaults.
        cross_state_targets: set[tuple[str, str]] = set()
        for e in list(getattr(graph, "edges", None) or []):
            try:
                if getattr(e, "kind", None) != F8EdgeKindEnum.state:
                    continue
                # Only cross-service edges targeting this service.
                if str(getattr(e, "fromServiceId", "")) == str(getattr(e, "toServiceId", "")):
                    continue
                if str(getattr(e, "toServiceId", "")) != str(self._bus.service_id):
                    continue
                to_node = str(getattr(e, "toOperatorId", "") or "").strip()
                to_field = str(getattr(e, "toPort", "") or "").strip()
                if not to_node or not to_field:
                    continue
                cross_state_targets.add((to_node, to_field))
            except Exception:
                continue

        if service_snapshot is not None:
            await self._seed_service_state_defaults(
                service_snapshot, rungraph_ts=rungraph_ts, skip_fields=cross_state_targets
            )
        await self._seed_state_defaults(want_operator_nodes, rungraph_ts=rungraph_ts, skip_fields=cross_state_targets)

    async def on_rungraph(self, graph: F8RuntimeGraph) -> None:
        await self.apply_rungraph(graph)

    async def validate_rungraph(self, graph: F8RuntimeGraph) -> None:
        _ = graph

    @staticmethod
    def _needs_recreate(node: Any, snapshot: F8RuntimeNode) -> bool:
        """
        Return True if the local runtime node should be re-created for the given snapshot.

        Nodes are cached by nodeId; without this check, editing ports in Studio can
        yield a rungraph that routes to ports the old instance doesn't expose.
        """
        try:
            desired_in = [str(p.name) for p in list(getattr(snapshot, "dataInPorts", None) or [])]
        except Exception:
            desired_in = []
        try:
            desired_out = [str(p.name) for p in list(getattr(snapshot, "dataOutPorts", None) or [])]
        except Exception:
            desired_out = []
        try:
            desired_state = [str(sf.name) for sf in list(getattr(snapshot, "stateFields", None) or [])]
        except Exception:
            desired_state = []
        try:
            current_in = [str(x) for x in list(getattr(node, "data_in_ports", None) or [])]
        except Exception:
            current_in = []
        try:
            current_out = [str(x) for x in list(getattr(node, "data_out_ports", None) or [])]
        except Exception:
            current_out = []
        try:
            current_state = [str(x) for x in list(getattr(node, "state_fields", None) or [])]
        except Exception:
            current_state = []

        if current_in != desired_in or current_out != desired_out or current_state != desired_state:
            return True

        # Best-effort exec port change detection (common pattern in runtime nodes).
        desired_exec_out = list(getattr(snapshot, "execOutPorts", None) or [])
        try:
            cur_exec_out = list(getattr(node, "_exec_out_ports", None) or [])
        except Exception:
            cur_exec_out = []
        if cur_exec_out and [str(x) for x in cur_exec_out] != [str(x) for x in desired_exec_out]:
            return True

        return False

    @staticmethod
    def _node_initial_state(n: F8RuntimeNode) -> dict[str, Any]:
        values = getattr(n, "stateValues", None) or {}
        out: dict[str, Any] = {}
        if not isinstance(values, dict):
            return out
        for k, v in values.items():
            out[str(k)] = _unwrap_json_value(v)
        return out

    async def _seed_state_defaults(
        self, nodes: list[F8RuntimeNode], *, rungraph_ts: int = 0, skip_fields: set[tuple[str, str]] | None = None
    ) -> None:
        """
        Seed rungraph-provided initial state values into KV.

        If KV already has a value with a newer/equal timestamp than the rungraph
        snapshot, prefer the KV value. Otherwise, allow the rungraph snapshot to
        overwrite older KV state.
        """
        for n in nodes:
            node_id = str(n.nodeId)
            try:
                access_by_name = {str(sf.name): sf.access for sf in list(getattr(n, "stateFields", None) or [])}
            except Exception:
                access_by_name = {}
            for k, v in self._node_initial_state(n).items():
                if skip_fields and (node_id, str(k)) in skip_fields:
                    continue
                # Never seed/overwrite read-only state from rungraph; it is runtime-owned.
                try:
                    if access_by_name.get(str(k)) == F8StateAccess.ro:
                        continue
                except Exception:
                    pass
                existing_ts = None
                existing_value = None
                try:
                    found, existing_value, existing_ts = await self._bus.get_state_with_ts(node_id, str(k))
                except Exception:
                    found = False
                if found:
                    try:
                        if existing_value == v:
                            continue
                    except Exception:
                        pass
                    try:
                        if int(existing_ts or 0) >= int(rungraph_ts or 0):
                            continue
                    except Exception:
                        continue
                try:
                    await self._bus._publish_state(
                        node_id,
                        str(k),
                        v,
                        ts_ms=(int(rungraph_ts) if rungraph_ts > 0 else None),
                        origin=StateWriteOrigin.rungraph,
                        source="rungraph",
                        meta={"rungraphReconcile": True},
                    )
                except Exception:
                    continue

    async def _seed_service_state_defaults(
        self, n: F8RuntimeNode, *, rungraph_ts: int = 0, skip_fields: set[tuple[str, str]] | None = None
    ) -> None:
        """
        Apply rungraph-provided `stateValues` for the service node into KV.
        """
        node_id = str(self._bus.service_id)
        try:
            access_by_name = {str(sf.name): sf.access for sf in list(n.stateFields or [])}
        except Exception:
            access_by_name = {}
        for k, v in self._node_initial_state(n).items():
            if skip_fields and (node_id, str(k)) in skip_fields:
                continue
            # Never seed/overwrite read-only state from rungraph; it is runtime-owned.
            if k not in access_by_name:
                continue
            
            if access_by_name[k] == F8StateAccess.ro:
                continue
            
            existing_ts = None
            existing_value = None
            try:
                found, existing_value, existing_ts = await self._bus.get_state_with_ts(node_id, str(k))
            except Exception:
                found = False
            if found:
                try:
                    if existing_value == v:
                        continue
                except Exception:
                    pass
                try:
                    if int(existing_ts or 0) >= int(rungraph_ts or 0):
                        continue
                except Exception:
                    continue
            try:
                await self._bus._publish_state(
                    node_id,
                    str(k),
                    v,
                    ts_ms=(int(rungraph_ts) if rungraph_ts > 0 else None),
                    origin=StateWriteOrigin.rungraph,
                    source="rungraph",
                    meta={"rungraphReconcile": True, "serviceNode": True},
                )
            except Exception:
                continue
