from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from f8pysdk import F8RuntimeGraph
from f8pysdk.runtime import ServiceBus, ServiceOperatorRuntimeRegistry, ensure_token

from .engine_executor import EngineExecutor


@dataclass(frozen=True)
class EngineHostConfig:
    service_class: str = "f8.pyengine"


class EngineHost:
    """
    Engine-side host that materializes runtime nodes from `F8RuntimeGraph` and wires them to:
    - `ServiceBus` for data/state routing
    - `EngineExecutor` for exec routing / source lifecycle
    """

    def __init__(
        self,
        runtime: ServiceBus,
        executor: EngineExecutor,
        *,
        config: EngineHostConfig,
        registry: ServiceOperatorRuntimeRegistry | None = None,
    ) -> None:
        self._runtime = runtime
        self._executor = executor
        self._config = config
        self._registry = registry or ServiceOperatorRuntimeRegistry.instance()
        self._nodes: dict[str, Any] = {}
        self._debug_state = str(os.getenv("F8_STATE_DEBUG", "")).lower() in ("1", "true", "yes", "on")

    async def apply_rungraph(self, graph: F8RuntimeGraph) -> None:
        # Only materialize executable operator nodes. Container/service nodes may exist in the
        # runtime graph for metadata/telemetry/state but are not engine-executed.
        want_nodes = [
            n
            for n in (graph.nodes or [])
            if str(getattr(n, "serviceClass", "")) == self._config.service_class
            and getattr(n, "operatorClass", None)
        ]
        if self._debug_state:
            try:
                node_ids = [str(n.nodeId) for n in want_nodes]
            except Exception:
                node_ids = []
            print(
                "state_debug[%s] apply_rungraph nodes=%s"
                % (self._runtime.service_id, ",".join(node_ids))
            )
        want_ids = {str(n.nodeId) for n in want_nodes}

        for node_id in list(self._nodes.keys()):
            if node_id in want_ids:
                continue
            try:
                self._executor.unregister_node(node_id)
            except Exception:
                pass
            try:
                self._runtime.unregister_node(node_id)
            except Exception:
                pass
            self._nodes.pop(node_id, None)

        for n in want_nodes:
            node_id = ensure_token(str(n.nodeId), label="nodeId")
            initial_state = {}
            try:
                state_values = getattr(n, "stateValues", None) or {}
                if isinstance(state_values, dict):
                    for k, v in state_values.items():
                        try:
                            initial_state[str(k)] = getattr(v, "root", v)
                        except Exception:
                            initial_state[str(k)] = v
            except Exception:
                initial_state = {}
            if self._debug_state:
                print(
                    "state_debug[%s] init_state node=%s state=%s"
                    % (self._runtime.service_id, node_id, repr(initial_state))
                )
            if node_id not in self._nodes:
                try:
                    node = self._registry.create(node_id=node_id, node=n, initial_state=initial_state)
                except Exception:
                    node = None
                if node is None:
                    continue
                self._nodes[node_id] = node
                try:
                    self._runtime.register_node(node)
                    self._executor.register_node(node)
                except Exception:
                    self._nodes.pop(node_id, None)
                    try:
                        self._runtime.unregister_node(node_id)
                    except Exception:
                        pass
                    try:
                        self._executor.unregister_node(node_id)
                    except Exception:
                        pass
                    continue

            # Reconcile rungraph-provided stateValues into KV for both new and existing nodes.
            # If KV already has a value and differs, prefer the rungraph value and write
            # it back with a fresh timestamp (current time).
            for k, v in (initial_state or {}).items():
                try:
                    existing = await self._runtime.get_state(node_id, str(k))
                except Exception:
                    existing = None
                if existing is not None and existing == v:
                    continue
                if self._debug_state:
                    print(
                        "state_debug[%s] reconcile node=%s field=%s old=%s new=%s"
                        % (self._runtime.service_id, node_id, str(k), repr(existing), repr(v))
                    )
                try:
                    await self._runtime.set_state_with_meta(
                        node_id,
                        str(k),
                        v,
                        source="rungraph",
                        meta={"rungraphReconcile": True},
                    )
                except Exception:
                    continue
