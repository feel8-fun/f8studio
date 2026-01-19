from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from qtpy import QtCore

from f8pysdk import F8OperatorSpec

from .service_host.service_host_registry import SERVICE_CLASS as STUDIO_SERVICE_CLASS
from .studio_runtime import StudioRuntime


def _now_ms() -> int:
    return int(time.time() * 1000)


class _NodeHandler(Protocol):
    """
    Per-operator behavior used by StudioExecutor.
    """

    operator_class: str

    def tick(self, *, node: Any, runtime: StudioRuntime, now_ms: int) -> None: ...

    def on_data(self, *, node: Any, port: str, value: Any, ts_ms: Any) -> None: ...


@dataclass
class _PrintNodeHandler:
    operator_class: str = "f8.print_node_operator"
    _last_pull_ms: dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._last_pull_ms is None:
            self._last_pull_ms = {}

    def tick(self, *, node: Any, runtime: StudioRuntime, now_ms: int) -> None:
        try:
            node_id = str(getattr(node, "id", "") or "")
        except Exception:
            node_id = ""
        if not node_id:
            return

        try:
            throttle = node.get_property("throttleMs")
        except Exception:
            throttle = None
        try:
            throttle_ms = max(0, int(throttle) if throttle is not None else 100)
        except Exception:
            throttle_ms = 100

        last = self._last_pull_ms.get(node_id, 0)
        if throttle_ms > 0 and (now_ms - last) < throttle_ms:
            return
        self._last_pull_ms[node_id] = now_ms
        runtime.request_pull(node_id, "inputData")

    def on_data(self, *, node: Any, port: str, value: Any, ts_ms: Any) -> None:
        if str(port) != "inputData":
            return
        try:
            if hasattr(node, "set_preview"):
                node.set_preview(value)
        except Exception:
            return


class StudioNodeRegistry:
    """
    Registry for per-operator editor runtime behavior.

    This is intentionally separate from pysdk RuntimeNodeRegistry:
    - It drives UI refresh/execution, not service process runtime.
    - It maps operatorClass -> editor-side handler.
    """

    @staticmethod
    def instance() -> "StudioNodeRegistry":
        if not hasattr(StudioNodeRegistry, "_instance"):
            StudioNodeRegistry._instance = StudioNodeRegistry()
        return StudioNodeRegistry._instance

    def __init__(self) -> None:
        self._handlers: dict[str, _NodeHandler] = {}
        self.register(_PrintNodeHandler())

    def register(self, handler: _NodeHandler) -> None:
        key = str(getattr(handler, "operator_class", "") or "").strip()
        if not key:
            raise ValueError("handler.operator_class must be non-empty")
        self._handlers[key] = handler

    def get(self, operator_class: str) -> _NodeHandler | None:
        return self._handlers.get(str(operator_class or "").strip())


class StudioExecutor(QtCore.QObject):
    """
    Editor-side executor: on every tick, runs each studio node once (throttled per node).

    Data arrives via `StudioRuntime.data_updated`, and is dispatched to render nodes
    using handler registry (no hard-coded type branches in MainWindow).
    """

    def __init__(self, *, studio_graph: Any, runtime: StudioRuntime, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._graph = studio_graph
        self._runtime = runtime
        self._registry = StudioNodeRegistry.instance()

    def tick(self) -> None:
        now_ms = _now_ms()
        try:
            nodes = list(self._graph.all_nodes() or [])
        except Exception:
            nodes = []

        for node in nodes:
            try:
                spec = getattr(node, "spec", None)
            except Exception:
                spec = None
            if not isinstance(spec, F8OperatorSpec):
                continue
            if str(getattr(spec, "serviceClass", "")) != STUDIO_SERVICE_CLASS:
                continue
            operator_class = str(getattr(spec, "operatorClass", "") or "")
            handler = self._registry.get(operator_class)
            if handler is None:
                continue
            handler.tick(node=node, runtime=self._runtime, now_ms=now_ms)

    @QtCore.Slot(str, str, object, object)
    def on_data_updated(self, node_id: str, port: str, value: Any, ts_ms: Any) -> None:
        try:
            node = self._graph.get_node_by_id(str(node_id))
        except Exception:
            node = None
        if node is None:
            return
        try:
            spec = getattr(node, "spec", None)
        except Exception:
            spec = None
        if not isinstance(spec, F8OperatorSpec):
            return
        if str(getattr(spec, "serviceClass", "")) != STUDIO_SERVICE_CLASS:
            return
        operator_class = str(getattr(spec, "operatorClass", "") or "")
        handler = self._registry.get(operator_class)
        if handler is None:
            return
        handler.on_data(node=node, port=str(port), value=value, ts_ms=ts_ms)

