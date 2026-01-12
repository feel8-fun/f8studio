from __future__ import annotations

from typing import Any

from f8pysdk import F8RuntimeNode
from f8pysdk.runtime import ServiceOperatorRuntimeRegistry, ServiceRuntimeNode

from .signal_runtime import PrintRuntimeNode, SineRuntimeNode
from .tick_runtime import TickRuntimeNode


def register_pyengine_runtimes(registry: ServiceOperatorRuntimeRegistry | None = None) -> ServiceOperatorRuntimeRegistry:
    """
    Register built-in f8.pyengine runtime implementations into the shared registry.
    """
    reg = registry or ServiceOperatorRuntimeRegistry.instance()

    def _tick_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> ServiceRuntimeNode:
        return TickRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    def _sine_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> ServiceRuntimeNode:
        return SineRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    def _print_factory(node_id: str, node: F8RuntimeNode, initial_state: dict[str, Any]) -> ServiceRuntimeNode:
        return PrintRuntimeNode(node_id=node_id, node=node, initial_state=initial_state)

    reg.register("f8.pyengine", "f8.tick", _tick_factory, overwrite=True)
    reg.register("f8.pyengine", "f8.sine", _sine_factory, overwrite=True)
    reg.register("f8.pyengine", "f8.print", _print_factory, overwrite=True)
    return reg
