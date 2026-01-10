from __future__ import annotations

import importlib
from typing import Callable

from f8pysdk import F8OperatorSpec

from .operator_runtime import (
    AddNode,
    ConstantNode,
    LogNode,
    OperatorContext,
    OperatorRuntimeNode,
    StartNode,
)


Factory = Callable[[str, OperatorContext], OperatorRuntimeNode]


class OperatorRuntimeRegistry:
    @staticmethod
    def instance() -> "OperatorRuntimeRegistry":
        global _GLOBAL_OPERATOR_RUNTIME_REGISTRY
        try:
            return _GLOBAL_OPERATOR_RUNTIME_REGISTRY
        except NameError:
            _GLOBAL_OPERATOR_RUNTIME_REGISTRY = OperatorRuntimeRegistry()
            return _GLOBAL_OPERATOR_RUNTIME_REGISTRY

    def __init__(self) -> None:
        self._factories: dict[str, Factory] = {}
        self._seed_defaults()

    def load_modules(self, modules: list[str]) -> None:
        """
        Import modules that register operator runtimes.

        A module can register by calling:
        - `OperatorRuntimeRegistry.instance().register(...)`
        """
        for m in modules:
            name = str(m or "").strip()
            if not name:
                continue
            importlib.import_module(name)

    def _seed_defaults(self) -> None:
        self.register("f8/start", lambda node_id, ctx: StartNode(node_id=node_id, ctx=ctx), overwrite=True)
        self.register("f8/constant", lambda node_id, ctx: ConstantNode(node_id=node_id, ctx=ctx), overwrite=True)
        self.register("f8/add", lambda node_id, ctx: AddNode(node_id=node_id, ctx=ctx), overwrite=True)
        self.register("f8/log", lambda node_id, ctx: LogNode(node_id=node_id, ctx=ctx), overwrite=True)

    def register(self, operator_class: str, factory: Factory, *, overwrite: bool = False) -> None:
        if operator_class in self._factories and not overwrite:
            raise ValueError(f"runtime already registered for {operator_class}")
        self._factories[str(operator_class)] = factory

    def create(self, *, node_id: str, spec: F8OperatorSpec, initial_state: dict) -> OperatorRuntimeNode:
        ctx = OperatorContext(spec=spec, initial_state=dict(initial_state or {}))
        factory = self._factories.get(str(spec.operatorClass))
        if factory is None:
            return OperatorRuntimeNode(node_id=node_id, ctx=ctx)
        return factory(str(node_id), ctx)
