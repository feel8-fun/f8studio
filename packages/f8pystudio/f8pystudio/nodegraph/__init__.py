from __future__ import annotations

from typing import TYPE_CHECKING

from .node_base import F8StudioBaseNode

if TYPE_CHECKING:
    from .node_graph import F8StudioGraph


__all__ = ["F8StudioGraph", "F8StudioBaseNode"]


def __getattr__(name: str):
    if name == "F8StudioGraph":
        from .node_graph import F8StudioGraph as _F8StudioGraph

        return _F8StudioGraph
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(__all__)
