from __future__ import annotations

from typing import Any


def exec_out_ports(node: Any, *, default: list[str] | None = None) -> list[str]:
    """
    Extract exec out port names from a runtime node.

    `F8RuntimeNode.execOutPorts` should be present for operator nodes, but we keep this
    best-effort because some tooling/tests may pass partial objects.
    """
    try:
        ports = list(node.execOutPorts or [])
    except Exception:
        ports = []
    if ports:
        return [str(p) for p in ports if str(p)]
    if default is None:
        return []
    return [str(p) for p in list(default) if str(p)]

