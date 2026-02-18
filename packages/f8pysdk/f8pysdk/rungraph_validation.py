from __future__ import annotations

from typing import Iterable

from .generated import F8Edge, F8EdgeKindEnum, F8RuntimeGraph, F8StateAccess


def _state_key(*, service_id: str, node_id: str, field: str) -> tuple[str, str, str]:
    return (str(service_id), str(node_id), str(field))


def _fmt_key(k: tuple[str, str, str]) -> str:
    sid, nid, fld = k
    return f"{sid}.{nid}.{fld}"


def validate_state_edges_or_raise(
    graph: F8RuntimeGraph,
    *,
    forbid_cycles: bool = True,
    forbid_multi_upstream: bool = True,
) -> None:
    """
    Validate state-edge wiring constraints on a (possibly global) rungraph.

    This does not depend on ServiceBus and is safe to call from Studio when
    compiling a global graph.

    Enforced (when enabled):
    - at most one upstream per downstream state field
    - no cycles in the state-edge graph
    """
    edges: Iterable[F8Edge] = list(graph.edges or [])

    # Build graph keyed by (serviceId, operatorId, port/field).
    out: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    inbound_count: dict[tuple[str, str, str], int] = {}
    upstream_by_target: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    nodes: set[tuple[str, str, str]] = set()

    for e in edges:
        if e.kind != F8EdgeKindEnum.state:
            continue
        from_sid = str(e.fromServiceId or "").strip()
        to_sid = str(e.toServiceId or "").strip()
        from_op = str(e.fromOperatorId or "").strip()
        to_op = str(e.toOperatorId or "").strip()
        from_field = str(e.fromPort or "").strip()
        to_field = str(e.toPort or "").strip()
        if not (from_sid and to_sid and from_op and to_op and from_field and to_field):
            # Incomplete edges are ignored here; other validation layers may reject.
            continue

        from_key = _state_key(service_id=from_sid, node_id=from_op, field=from_field)
        to_key = _state_key(service_id=to_sid, node_id=to_op, field=to_field)

        if forbid_multi_upstream:
            prev = upstream_by_target.get(to_key)
            if prev is not None and prev != from_key:
                raise ValueError(
                    "multiple upstreams for state field: "
                    f"{_fmt_key(to_key)} <- {_fmt_key(prev)} and {_fmt_key(from_key)}"
                )
            upstream_by_target[to_key] = from_key

        out.setdefault(from_key, []).append(to_key)
        inbound_count[to_key] = int(inbound_count.get(to_key, 0)) + 1
        nodes.add(from_key)
        nodes.add(to_key)

    if not forbid_cycles or not out:
        return

    # Cycle detection (DFS with recursion stack) so we can report one concrete cycle.
    visiting: set[tuple[str, str, str]] = set()
    visited: set[tuple[str, str, str]] = set()
    parent: dict[tuple[str, str, str], tuple[str, str, str] | None] = {}

    def _reconstruct_cycle(start: tuple[str, str, str], end: tuple[str, str, str]) -> list[tuple[str, str, str]]:
        # We found an edge start -> end where end is on the stack.
        # Walk back from start to end using parent pointers.
        cycle: list[tuple[str, str, str]] = [end, start]
        cur = parent.get(start)
        while cur is not None and cur != end and cur not in cycle:
            cycle.append(cur)
            cur = parent.get(cur)
        cycle.append(end)
        cycle.reverse()
        return cycle

    def _dfs(n: tuple[str, str, str]) -> list[tuple[str, str, str]] | None:
        visiting.add(n)
        for m in out.get(n, []):
            if m in visited:
                continue
            if m in visiting:
                return _reconstruct_cycle(n, m)
            parent[m] = n
            cyc = _dfs(m)
            if cyc is not None:
                return cyc
        visiting.remove(n)
        visited.add(n)
        return None

    # Prefer starting from roots when available, but still scan all components.
    roots = [n for n in nodes if inbound_count.get(n, 0) == 0]
    start_nodes = roots + [n for n in nodes if n not in roots]
    for n in start_nodes:
        if n in visited:
            continue
        parent.setdefault(n, None)
        cyc = _dfs(n)
        if cyc is not None:
            msg = "cyclic state-edge loop detected: " + " -> ".join(_fmt_key(x) for x in cyc)
            raise ValueError(msg)


def validate_state_edge_targets_writable_or_raise(
    graph: F8RuntimeGraph,
    *,
    local_service_id: str | None = None,
) -> None:
    """
    Validate that state-edge targets are writable.

    If `local_service_id` is provided, only validate edges whose `toServiceId`
    equals that service id.
    """
    access_map: dict[tuple[str, str, str], F8StateAccess] = {}
    for n in list(graph.nodes or []):
        service_id = str(n.serviceId or "").strip()
        node_id = str(n.nodeId or "").strip()
        if not service_id or not node_id:
            continue
        for sf in list(n.stateFields or []):
            field = str(sf.name or "").strip()
            if not field:
                continue
            access = sf.access
            if isinstance(access, F8StateAccess):
                access_map[(service_id, node_id, field)] = access

    service_filter = str(local_service_id or "").strip()
    for e in list(graph.edges or []):
        if e.kind != F8EdgeKindEnum.state:
            continue
        from_service = str(e.fromServiceId or "").strip()
        from_node = str(e.fromOperatorId or "").strip()
        from_field = str(e.fromPort or "").strip()
        to_service = str(e.toServiceId or "").strip()
        to_node = str(e.toOperatorId or "").strip()
        to_field = str(e.toPort or "").strip()
        if not from_service or not from_node or not from_field or not to_service or not to_node or not to_field:
            continue
        if service_filter and to_service != service_filter:
            continue
        from_access = access_map.get((from_service, from_node, from_field))
        if from_access is None:
            raise ValueError(
                f"state edge source field not found: {from_node}.{from_field}"
            )
        if from_access == F8StateAccess.wo:
            raise ValueError(
                f"state edge source is write-only: {from_node}.{from_field} ({from_access.value})"
            )
        access = access_map.get((to_service, to_node, to_field))
        if access is None:
            raise ValueError(
                f"state edge target field not found: {to_node}.{to_field}"
            )
        if access == F8StateAccess.ro:
            raise ValueError(
                f"state edge targets non-writable field: {to_node}.{to_field} ({access.value})"
            )
