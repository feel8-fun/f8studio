from __future__ import annotations

from typing import Iterable

from .generated import F8Edge, F8EdgeKindEnum, F8RuntimeGraph, F8RuntimeNode, F8StateAccess


def _state_key(*, service_id: str, node_id: str, field: str) -> tuple[str, str, str]:
    return (str(service_id), str(node_id), str(field))


def _fmt_key(k: tuple[str, str, str]) -> str:
    sid, nid, fld = k
    return f"{sid}.{nid}.{fld}"


def validate_exec_edges_or_raise(
    graph: F8RuntimeGraph,
) -> None:
    """
    Validate exec-edge wiring constraints on a global rungraph.

    Enforced:
    - exec edges must be intra-service (`fromServiceId == toServiceId`)
    - exec endpoints must be operator nodes (not service nodes)
    - each exec out port has at most one downstream
    - each exec in port has at most one upstream
    """
    nodes_by_key: dict[tuple[str, str], F8RuntimeNode] = {}
    for n in list(graph.nodes or []):
        service_id = str(n.serviceId or "").strip()
        node_id = str(n.nodeId or "").strip()
        if not service_id or not node_id:
            continue
        nodes_by_key[(service_id, node_id)] = n

    out_map: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    in_map: dict[tuple[str, str, str], tuple[str, str, str]] = {}

    for e in list(graph.edges or []):
        if e.kind != F8EdgeKindEnum.exec:
            continue
        from_sid = str(e.fromServiceId or "").strip()
        to_sid = str(e.toServiceId or "").strip()
        from_op = str(e.fromOperatorId or "").strip()
        to_op = str(e.toOperatorId or "").strip()
        from_port = str(e.fromPort or "").strip()
        to_port = str(e.toPort or "").strip()
        if not (from_sid and to_sid and from_op and to_op and from_port and to_port):
            continue

        if from_sid != to_sid:
            raise ValueError(
                f"cross-service exec edge is not allowed: {from_sid}.{from_op}.{from_port} -> {to_sid}.{to_op}.{to_port}"
            )

        from_node = nodes_by_key.get((from_sid, from_op))
        if from_node is None:
            raise ValueError(f"exec edge source node not found: {from_sid}.{from_op}")
        to_node = nodes_by_key.get((to_sid, to_op))
        if to_node is None:
            raise ValueError(f"exec edge target node not found: {to_sid}.{to_op}")

        from_operator_class = str(from_node.operatorClass or "").strip()
        if not from_operator_class:
            raise ValueError(f"exec edge source must be operator node: {from_sid}.{from_op}")
        to_operator_class = str(to_node.operatorClass or "").strip()
        if not to_operator_class:
            raise ValueError(f"exec edge target must be operator node: {to_sid}.{to_op}")

        from_key = (from_sid, from_op, from_port)
        to_key = (to_sid, to_op, to_port)

        prev_to = out_map.get(from_key)
        if prev_to is not None and prev_to != to_key:
            raise ValueError(
                "exec out port must be single-connected: "
                f"{_fmt_key(from_key)} -> {_fmt_key(prev_to)} and {_fmt_key(to_key)}"
            )
        out_map[from_key] = to_key

        prev_from = in_map.get(to_key)
        if prev_from is not None and prev_from != from_key:
            raise ValueError(
                "exec in port must be single-connected: "
                f"{_fmt_key(to_key)} <- {_fmt_key(prev_from)} and {_fmt_key(from_key)}"
            )
        in_map[to_key] = from_key


def validate_data_edges_or_raise(
    graph: F8RuntimeGraph,
) -> None:
    """
    Validate data-edge wiring constraints on a global rungraph.

    Enforced:
    - data input port is single-upstream (multiple upstream to one input is invalid)
    - data output ports are unrestricted (fan-out allowed)
    """
    inbound_map: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    for e in list(graph.edges or []):
        if e.kind != F8EdgeKindEnum.data:
            continue
        from_sid = str(e.fromServiceId or "").strip()
        to_sid = str(e.toServiceId or "").strip()
        from_op = str(e.fromOperatorId or "").strip()
        to_op = str(e.toOperatorId or "").strip()
        from_port = str(e.fromPort or "").strip()
        to_port = str(e.toPort or "").strip()
        if not (from_sid and to_sid and from_port and to_port):
            continue

        from_node = from_op or f"$service:{from_sid}"
        to_node = to_op or f"$service:{to_sid}"
        from_key = (from_sid, from_node, from_port)
        to_key = (to_sid, to_node, to_port)

        prev_from = inbound_map.get(to_key)
        if prev_from is not None and prev_from != from_key:
            raise ValueError(
                "multiple upstreams for data input: "
                f"{_fmt_key(to_key)} <- {_fmt_key(prev_from)} and {_fmt_key(from_key)}"
            )
        inbound_map[to_key] = from_key


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

    Note:
    - In per-service rungraphs, cross-service inbound edges may be present as
      half-edges while the upstream node definition is intentionally absent.
      In that case, source-access validation cannot be performed locally and
      should be skipped. Global graph validation still checks both ends.
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
            # Per-service apply: for cross-service inbound state edges, the
            # source node may be absent in the local half-graph.
            if service_filter and from_service != service_filter:
                from_access = None
            else:
                raise ValueError(
                    f"state edge source field not found: {from_node}.{from_field}"
                )
        if from_access is None:
            # Cross-service source in local half-graph: can't validate access
            # here; rely on global validation at compile/deploy time.
            pass
        elif from_access == F8StateAccess.wo:
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
