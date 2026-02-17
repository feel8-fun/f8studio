from __future__ import annotations

import json
import time
import uuid
from typing import Any

from f8pysdk import F8Edge, F8EdgeKindEnum, F8EdgeStrategyEnum, F8RuntimeGraph, F8RuntimeNode, F8StateAccess
from f8pysdk.nats_naming import ensure_token, kv_bucket_for_service, kv_key_rungraph, new_id, svc_endpoint_subject, svc_micro_name
from f8pysdk.nats_transport import NatsTransport, NatsTransportConfig
from f8pysdk.service_ready import wait_service_ready


def _now_rev() -> str:
    return str(int(time.time() * 1000))


def _port_kind(name: str) -> str | None:
    n = str(name or "")
    if n.startswith("[E]") or n.endswith("[E]"):
        return "exec"
    if n.startswith("[D]") or n.endswith("[D]"):
        return "data"
    if n.startswith("[S]") or n.endswith("[S]"):
        return "state"
    return None


def _raw_port_name(name: str) -> str:
    n = str(name or "")
    for prefix in ("[E]", "[D]", "[S]"):
        if n.startswith(prefix):
            n = n[len(prefix) :]
    for suffix in ("[E]", "[D]", "[S]"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n.strip()

def _port_name(port: Any) -> str:
    """
    NodeGraphQt `Port` typically exposes `name()` (method).
    """
    try:
        return str(port.name() or "")
    except (AttributeError, RuntimeError, TypeError):
        pass
    try:
        return str(port.name or "")
    except (AttributeError, RuntimeError, TypeError):
        return ""


def export_runtime_graph(
    node_graph: Any,
    *,
    service_id: str,
    include_nodes: list[Any] | None = None,
    graph_id: str | None = None,
    revision: str | None = None,
) -> F8RuntimeGraph:
    """
    Best-effort export from a NodeGraphQt graph into a `F8RuntimeGraph`.
    """
    service_id = ensure_token(service_id, label="service_id")
    gid = ensure_token(graph_id or uuid.uuid4().hex, label="graphId")
    rev = str(revision or _now_rev()).strip() or _now_rev()

    nodes: list[F8RuntimeNode] = []
    edges: list[F8Edge] = []
    node_service_ids: dict[Any, str] = {}
    node_is_service: dict[Any, bool] = {}

    # Nodes.
    if include_nodes is not None:
        all_nodes = list(include_nodes)
    else:
        try:
            all_nodes = list(node_graph.all_nodes())
        except Exception:
            all_nodes = []

    id_map: dict[Any, str] = {}
    for n in all_nodes:
        try:
            spec = n.spec
        except (AttributeError, RuntimeError, TypeError):
            continue
        try:
            node_id = ensure_token(str(n.id or uuid.uuid4().hex).replace(".", "_"), label="nodeId")
        except Exception:
            node_id = uuid.uuid4().hex
        id_map[n] = node_id

        sid = None
        if not sid:
            try:
                sid = n.get_property("serviceId")
            except Exception:
                sid = None
        sid_s = str(sid or "").strip()
        node_service_ids[n] = sid_s or service_id

        service_class = str(spec.serviceClass or "").strip()
        operator_class = str(spec.operatorClass).strip() if spec.operatorClass is not None else None
        is_service_node = operator_class is None
        node_is_service[n] = is_service_node

        # Service/container nodes use `nodeId == serviceId`.
        if is_service_node:
            try:
                node_id = ensure_token(node_service_ids[n], label="nodeId")
            except Exception:
                node_id = node_service_ids[n] or uuid.uuid4().hex
            id_map[n] = node_id

        exec_in = list(spec.execInPorts or [])
        exec_out = list(spec.execOutPorts or [])
        data_in = list(spec.dataInPorts or [])
        data_out = list(spec.dataOutPorts or [])
        state_fields = list(spec.stateFields or [])

        state_values: dict[str, Any] = {}
        for f in state_fields:
            name = str(f.name or "").strip()
            if not name:
                continue
            # Do not include read-only state values in the rungraph snapshot.
            if f.access == F8StateAccess.ro:
                continue
            try:
                state_values[name] = n.get_property(name)
            except (AttributeError, KeyError, RuntimeError, TypeError):
                pass

        nodes.append(
            F8RuntimeNode(
                nodeId=node_id,
                serviceId=node_service_ids[n],
                serviceClass=service_class,
                operatorClass=operator_class,
                execInPorts=[str(p) for p in exec_in],
                execOutPorts=[str(p) for p in exec_out],
                dataInPorts=data_in,
                dataOutPorts=data_out,
                stateFields=state_fields,
                stateValues=state_values or None,
            )
        )

    # Edges (best-effort via per-port connected ports).
    for src_node in all_nodes:
        try:
            outs = list(src_node.output_ports())
        except Exception:
            outs = []
        for out_port in outs:
            try:
                out_name = _port_name(out_port)
            except Exception:
                out_name = ""
            kind = _port_kind(out_name)
            if not kind:
                continue
            try:
                connected = list(out_port.connected_ports())
            except Exception:
                connected = []
            for in_port in connected:
                try:
                    dst_node = in_port.node()
                except Exception:
                    dst_node = None
                if dst_node is None:
                    continue
                from_id = id_map.get(src_node)
                to_id = id_map.get(dst_node)
                if not from_id or not to_id:
                    continue
                try:
                    in_name = _port_name(in_port)
                except Exception:
                    in_name = ""

                edge_kind = {
                    "exec": F8EdgeKindEnum.exec,
                    "data": F8EdgeKindEnum.data,
                    "state": F8EdgeKindEnum.state,
                }[kind]

                edges.append(
                    F8Edge(
                        edgeId=uuid.uuid4().hex,
                        fromServiceId=node_service_ids.get(src_node, service_id),
                        fromOperatorId=(None if node_is_service.get(src_node, False) else from_id),
                        fromPort=_raw_port_name(out_name),
                        toServiceId=node_service_ids.get(dst_node, service_id),
                        toOperatorId=(None if node_is_service.get(dst_node, False) else to_id),
                        toPort=_raw_port_name(in_name),
                        kind=edge_kind,
                        strategy=F8EdgeStrategyEnum.latest,
                    )
                )

    return F8RuntimeGraph(graphId=gid, revision=rev, nodes=nodes, edges=edges)


async def deploy_to_service(*, service_id: str, nats_url: str, graph: F8RuntimeGraph) -> None:
    service_id = ensure_token(service_id, label="service_id")
    bucket = kv_bucket_for_service(service_id)
    key = kv_key_rungraph()

    tr = NatsTransport(NatsTransportConfig(url=str(nats_url).strip(), kv_bucket=bucket))
    await tr.connect()
    try:
        await wait_service_ready(tr, timeout_s=6.0)
        graph_payload = graph.model_dump(mode="json", by_alias=True)
        payload = json.dumps(graph_payload, ensure_ascii=False, default=str).encode("utf-8")
        # Endpoint-only mode: deploy via service endpoint (allows validation/rejection).
        req = {"reqId": new_id(), "args": {"graph": graph_payload}, "meta": {"source": "deploy"}}
        req_bytes = json.dumps(req, ensure_ascii=False, default=str).encode("utf-8")
        resp_raw = await tr.request(
            svc_endpoint_subject(service_id, "set_rungraph"),
            req_bytes,
            timeout=2.0,
            raise_on_error=True,
        )
        if not resp_raw:
            raise RuntimeError("set_rungraph request failed: empty response")
        try:
            resp = json.loads(resp_raw.decode("utf-8"))
        except Exception:
            resp = {}
        if isinstance(resp, dict) and resp.get("ok") is True:
            return
        if isinstance(resp, dict) and resp.get("ok") is False:
            msg = ""
            try:
                msg = str((resp.get("error") or {}).get("message") or "")
            except (AttributeError, RuntimeError, TypeError, ValueError):
                msg = ""
            raise RuntimeError(msg or "set_rungraph rejected")
        raise RuntimeError("invalid set_rungraph response")
    finally:
        await tr.close()
