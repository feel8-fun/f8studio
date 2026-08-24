from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from typing import Any

import pytest

from f8pysdk.bus import ServiceBus, ServiceBusConfig
from f8pysdk.codec import decode_as, encode_obj, validate_as
from f8pysdk.f8_naming import cmd_channel_key, svc_endpoint_key
from f8pysdk.nodes import RuntimeNode, ServiceNode
from f8pysdk.service_runtime_tools.deploy.readiness import wait_service_ready
from f8pysdk.specs import (
    F8ActivateRequest,
    F8ActiveReply,
    F8Command,
    F8CommandInvokeReply,
    F8CommandInvokeRequest,
    F8CommandParam,
    F8DeactivateRequest,
    F8DataPortSpec,
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8EmptyArgs,
    F8RuntimeGraph,
    F8RuntimeNode,
    F8ServiceSpec,
    F8SetRungraphArgs,
    F8SetRungraphReply,
    F8SetRungraphRequest,
    F8SetStateArgs,
    F8SetStateReply,
    F8SetStateRequest,
    F8StateAccess,
    F8StateSpec,
    F8StatusReply,
    F8StatusRequest,
    F8TerminateReply,
    F8TerminateRequest,
    string_schema,
)
from f8pysdk.zenoh_transport import ZenohTransport, ZenohTransportConfig


def _sid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class _CommandServiceNode(ServiceNode):
    def __init__(self, node_id: str) -> None:
        super().__init__(node_id=node_id)
        self.spec = F8ServiceSpec(
            serviceClass="test.zenoh.service",
            label="Zenoh Service",
            commands=[
                F8Command(
                    name="echo",
                    params=[F8CommandParam(name="text", valueSchema=string_schema())],
                )
            ],
        )
        self.command_calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    async def on_command(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        *,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        call_args = dict(args or {})
        call_meta = dict(meta or {})
        self.command_calls.append((str(name), call_args, call_meta))
        return {"echo": call_args.get("text"), "source": call_meta.get("source")}


class _SinkNode(RuntimeNode):
    def __init__(self, node_id: str) -> None:
        super().__init__(node_id=node_id, data_in_ports=["in"], state_fields=["input"])
        self.data_calls: list[tuple[str, Any, int | None]] = []
        self.state_calls: list[tuple[str, Any, int | None]] = []

    async def on_data(self, port: str, value: Any, *, ts_ms: int | None = None) -> None:
        self.data_calls.append((str(port), value, ts_ms))

    async def on_state(self, field: str, value: Any, *, ts_ms: int | None = None) -> None:
        self.state_calls.append((str(field), value, ts_ms))


def _runtime_node(
    *,
    node_id: str,
    service_id: str,
    state_fields: list[F8StateSpec] | None = None,
    data_in: list[str] | None = None,
    data_out: list[str] | None = None,
) -> F8RuntimeNode:
    data_in_specs = [F8DataPortSpec(name=str(name), valueSchema=string_schema()) for name in list(data_in or [])]
    data_out_specs = [F8DataPortSpec(name=str(name), valueSchema=string_schema()) for name in list(data_out or [])]
    return F8RuntimeNode(
        nodeId=node_id,
        serviceId=service_id,
        serviceClass=f"test.{service_id}",
        operatorClass=f"test.{node_id}",
        dataInPorts=data_in_specs,
        dataOutPorts=data_out_specs,
        stateFields=list(state_fields or []),
    )


def _state_field(name: str) -> F8StateSpec:
    return F8StateSpec(name=str(name), valueSchema=string_schema(), access=F8StateAccess.rw)


def _graph(service_a: str, service_b: str) -> F8RuntimeGraph:
    node_src = _runtime_node(
        node_id="src",
        service_id=service_a,
        state_fields=[_state_field("value")],
        data_out=["out"],
    )
    node_sink = _runtime_node(
        node_id="sink",
        service_id=service_b,
        state_fields=[_state_field("input")],
        data_in=["in"],
    )
    data_edge = F8Edge(
        edgeId="data-src-sink",
        fromServiceId=service_a,
        fromOperatorId="src",
        fromPort="out",
        toServiceId=service_b,
        toOperatorId="sink",
        toPort="in",
        kind=F8EdgeKindEnum.data,
        strategy=F8EdgeStrategyEnum.latest,
    )
    state_edge = F8Edge(
        edgeId="state-src-sink",
        fromServiceId=service_a,
        fromOperatorId="src",
        fromPort="value",
        toServiceId=service_b,
        toOperatorId="sink",
        toPort="input",
        kind=F8EdgeKindEnum.state,
        strategy=F8EdgeStrategyEnum.latest,
    )
    return F8RuntimeGraph(
        graphId=f"zenoh-it-{service_a}-{service_b}",
        revision="r1",
        nodes=[node_src, node_sink],
        edges=[data_edge, state_edge],
    )


async def _wait_until(predicate: Callable[[], bool], *, timeout_s: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + float(timeout_s)
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise TimeoutError("condition was not satisfied")


async def _request_endpoint(
    transport: ZenohTransport,
    *,
    service_id: str,
    endpoint: str,
    payload: Any,
    reply_type: type[Any],
) -> Any:
    raw = await transport.request(
        svc_endpoint_key(service_id, endpoint),
        encode_obj(payload),
        timeout=2.0,
        raise_on_error=True,
    )
    assert raw is not None
    return decode_as(raw, reply_type)


async def _request_cmd(
    transport: ZenohTransport,
    *,
    service_id: str,
    payload: F8CommandInvokeRequest,
) -> F8CommandInvokeReply:
    raw = await transport.request(
        cmd_channel_key(service_id),
        encode_obj(payload),
        timeout=2.0,
        raise_on_error=True,
    )
    assert raw is not None
    return decode_as(raw, F8CommandInvokeReply)


def test_two_python_service_buses_roundtrip_over_zenoh() -> None:
    pytest.importorskip("zenoh")

    async def _run() -> None:
        service_a = _sid("zenohA")
        service_b = _sid("zenohB")
        client_id = _sid("zenohClient")

        bus_a = ServiceBus(
            ServiceBusConfig(
                service_id=service_a,
                service_class="test.zenoh.serviceA",
                bus_backend="zenoh",
                monitor_enabled=False,
            )
        )
        bus_b = ServiceBus(
            ServiceBusConfig(
                service_id=service_b,
                service_class="test.zenoh.serviceB",
                bus_backend="zenoh",
                data_delivery="callback",
                monitor_enabled=False,
            )
        )
        service_node = _CommandServiceNode(service_a)
        src_node = RuntimeNode(node_id="src", data_out_ports=["out"], state_fields=["value"])
        sink_node = _SinkNode("sink")
        bus_a.register_node(service_node)
        bus_a.register_node(src_node)
        bus_b.register_node(sink_node)

        client = ZenohTransport(ZenohTransportConfig(service_id=client_id))
        await bus_a.start()
        await bus_b.start()
        await client.connect()
        try:
            await wait_service_ready(client, service_id=service_a, timeout_s=2.0)
            await wait_service_ready(client, service_id=service_b, timeout_s=2.0)

            graph = _graph(service_a, service_b)
            await bus_a.publish_state_runtime("src", "value", "initial", ts_ms=1)

            reply_a = await _request_endpoint(
                client,
                service_id=service_a,
                endpoint="set_rungraph",
                payload=F8SetRungraphRequest(
                    reqId="set-rungraph-a",
                    args=F8SetRungraphArgs(graph=graph),
                    meta={"source": "test"},
                ),
                reply_type=F8SetRungraphReply,
            )
            reply_b = await _request_endpoint(
                client,
                service_id=service_b,
                endpoint="set_rungraph",
                payload=F8SetRungraphRequest(
                    reqId="set-rungraph-b",
                    args=F8SetRungraphArgs(graph=graph),
                    meta={"source": "test"},
                ),
                reply_type=F8SetRungraphReply,
            )
            assert reply_a.ok is True
            assert reply_b.ok is True

            await _wait_until(
                lambda: any(field == "input" and value == "initial" for field, value, _ts in sink_node.state_calls)
            )

            await bus_a.publish_state_runtime("src", "value", "updated", ts_ms=2)
            await _wait_until(
                lambda: any(field == "input" and value == "updated" for field, value, _ts in sink_node.state_calls)
            )

            await bus_a.emit_data("src", "out", {"frame": 1}, ts_ms=3)
            await _wait_until(lambda: ("in", {"frame": 1}, 3) in sink_node.data_calls)

            set_state_reply = await _request_endpoint(
                client,
                service_id=service_b,
                endpoint="set_state",
                payload=F8SetStateRequest(
                    reqId="set-state-b",
                    args=validate_as(
                        F8SetStateArgs,
                        {"nodeId": "sink", "field": "input", "value": "endpoint"},
                    ),
                    meta={"source": "test"},
                ),
                reply_type=F8SetStateReply,
            )
            assert set_state_reply.ok is True
            await _wait_until(
                lambda: any(field == "input" and value == "endpoint" for field, value, _ts in sink_node.state_calls)
            )

            status_reply = await _request_endpoint(
                client,
                service_id=service_a,
                endpoint="status",
                payload=F8StatusRequest(reqId="status-a", args=F8EmptyArgs(), meta={"source": "test"}),
                reply_type=F8StatusReply,
            )
            assert status_reply.ok is True
            assert status_reply.result is not None
            assert status_reply.result.serviceId == service_a
            assert status_reply.result.serviceClass == "test.zenoh.serviceA"
            assert status_reply.result.runtimeInstanceId
            assert status_reply.result.active is True

            deactivate_reply = await _request_endpoint(
                client,
                service_id=service_a,
                endpoint="deactivate",
                payload=F8DeactivateRequest(reqId="deactivate-a", args=F8EmptyArgs(), meta={"source": "test"}),
                reply_type=F8ActiveReply,
            )
            assert deactivate_reply.ok is True
            assert bus_a.active is False

            activate_reply = await _request_endpoint(
                client,
                service_id=service_a,
                endpoint="activate",
                payload=F8ActivateRequest(reqId="activate-a", args=F8EmptyArgs(), meta={"source": "test"}),
                reply_type=F8ActiveReply,
            )
            assert activate_reply.ok is True
            assert bus_a.active is True

            cmd_reply = await _request_cmd(
                client,
                service_id=service_a,
                payload=F8CommandInvokeRequest(
                    reqId="cmd-a",
                    call="echo",
                    args={"text": "hello"},
                    meta={"source": "test"},
                ),
            )
            assert cmd_reply.ok is True
            assert cmd_reply.result == {"echo": "hello", "source": "test"}
            assert service_node.command_calls == [("echo", {"text": "hello"}, {"source": "test"})]

            terminate_reply = await _request_endpoint(
                client,
                service_id=service_a,
                endpoint="terminate",
                payload=F8TerminateRequest(reqId="terminate-a", args=F8EmptyArgs(), meta={"source": "test"}),
                reply_type=F8TerminateReply,
            )
            assert terminate_reply.ok is True
            await asyncio.wait_for(bus_a.wait_terminate(), timeout=1.0)
        finally:
            await client.close()
            await bus_b.stop()
            await bus_a.stop()

    asyncio.run(_run())
