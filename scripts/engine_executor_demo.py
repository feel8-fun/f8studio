from __future__ import annotations

import asyncio
import json
import os
import uuid

import nats  # type: ignore[import-not-found]

from f8pysdk import (
    F8DataPortSpec,
    F8Edge,
    F8EdgeKindEnum,
    F8EdgeStrategyEnum,
    F8OperatorSpec,
    F8StateAccess,
    F8StateSpec,
    number_schema,
    string_schema,
)

from f8pyengineqt.engine.engine_service_process import EngineServiceProcess, EngineServiceProcessConfig
from f8pyengineqt.engine.nats_naming import cmd_subject, kv_bucket_for_service, kv_key_rungraph
from f8pyengineqt.runtime.nats_transport import NatsTransport, NatsTransportConfig


async def _put_rungraph(*, nats_url: str, service_id: str, payload: dict) -> None:
    bucket = kv_bucket_for_service(service_id)
    t = NatsTransport(NatsTransportConfig(url=nats_url, kv_bucket=bucket))
    await t.connect()
    try:
        await t.kv_put(kv_key_rungraph(), json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    finally:
        await t.close()


async def main() -> None:
    nats_url = (os.environ.get("F8_NATS_URL") or "nats://127.0.0.1:4222").strip()
    service_id = uuid.uuid4().hex

    node_start = uuid.uuid4().hex
    node_a = uuid.uuid4().hex
    node_b = uuid.uuid4().hex
    node_add = uuid.uuid4().hex
    node_log = uuid.uuid4().hex

    spec_start = F8OperatorSpec(operatorClass="f8/start", version="0.0.1", label="Start", execOutPorts=["exec"])
    spec_const = F8OperatorSpec(
        operatorClass="f8/constant",
        version="0.0.1",
        label="Constant",
        execInPorts=["exec"],
        execOutPorts=["exec"],
        dataOutPorts=[F8DataPortSpec(name="value", valueSchema=number_schema())],
        states=[
            F8StateSpec(name="value2", label="Value", valueSchema=number_schema(default=1.0), access=F8StateAccess.rw),
        ],
    )
    spec_add = F8OperatorSpec(
        operatorClass="f8/add",
        version="0.0.1",
        label="Add",
        execInPorts=["exec"],
        dataInPorts=[F8DataPortSpec(name="a", valueSchema=number_schema()), F8DataPortSpec(name="b", valueSchema=number_schema())],
        dataOutPorts=[F8DataPortSpec(name="sum", valueSchema=number_schema())],
    )
    spec_log = F8OperatorSpec(
        operatorClass="f8/log",
        version="0.0.1",
        label="Log",
        dataInPorts=[F8DataPortSpec(name="value", valueSchema=number_schema())],
        states=[F8StateSpec(name="label", label="Label", valueSchema=string_schema(default="Log"), access=F8StateAccess.ro)],
    )

    rungraph = {
        "nodes": [
            {"id": node_start, "operatorClass": spec_start.operatorClass, "spec": spec_start.model_dump(mode="json"), "state": {}},
            {
                "id": node_a,
                "operatorClass": spec_const.operatorClass,
                "spec": spec_const.model_dump(mode="json"),
                "state": {"value2": 3.0},
            },
            {
                "id": node_b,
                "operatorClass": spec_const.operatorClass,
                "spec": spec_const.model_dump(mode="json"),
                "state": {"value2": 7.0},
            },
            {"id": node_add, "operatorClass": spec_add.operatorClass, "spec": spec_add.model_dump(mode="json"), "state": {}},
            {
                "id": node_log,
                "operatorClass": spec_log.operatorClass,
                "spec": spec_log.model_dump(mode="json"),
                "state": {"label": "Sum"},
            },
        ],
        "edges": [
            F8Edge(
                edgeId=uuid.uuid4().hex,
                fromServiceId=service_id,
                fromOperatorId=node_start,
                fromPort="exec",
                toServiceId=service_id,
                toOperatorId=node_a,
                toPort="exec",
                kind=F8EdgeKindEnum.exec,
                strategy=F8EdgeStrategyEnum.latest,
            ).model_dump(by_alias=True, mode="json"),
            F8Edge(
                edgeId=uuid.uuid4().hex,
                fromServiceId=service_id,
                fromOperatorId=node_a,
                fromPort="exec",
                toServiceId=service_id,
                toOperatorId=node_b,
                toPort="exec",
                kind=F8EdgeKindEnum.exec,
                strategy=F8EdgeStrategyEnum.latest,
            ).model_dump(by_alias=True, mode="json"),
            F8Edge(
                edgeId=uuid.uuid4().hex,
                fromServiceId=service_id,
                fromOperatorId=node_b,
                fromPort="exec",
                toServiceId=service_id,
                toOperatorId=node_add,
                toPort="exec",
                kind=F8EdgeKindEnum.exec,
                strategy=F8EdgeStrategyEnum.latest,
            ).model_dump(by_alias=True, mode="json"),
            F8Edge(
                edgeId=uuid.uuid4().hex,
                fromServiceId=service_id,
                fromOperatorId=node_a,
                fromPort="value",
                toServiceId=service_id,
                toOperatorId=node_add,
                toPort="a",
                kind=F8EdgeKindEnum.data,
                strategy=F8EdgeStrategyEnum.latest,
            ).model_dump(by_alias=True, mode="json"),
            F8Edge(
                edgeId=uuid.uuid4().hex,
                fromServiceId=service_id,
                fromOperatorId=node_b,
                fromPort="value",
                toServiceId=service_id,
                toOperatorId=node_add,
                toPort="b",
                kind=F8EdgeKindEnum.data,
                strategy=F8EdgeStrategyEnum.latest,
            ).model_dump(by_alias=True, mode="json"),
            F8Edge(
                edgeId=uuid.uuid4().hex,
                fromServiceId=service_id,
                fromOperatorId=node_add,
                fromPort="sum",
                toServiceId=service_id,
                toOperatorId=node_log,
                toPort="value",
                kind=F8EdgeKindEnum.data,
                strategy=F8EdgeStrategyEnum.latest,
            ).model_dump(by_alias=True, mode="json"),
        ],
    }

    print(f"[demo] nats={nats_url}")
    print(f"[demo] serviceId={service_id} bucket={kv_bucket_for_service(service_id)}")
    await _put_rungraph(nats_url=nats_url, service_id=service_id, payload=rungraph)

    proc = EngineServiceProcess(EngineServiceProcessConfig(service_id=service_id, nats_url=nats_url))
    task = asyncio.create_task(proc.run(), name="engine_service_process")
    await asyncio.sleep(0.5)

    nc = await nats.connect(servers=[nats_url], connect_timeout=1)
    try:
        await nc.publish(cmd_subject(service_id, "run"), json.dumps({"mode": "once"}).encode("utf-8"))
        await nc.flush(timeout=1.0)
    finally:
        await nc.drain()

    await asyncio.sleep(0.8)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
