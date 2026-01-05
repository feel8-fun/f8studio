from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

from f8pysdk import (
    F8DataPortSpec,
    F8EdgeKindEmum,
    F8EdgeScopeEnum,
    F8EdgeSpec,
    F8EdgeStrategyEnum,
    F8OperatorSpec,
    number_schema,
)

from f8pyengineqt.engine.nats_naming import data_subject, kv_bucket_for_service, kv_key_topology
from f8pyengineqt.runtime import NatsTransport, NatsTransportConfig, ServiceRuntime, ServiceRuntimeConfig, ServiceRuntimeNode


class Producer(ServiceRuntimeNode):
    async def run(self) -> None:
        i = 0.0
        while True:
            await self.emit("out", i)
            i += 1.0
            await asyncio.sleep(1.0)


class Consumer(ServiceRuntimeNode):
    async def run(self) -> None:
        # Pull faster than producer publish rate to demonstrate `repeat`.
        while True:
            v = await self.pull("in")
            print(f"[consumer] in={v}")
            await asyncio.sleep(0.2)


async def _put_topology(*, nats_url: str, service_id: str, payload: dict[str, Any]) -> None:
    bucket = kv_bucket_for_service(service_id)
    t = NatsTransport(NatsTransportConfig(url=nats_url, kv_bucket=bucket))
    await t.connect()
    try:
        await t.kv_put(kv_key_topology(service_id), json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    finally:
        await t.close()


async def main() -> None:
    """End-to-end demo using pull-based inputs (no SDK tick)."""
    nats_url = (os.environ.get("F8_NATS_URL") or "nats://127.0.0.1:4222").strip()

    svc_a = uuid.uuid4().hex
    svc_b = uuid.uuid4().hex
    node_a = uuid.uuid4().hex
    node_b = uuid.uuid4().hex
 
    subj = data_subject(svc_a, from_node_id=node_a, port_id="out")
    bucket_a = kv_bucket_for_service(svc_a)
    bucket_b = kv_bucket_for_service(svc_b)
    print(f"[demo] nats={nats_url}")
    print(f"[demo] svc_a={svc_a} bucket={bucket_a}")
    print(f"[demo] svc_b={svc_b} bucket={bucket_b}")
    print(f"[demo] subject={subj}")

    spec_a = F8OperatorSpec(
        operatorClass="demo.producer",
        version="0.0.1",
        label="Producer",
        dataOutPorts=[F8DataPortSpec(name="out", valueSchema=number_schema())],
    )
    spec_b = F8OperatorSpec(
        operatorClass="demo.consumer",
        version="0.0.1",
        label="Consumer",
        dataInPorts=[F8DataPortSpec(name="in", valueSchema=number_schema())],
    )

    topo_a = {
        "nodes": [{"id": node_a, "operatorClass": spec_a.operatorClass, "spec": spec_a.model_dump(mode="json"), "state": {}}],
        "edges": [],
    }
    topo_b = {
        "nodes": [{"id": node_b, "operatorClass": spec_b.operatorClass, "spec": spec_b.model_dump(mode="json"), "state": {}}],
        "edges": [
            F8EdgeSpec(
                from_=node_a,
                fromPort="out",
                to=node_b,
                toPort="in",
                kind=F8EdgeKindEmum.data,
                scope=F8EdgeScopeEnum.cross,
                strategy=F8EdgeStrategyEnum.repeat,
                timeoutMs=2500,
                queueSize=8,
                edgeId=uuid.uuid4().hex,
                direction="in",
                subject=subj,
                peerServiceId=svc_a,
            ).model_dump(by_alias=True, mode="json")
        ],
    }

    await _put_topology(nats_url=nats_url, service_id=svc_a, payload=topo_a)
    await _put_topology(nats_url=nats_url, service_id=svc_b, payload=topo_b)

    # Pass explicit buckets to avoid environment overrides (eg. F8_NATS_BUCKET).
    runtime_a = ServiceRuntime(
        ServiceRuntimeConfig(service_id=svc_a, nats_url=nats_url, kv_bucket=bucket_a, publish_all_data=True)
    )
    runtime_b = ServiceRuntime(
        ServiceRuntimeConfig(service_id=svc_b, nats_url=nats_url, kv_bucket=bucket_b, publish_all_data=True)
    )

    producer = Producer(node_id=node_a, data_out_ports=["out"])
    consumer = Consumer(node_id=node_b, data_in_ports=["in"])

    runtime_a.register_node(producer)
    runtime_b.register_node(consumer)

    await runtime_a.start()
    await runtime_b.start()

    prod_task = asyncio.create_task(producer.run(), name="producer.run")
    cons_task = asyncio.create_task(consumer.run(), name="consumer.run")
    try:
        await asyncio.sleep(8.0)
    finally:
        prod_task.cancel()
        cons_task.cancel()
        await runtime_a.stop()
        await runtime_b.stop()


if __name__ == "__main__":
    asyncio.run(main())
