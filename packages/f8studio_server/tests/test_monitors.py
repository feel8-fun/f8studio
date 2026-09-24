import asyncio

import msgspec

from f8pysdk.codec import encode_obj
from f8pysdk.specs import (
    F8MonitorCpu,
    F8MonitorError,
    F8MonitorFrame,
    F8MonitorGpu,
    F8MonitorMemory,
    F8MonitorQueue,
    F8MonitorSnapshot,
    F8MonitorTiming,
    SchemaVersion1,
)
from f8studio_server.events import EventJournal
from f8studio_server.monitors import RuntimeMonitorStore


def test_monitor_store_decodes_data_envelope_and_keeps_latest_snapshot() -> None:
    async def scenario() -> None:
        events = EventJournal(server_epoch="epoch1")
        store = RuntimeMonitorStore(events)
        snapshot = F8MonitorSnapshot(
            schemaVersion=SchemaVersion1.f8monitor_1,
            serviceId="engine1",
            serviceClass="f8.pyengine",
            nodeId="engine1",
            tsMs=123,
            alive=True,
            ready=True,
            active=True,
            uptimeMs=1000,
            cpu=F8MonitorCpu(),
            memory=F8MonitorMemory(),
            gpu=F8MonitorGpu(),
            frame=F8MonitorFrame(observed=2, processed=1),
            timing=F8MonitorTiming(),
            queue=F8MonitorQueue(),
            error=F8MonitorError(),
        )
        await store.ingest(
            "f8/svc/engine1/nodes/engine1/data/monitor",
            encode_obj({"value": snapshot, "ts": 123}),
        )

        stored = await store.snapshot()
        assert len(stored) == 1
        assert stored[0].serviceId == "engine1"
        assert stored[0].nodeId == "engine1"
        assert stored[0].tsMs == 123
        assert stored[0].frame.observed == 2
        assert stored[0].frame.processed == 1

        isolated = RuntimeMonitorStore(events, studio_service_id="studio_local")
        await isolated.ingest(
            "f8/svc/studio/nodes/studio/data/monitor",
            encode_obj({"value": msgspec.structs.replace(snapshot, serviceId="studio", nodeId="studio"), "ts": 123}),
        )
        await isolated.ingest(
            "f8/svc/studio_remote/nodes/studio_remote/data/monitor",
            encode_obj({"value": msgspec.structs.replace(snapshot, serviceId="studio_remote", nodeId="studio_remote"), "ts": 123}),
        )
        await isolated.ingest(
            "f8/svc/studio_local/nodes/studio_local/data/monitor",
            encode_obj({"value": msgspec.structs.replace(snapshot, serviceId="studio_local", nodeId="studio_local"), "ts": 123}),
        )
        mapped = await isolated.snapshot()
        assert [(item.serviceId, item.nodeId) for item in mapped] == [("studio", "studio")]

    asyncio.run(scenario())
