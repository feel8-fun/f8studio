import asyncio

from f8studio_server.live import LiveValueHub


def test_live_snapshot_coalescing_and_deletion() -> None:
    async def scenario() -> None:
        hub = LiveValueHub()
        hub.set("node/value", 1)
        subscriber, snapshot = hub.subscribe()
        assert snapshot == {"node/value": 1}
        hub.set("node/value", 2)
        hub.set("node/value", 3)
        assert await subscriber.next_patch() == {"type": "live.patch", "set": {"node/value": 3}, "delete": []}
        hub.delete_prefix("node/")
        assert await subscriber.next_patch() == {"type": "live.patch", "set": {}, "delete": ["node/value"]}
        hub.unsubscribe(subscriber)

    asyncio.run(scenario())


def test_slow_live_subscriber_is_bounded_even_when_keys_churn() -> None:
    async def scenario() -> None:
        hub = LiveValueHub(max_keys=2)
        subscriber, _ = hub.subscribe()
        for index in range(100):
            hub.set(str(index), index)
        assert len(subscriber.pending) + len(subscriber.deleted) <= 2
        assert await subscriber.next_patch() == {"type": "live.snapshot", "values": {"98": 98, "99": 99}}

    asyncio.run(scenario())


def test_presentation_full_values_do_not_enter_reliable_journal_and_detach_clears_snapshot() -> None:
    from f8studio_server.events import EventJournal
    from f8studio_server.studio_runtime.presentation import EventPresentationOutlet

    async def scenario() -> None:
        events = EventJournal(server_epoch="epoch")
        outlet = EventPresentationOutlet(events)
        for index in range(100):
            outlet.emit("node", "viz.wave.set", {"value": index})
        stream = await events.open_stream(client_epoch="epoch", after_sequence=0)
        assert stream.replay == ()
        subscriber, snapshot = events.live.subscribe()
        assert snapshot["presentation/node/viz.wave.set"]["payload"] == {"value": 99}
        outlet.emit("node", "viz.wave.detach", {})
        assert (await subscriber.next_patch())["delete"] == ["presentation/node/viz.wave.set"]
        await outlet.close()

    asyncio.run(scenario())


def test_runtime_state_live_mapping_uses_logical_studio_identity() -> None:
    from f8pysdk.codec import encode_obj
    from f8studio_server.runtime import ZenohRuntimeGateway

    async def scenario() -> None:
        hub = LiveValueHub()
        runtime = ZenohRuntimeGateway(live=hub, studio_service_id="studio_current")
        await runtime._ingest_state("f8/svc/studio_current/state/nodes/studio_current/state/active", encode_obj({"value": True, "ts": 123}))
        await runtime._ingest_state("f8/svc/studio_other/state/nodes/studio_other/state/active", encode_obj({"value": False, "ts": 124}))
        subscription, values = hub.subscribe()
        assert values == {"state/studio/studio/active": {"field": "active", "found": True, "value": True, "tsMs": 123}}
        hub.unsubscribe(subscription)

    asyncio.run(scenario())


def test_presentation_snapshot_uses_live_authority_and_preserves_extensions() -> None:
    from f8studio_server.events import EventJournal
    from f8studio_server.studio_runtime.presentation import EventPresentationOutlet

    async def scenario() -> None:
        events = EventJournal(server_epoch="epoch")
        outlet = EventPresentationOutlet(events)
        outlet.emit("node", "viz.text.update", {"value": "latest"})
        outlet.emit("extension", "custom.append", {"value": "event"})
        assert {item.node_id for item in outlet.snapshot()} == {"node", "extension"}
        events.live.delete_prefix("presentation/node/")
        assert [item.node_id for item in outlet.snapshot()] == ["extension"]
        outlet.emit("extension", "custom.detach", {})
        assert outlet.snapshot() == ()
        await outlet.close()

    asyncio.run(scenario())
