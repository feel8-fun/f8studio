import asyncio

from f8studio_server.events import EventJournal


def test_event_journal_replays_reliable_events_from_cursor() -> None:
    async def scenario() -> None:
        journal = EventJournal(server_epoch="epoch1", retention=4)
        first = await journal.publish(event_type="graph.committed", scope="project:p1", payload={"revision": 1})
        second = await journal.publish(event_type="graph.committed", scope="project:p1", payload={"revision": 2})

        stream = await journal.open_stream(client_epoch="epoch1", after_sequence=first.sequence)

        assert stream.snapshot_required is False
        assert stream.replay == (second,)
        await journal.close_stream(stream.subscription_id)

    asyncio.run(scenario())


def test_event_journal_requires_snapshot_for_expired_cursor_and_bounds_slow_client() -> None:
    async def scenario() -> None:
        journal = EventJournal(server_epoch="epoch1", retention=2, subscriber_queue_size=1)
        await journal.publish(event_type="one", scope="server", payload={})
        await journal.publish(event_type="two", scope="server", payload={})
        await journal.publish(event_type="three", scope="server", payload={})
        expired = await journal.open_stream(client_epoch="epoch1", after_sequence=0)
        assert expired.snapshot_required is True
        await journal.close_stream(expired.subscription_id)

        slow = await journal.open_stream(client_epoch=None, after_sequence=None)
        await journal.publish(event_type="four", scope="server", payload={})
        await journal.publish(event_type="five", scope="server", payload={})
        reset = slow.queue.get_nowait()
        assert reset is None
        await journal.close_stream(slow.subscription_id)

    asyncio.run(scenario())


def test_overflow_can_replay_from_last_processed_cursor_without_loss() -> None:
    async def scenario() -> None:
        journal = EventJournal(server_epoch="epoch1", subscriber_queue_size=1)
        stream = await journal.open_stream(client_epoch=None, after_sequence=None)
        first = await journal.publish(event_type="graph.committed", scope="project:p", payload={})
        second = await journal.publish(event_type="graph.committed", scope="project:p", payload={})
        assert stream.queue.get_nowait() is None
        replay = await journal.open_stream(client_epoch="epoch1", after_sequence=0)
        assert replay.replay == (first, second)
        assert not replay.snapshot_required
        await journal.close_stream(replay.subscription_id)

    asyncio.run(scenario())


def test_log_history_retains_recent_service_output_and_deployment_errors() -> None:
    async def scenario() -> None:
        journal = EventJournal(server_epoch="epoch1", log_retention=2)
        await journal.publish(event_type="graph.committed", scope="project:p1", payload={})
        await journal.publish(event_type="service.log", scope="service:capture", payload={"line": "started"})
        failure = await journal.publish(
            event_type="deploy.finished", scope="project:p1", payload={"status": "failed"},
        )
        latest = await journal.publish(
            event_type="runtime.error", scope="server", payload={"message": "endpoint timed out"},
        )

        assert await journal.recent_logs() == (failure, latest)
        assert await journal.recent_logs(limit=1) == (latest,)
        assert await journal.recent_logs(limit=1, before_sequence=latest.sequence) == (failure,)
        assert await journal.recent_logs(before_sequence=failure.sequence) == ()

    asyncio.run(scenario())
