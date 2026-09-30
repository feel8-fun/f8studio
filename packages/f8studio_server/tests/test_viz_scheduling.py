from __future__ import annotations

import asyncio
import logging

from f8studio_server.studio_runtime.operators._throttled_flusher import ThrottledFlusher
from f8studio_server.studio_runtime.operators._viz_base import StudioVizRuntimeNodeBase


def test_refresh_burst_coalesces_and_close_prevents_late_flush() -> None:
    async def scenario() -> None:
        calls: list[int] = []
        emitted = asyncio.Event()

        async def flush(now: int) -> None:
            calls.append(now)
            emitted.set()

        flusher = ThrottledFlusher(flush, name="test-refresh")
        await flusher.schedule(now_ms=100, throttle_ms=20)
        emitted.clear()
        for now in range(101, 110):
            await flusher.schedule(now_ms=now, throttle_ms=20)
        await asyncio.wait_for(emitted.wait(), 1)
        assert len(calls) == 2
        await flusher.schedule(now_ms=calls[-1] + 1, throttle_ms=100)
        await flusher.close()
        await flusher.schedule(now_ms=calls[-1] + 200, throttle_ms=0)
        assert len(calls) == 2

    asyncio.run(scenario())


def test_immediate_flush_replaces_pending_and_close_waits_for_callback() -> None:
    async def scenario() -> None:
        calls: list[int] = []
        entered, release = asyncio.Event(), asyncio.Event()

        async def flush(now: int) -> None:
            calls.append(now)
            if now == 200:
                entered.set()
                await release.wait()

        flusher = ThrottledFlusher(flush, name="test-refresh")
        await flusher.schedule(now_ms=100, throttle_ms=50)
        await flusher.schedule(now_ms=101, throttle_ms=50)
        immediate = asyncio.create_task(flusher.schedule(now_ms=200, throttle_ms=50))
        await entered.wait()
        closing = asyncio.create_task(flusher.close())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        await asyncio.gather(immediate, closing)
        assert calls == [100, 200]

    asyncio.run(scenario())


def test_delayed_failure_logs_traceback_and_recovers(caplog) -> None:
    async def scenario() -> None:
        calls = 0
        failed = asyncio.Event()

        async def flush(now: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                failed.set()
                raise ValueError("refresh failed")

        flusher = ThrottledFlusher(flush, name="test-refresh")
        await flusher.schedule(now_ms=100, throttle_ms=5)
        await flusher.schedule(now_ms=101, throttle_ms=5)
        await asyncio.wait_for(failed.wait(), 1)
        await asyncio.sleep(0)
        await flusher.schedule(now_ms=200, throttle_ms=5)
        await flusher.close()
        assert calls == 3

    with caplog.at_level(logging.ERROR):
        asyncio.run(scenario())
    assert any(record.exc_info and "visualization refresh failed" in record.message for record in caplog.records)


def test_shared_config_parses_false_clamps_and_rejects_nonfinite() -> None:
    async def scenario() -> None:
        node = StudioVizRuntimeNodeBase(
            node_id="test",
            data_in_ports=[],
            data_out_ports=[],
            state_fields=[],
            initial_state={"enabled": "false", "limit": 999, "value": "nan", "nonfinite": float("inf")},
        )
        assert await node._get_bool_state("enabled", default=True) is False
        assert await node._get_int_state("limit", default=10, minimum=0, maximum=100) == 100
        assert await node._get_float_state_optional("value") is None
        assert await node._get_int_state("nonfinite", default=10, minimum=0, maximum=100) == 10
        assert await node.get_upstream_sampling_mode() == "auto"

    asyncio.run(scenario())
