import asyncio
import threading
from dataclasses import dataclass

from f8media_gateway.hub_pool import HubPool


@dataclass
class Hub:
    source: str
    closed: bool = False
    closes: int = 0

    def start(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True
        self.closes += 1


def test_failed_hub_replacement_keeps_old_leases_separate() -> None:
    async def scenario() -> None:
        pool = HubPool(Hub)
        old = await pool.acquire("source")
        assert await pool.acquire("source") is old
        old.closed = True
        replacement = await pool.acquire("source")
        assert replacement is not old
        await pool.release(old)
        await pool.release(old)
        assert old.closes == 1
        assert not replacement.closed
        assert await pool.acquire("source") is replacement
        await pool.release(replacement)
        assert not replacement.closed
        await pool.release(replacement)
        assert replacement.closes == 1
        assert pool.source_count == 0
        await pool.close()

    asyncio.run(scenario())


def test_source_open_does_not_block_event_loop_or_other_sources() -> None:
    entered = threading.Event()
    unblock = threading.Event()

    def factory(source: str) -> Hub:
        if source == "slow":
            entered.set()
            if not unblock.wait(timeout=5):
                raise TimeoutError("test source construction was not released")
        return Hub(source)

    async def scenario() -> None:
        pool = HubPool(factory)
        slow = asyncio.create_task(pool.acquire("slow"))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            fast = await asyncio.wait_for(pool.acquire("fast"), timeout=1)
            assert fast.source == "fast"
            unblock.set()
            await slow
        finally:
            unblock.set()
            await slow
            await pool.close()

    asyncio.run(scenario())


def test_concurrent_consumers_share_one_source_instance() -> None:
    async def scenario() -> None:
        pool = HubPool(Hub)
        hubs = await asyncio.gather(*(pool.acquire("same") for _ in range(10)))
        assert all(hub is hubs[0] for hub in hubs)
        await asyncio.gather(*(pool.release(hub) for hub in hubs))
        assert hubs[0].closes == 1
        await pool.close()

    asyncio.run(scenario())
