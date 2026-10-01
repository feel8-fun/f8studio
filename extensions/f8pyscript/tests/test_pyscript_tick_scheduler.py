import asyncio

from f8pyscript.tick_scheduler import PyScriptTickScheduler


def test_coerce_tick_ms_uses_positive_integer_interval() -> None:
    assert PyScriptTickScheduler.coerce_tick_ms(20, default=100) == 20
    assert PyScriptTickScheduler.coerce_tick_ms("30", default=100) == 30
    assert PyScriptTickScheduler.coerce_tick_ms(None, default=100) == 100
    assert PyScriptTickScheduler.coerce_tick_ms("bad", default=50) == 50
    assert PyScriptTickScheduler.coerce_tick_ms(0, default=100) == 1
    assert PyScriptTickScheduler.coerce_tick_ms(-10, default=100) == 1


def test_scheduler_runs_ticks_and_shutdown_cancels_task() -> None:
    async def _run() -> tuple[list[dict[str, int]], list[str]]:
        current_ts_ms = 1000
        payloads: list[dict[str, int]] = []
        errors: list[str] = []

        def _now_ms() -> int:
            nonlocal current_ts_ms
            current_ts_ms += 10
            return current_ts_ms

        async def _run_tick(payload: dict[str, int]) -> None:
            payloads.append(dict(payload))

        scheduler = PyScriptTickScheduler(
            node_id="svcA",
            tick_enabled=True,
            tick_ms=1,
            now_ms=_now_ms,
            is_closing=lambda: False,
            is_tick_allowed=lambda: True,
            run_tick=_run_tick,
            log_error=lambda stage, message, exc: errors.append(f"{stage}:{message}:{type(exc).__name__}"),
        )
        scheduler.ensure_task()

        for _ in range(20):
            if len(payloads) >= 2:
                break
            await asyncio.sleep(0.01)

        await scheduler.shutdown()
        return payloads, errors

    payloads, errors = asyncio.run(_run())

    assert errors == []
    assert len(payloads) >= 2
    assert payloads[0] == {"seq": 1, "tsMs": 1020, "deltaMs": 10}
    assert payloads[1] == {"seq": 2, "tsMs": 1030, "deltaMs": 10}
