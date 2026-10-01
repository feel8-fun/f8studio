"""Stop push-only websocket producers as soon as their client disconnects."""
from __future__ import annotations

import asyncio
from collections.abc import Coroutine

from anyio import CancelScope
from starlette.websockets import WebSocket


async def send_until_disconnect(websocket: WebSocket, producer: Coroutine[None, None, None]) -> None:
    async def receive_disconnect() -> None:
        while True:
            message = await websocket.receive()
            if message['type'] == 'websocket.disconnect':
                return

    sender = asyncio.create_task(producer)
    receiver = asyncio.create_task(receive_disconnect())
    try:
        done, _ = await asyncio.wait((sender, receiver), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        sender.cancel()
        receiver.cancel()
        # ASGI test/server hosts can cancel their enclosing AnyIO scope while
        # disconnect cleanup is running. Always join our two owned tasks.
        with CancelScope(shield=True):
            await asyncio.gather(sender, receiver, return_exceptions=True)
