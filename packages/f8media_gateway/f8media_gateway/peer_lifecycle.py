from __future__ import annotations

import asyncio
import logging

from aiortc import RTCPeerConnection


logger = logging.getLogger(__name__)
ICE_CHECK_TIMEOUT_S = 35.0


class PeerCloseQueue:
    def __init__(self) -> None:
        self._pending: dict[asyncio.Task[None], RTCPeerConnection] = {}

    async def close(self, peer: RTCPeerConnection, *, context: str) -> None:
        if peer.connectionState != "connecting":
            await peer.close()
            return
        logger.info("deferring peer close until ICE checks settle context=%s", context)
        task = asyncio.create_task(
            self._close_after_ice(peer, context=context),
            name=f"deferred-peer-close:{context}",
        )
        self._pending[task] = peer
        task.add_done_callback(self._peer_closed)

    async def shutdown(self) -> None:
        tasks = tuple(self._pending)
        peers = tuple(set(self._pending.values()))
        self._pending.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for peer in peers:
            await peer.close()

    async def _close_after_ice(self, peer: RTCPeerConnection, *, context: str) -> None:
        deadline = asyncio.get_running_loop().time() + ICE_CHECK_TIMEOUT_S
        while peer.connectionState == "connecting" and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.1)
        if peer.connectionState == "connecting":
            logger.warning("ICE checks did not settle before deferred peer close context=%s", context)
        await peer.close()

    def _peer_closed(self, task: asyncio.Task[None]) -> None:
        self._pending.pop(task, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("deferred peer close failed", exc_info=error)


__all__ = ["PeerCloseQueue"]
