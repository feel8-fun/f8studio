"""Shared RTC negotiation, source ownership and session shutdown."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Generic, Protocol, TypeVar
from uuid import uuid4

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from f8media_protocol.models import MediaInputError

from .hub_pool import HubPool, SourceHub
from .peer_lifecycle import PeerCloseQueue

logger = logging.getLogger(__name__)


class ManagedSession(Protocol):
    @property
    def session_id(self) -> str: ...

    @property
    def hub(self) -> SourceHub: ...

    @property
    def peer(self) -> RTCPeerConnection: ...

    @property
    def track(self) -> MediaStreamTrack: ...


H = TypeVar("H", bound=SourceHub)
S = TypeVar("S", bound=ManagedSession)


class SessionManager(Generic[H, S]):
    def __init__(self, *, hub_factory: Callable[[str], H], disconnected_grace_s: float, kind: str) -> None:
        if disconnected_grace_s < 0:
            raise ValueError("disconnected grace period must be non-negative")
        self._kind = kind
        self._sessions: dict[str, S] = {}
        self._hubs = HubPool(hub_factory)
        self._lock = asyncio.Lock()
        self._janitor: asyncio.Task[None] | None = None
        self._disconnected_grace_s = disconnected_grace_s
        self._disconnected_since: dict[str, float] = {}
        self._peer_closer = PeerCloseQueue()
        self._pending: set[asyncio.Task[tuple[S, RTCSessionDescription]]] = set()
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    @property
    def source_count(self) -> int:
        return self._hubs.source_count

    async def _acquire_hub(self, source: str) -> H:
        if self._closing:
            raise RuntimeError("media session manager is closed")
        return await self._hubs.acquire(source)

    async def _release_hub(self, hub: SourceHub) -> None:
        await self._hubs.release(hub)

    async def _create_session(
        self, *, source: str, sdp: str, offer_type: str, build: Callable[[H, str, RTCPeerConnection], S]
    ) -> tuple[S, RTCSessionDescription]:
        if self._closing:
            raise RuntimeError("media session manager is closed")
        if offer_type != "offer" or not sdp.strip():
            raise MediaInputError("a non-empty WebRTC offer SDP is required")
        task = asyncio.create_task(self._negotiate(source, sdp, build), name=f"{self._kind}-session-negotiate")
        self._pending.add(task)
        try:
            return await task
        finally:
            self._pending.discard(task)

    async def _negotiate(
        self, source: str, sdp: str, build: Callable[[H, str, RTCPeerConnection], S]
    ) -> tuple[S, RTCSessionDescription]:
        hub = await self._acquire_hub(source)
        peer: RTCPeerConnection | None = None
        session: S | None = None
        registered = False
        session_id = uuid4().hex
        try:
            peer = RTCPeerConnection()
            self._observe_peer(peer, source=source, session_id=session_id)
            session = build(hub, session_id, peer)
            peer.addTrack(session.track)
            await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
            local = peer.localDescription
            async with self._lock:
                if self._closing:
                    raise RuntimeError("media session manager closed during negotiation")
                self._sessions[session_id] = session
                registered = True
                if self._janitor is None:
                    self._janitor = asyncio.create_task(self._run_janitor(), name=f"{self._kind}-session-janitor")
                    self._janitor.add_done_callback(self._janitor_finished)
            return session, local
        except Exception:
            logger.exception("failed to negotiate %s session source=%s session_id=%s", self._kind, source, session_id)
            raise
        finally:
            if not registered:
                try:
                    if session is not None:
                        session.track.stop()
                    if peer is not None:
                        await peer.close()
                finally:
                    await self._release_hub(hub)

    def _observe_peer(self, peer: RTCPeerConnection, *, source: str, session_id: str) -> None:
        @peer.on("iceconnectionstatechange")
        async def ice_changed() -> None:
            logger.info(
                "%s ICE state changed session_id=%s source=%s state=%s",
                self._kind,
                session_id,
                source,
                peer.iceConnectionState,
            )

        @peer.on("connectionstatechange")
        async def connection_changed() -> None:
            logger.info(
                "%s peer state changed session_id=%s source=%s state=%s",
                self._kind,
                session_id,
                source,
                peer.connectionState,
            )

    def _session_closed(self, session: S) -> None:
        """Hook for media-specific accounting before a track is stopped."""

    async def close_session(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            self._disconnected_since.pop(session_id, None)
        if session is None:
            return False
        try:
            try:
                self._session_closed(session)
                session.track.stop()
            finally:
                await self._peer_closer.close(session.peer, context=f"{self._kind}:{session_id}")
        finally:
            await self._release_hub(session.hub)
        return True

    async def _reap_sessions(self, now: float) -> None:
        async with self._lock:
            stale: list[str] = []
            for session_id, session in self._sessions.items():
                state = session.peer.connectionState
                if session.hub.closed or state in {"failed", "closed"}:
                    stale.append(session_id)
                elif state == "disconnected":
                    disconnected_at = self._disconnected_since.setdefault(session_id, now)
                    if now - disconnected_at >= self._disconnected_grace_s:
                        stale.append(session_id)
                else:
                    self._disconnected_since.pop(session_id, None)
        for session_id in stale:
            await self.close_session(session_id)

    async def _run_janitor(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            await self._reap_sessions(asyncio.get_running_loop().time())

    def _janitor_finished(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error("media session cleanup failed kind=%s", self._kind, exc_info=error)

    async def close(self) -> None:
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close_all(), name=f"{self._kind}-sessions-close")
        await asyncio.shield(self._close_task)

    async def _close_all(self) -> None:
        tasks: list[asyncio.Task[object]] = list(self._pending)
        if self._janitor is not None:
            tasks.append(self._janitor)
            self._janitor = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        async with self._lock:
            session_ids = tuple(self._sessions)
        results: list[bool | None | BaseException] = list(
            await asyncio.gather(*(self.close_session(sid) for sid in session_ids), return_exceptions=True)
        )
        results.extend(await asyncio.gather(self._hubs.close(), self._peer_closer.shutdown(), return_exceptions=True))
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise BaseExceptionGroup("media session shutdown failed", errors)
