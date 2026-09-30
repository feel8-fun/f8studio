from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest
from aiortc import AudioStreamTrack, MediaStreamTrack, RTCPeerConnection, RTCSessionDescription

from f8media_gateway.session_manager import SessionManager


class Hub:
    def __init__(self, source: str) -> None:
        self.source = source
        self.closed = False

    def start(self) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


class Peer(RTCPeerConnection):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.fail = False
        self.state = "connected"

    @property
    def connectionState(self) -> str:
        return self.state

    async def setRemoteDescription(self, description: RTCSessionDescription) -> None:
        self.entered.set()
        await self.release.wait()
        if self.fail:
            raise ValueError("invalid offer")


@dataclass
class Session:
    hub: Hub
    session_id: str
    peer: RTCPeerConnection
    track: MediaStreamTrack


def test_shutdown_cancels_inflight_negotiation_and_reclaims_resources() -> None:
    async def scenario() -> None:
        hub, peer, track = Hub("test"), Peer(), AudioStreamTrack()
        manager = SessionManager[Hub, Session](hub_factory=lambda _: hub, disconnected_grace_s=2, kind="test")
        with patch("f8media_gateway.session_manager.RTCPeerConnection", return_value=peer):
            creating = asyncio.create_task(
                manager._create_session(
                    source="test",
                    sdp="offer",
                    offer_type="offer",
                    build=lambda h, sid, p: Session(h, sid, p, track),
                )
            )
            await peer.entered.wait()
            await manager.close()
            with pytest.raises(asyncio.CancelledError):
                await creating
        assert hub.closed and track.readyState == "ended"
        assert peer.signalingState == "closed"
        assert manager.session_count == manager.source_count == 0
        await manager.close()
        with pytest.raises(RuntimeError, match="closed"):
            await manager._acquire_hub("late")

    asyncio.run(scenario())


def test_failed_negotiation_releases_track_peer_and_source() -> None:
    async def scenario() -> None:
        hub, peer, track = Hub("test"), Peer(), AudioStreamTrack()
        peer.fail = True
        peer.release.set()
        manager = SessionManager[Hub, Session](hub_factory=lambda _: hub, disconnected_grace_s=2, kind="test")
        with patch("f8media_gateway.session_manager.RTCPeerConnection", return_value=peer):
            with pytest.raises(ValueError, match="invalid offer"):
                await manager._create_session(
                    source="test", sdp="offer", offer_type="offer", build=lambda h, sid, p: Session(h, sid, p, track)
                )
        assert hub.closed and track.readyState == "ended"
        assert peer.signalingState == "closed"
        assert manager.source_count == 0
        await manager.close()

    asyncio.run(scenario())


def test_disconnection_grace_resets_after_reconnection() -> None:
    async def scenario() -> None:
        hub, peer, track = Hub("test"), Peer(), AudioStreamTrack()
        manager = SessionManager[Hub, Session](hub_factory=lambda _: hub, disconnected_grace_s=2, kind="test")
        await manager._acquire_hub("test")
        manager._sessions["s"] = Session(hub, "s", peer, track)
        peer.state = "disconnected"
        await manager._reap_sessions(10)
        peer.state = "connected"
        await manager._reap_sessions(11)
        peer.state = "disconnected"
        await manager._reap_sessions(12)
        await manager._reap_sessions(13)
        assert manager.session_count == 1
        await manager._reap_sessions(14)
        assert manager.session_count == manager.source_count == 0
        assert hub.closed
        await manager.close()

    asyncio.run(scenario())


def test_shutdown_attempts_all_sessions_when_one_peer_close_fails() -> None:
    class BrokenPeer(Peer):
        async def close(self) -> None:
            await super().close()
            raise OSError("peer cleanup failed")

    async def scenario() -> None:
        hubs: list[Hub] = []

        def make_hub(source: str) -> Hub:
            hub = Hub(source)
            hubs.append(hub)
            return hub

        manager = SessionManager[Hub, Session](hub_factory=make_hub, disconnected_grace_s=2, kind="test")
        peers = [BrokenPeer(), Peer()]
        tracks = [AudioStreamTrack(), AudioStreamTrack()]
        for index, (peer, track) in enumerate(zip(peers, tracks, strict=True)):
            sid = str(index)
            hub = await manager._acquire_hub(sid)
            manager._sessions[sid] = Session(hub, sid, peer, track)
        with pytest.raises(ExceptionGroup, match="shutdown failed"):
            await manager.close()
        assert manager.session_count == manager.source_count == 0
        assert all(hub.closed for hub in hubs)
        assert all(track.readyState == "ended" for track in tracks)
        assert all(peer.signalingState == "closed" for peer in peers)

    asyncio.run(scenario())
