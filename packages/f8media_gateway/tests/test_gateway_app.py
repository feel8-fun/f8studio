import asyncio
import os

from aiortc import RTCPeerConnection
from fastapi.testclient import TestClient

from f8media_gateway.app import create_app
from f8media_protocol.models import MEDIA_API_VERSION
from f8media_gateway.service import InProcessMediaGateway


async def create_offer(kind: str) -> str:
    peer = RTCPeerConnection()
    peer.addTransceiver(kind, direction="recvonly")
    try:
        offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        return peer.localDescription.sdp
    finally:
        await peer.close()


def test_gateway_health_identifies_process_and_protocol() -> None:
    gateway = InProcessMediaGateway(gateway_epoch="test-epoch")

    with TestClient(create_app(gateway=gateway)) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "f8media-gateway",
        "version": "0.1.0",
        "protocolVersion": MEDIA_API_VERSION,
        "gatewayEpoch": "test-epoch",
        "processId": os.getpid(),
    }


def test_gateway_rejects_invalid_video_requests_and_unknown_sessions() -> None:
    with TestClient(create_app()) as client:
        invalid = client.post(
            "/api/media/sessions",
            json={
                "source": "synthetic://bars",
                "quality": "ultra",
                "sdp": "offer",
                "type": "offer",
            },
        )
        missing_video = client.delete("/api/media/sessions/missing")
        missing_audio = client.delete("/api/audio/sessions/missing")

    assert invalid.status_code == 422
    assert invalid.json() == {"detail": "media quality must be thumbnail or main"}
    assert missing_video.status_code == 404
    assert missing_video.json() == {"detail": "media session not found"}
    assert missing_audio.status_code == 404
    assert missing_audio.json() == {"detail": "audio session not found"}


def test_gateway_shutdown_releases_media_resources() -> None:
    gateway = InProcessMediaGateway()
    video_offer = asyncio.run(create_offer("video"))
    audio_offer = asyncio.run(create_offer("audio"))

    with TestClient(create_app(gateway=gateway)) as client:
        video = client.post(
            "/api/media/sessions",
            json={
                "source": "synthetic://bars",
                "quality": "thumbnail",
                "sdp": video_offer,
                "type": "offer",
            },
        )
        audio = client.post(
            "/api/audio/sessions",
            json={"source": "synthetic://tone", "sdp": audio_offer, "type": "offer"},
        )
        assert video.status_code == 201
        assert audio.status_code == 201
        assert gateway.video.session_count == 1
        assert gateway.video.source_count == 1
        assert gateway.audio.session_count == 1
        assert gateway.audio.source_count == 1

    assert gateway.video.session_count == 0
    assert gateway.video.source_count == 0
    assert gateway.audio.session_count == 0
    assert gateway.audio.source_count == 0
